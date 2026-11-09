"""Train, compare, and save the loan-approval pipeline.

What `run_training` does, in order:

1. Load the CSV and validate it (columns, types, missing values).
2. Clean the rows and turn Loan_Status Y/N into 1/0. Drop Loan_ID; keep
   Gender aside for the fairness audit only.
3. Split 80% train / 20% test, stratified so both keep the 69/31 class mix.
4. For each of five models, tune hyperparameters with 5-fold
   cross-validation on the TRAINING rows only.
5. Pick the model with the best cross-validated accuracy.
6. Pick the decision threshold (approve if probability >= threshold) on
   out-of-fold training predictions: the standard one maximises accuracy,
   an optional "cautious" one maximises balanced accuracy.
7. Score the chosen model ONCE on the untouched test set and write the
   reports, charts, feature importance and fairness audit.
8. Save the whole pipeline (cleaning + preprocessing + model) to
   artifacts/best_model.joblib so the app can reuse it without retraining.

The test set is never used to choose a model, a parameter, or the threshold.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier

from src.loan_approval.config import (
    CAUSATION_NOTE,
    CV_FOLDS,
    ENGINEERED_FEATURES,
    EXCLUDED_FEATURES,
    FEATURE_NOTES,
    IDENTIFIER_CANDIDATES,
    RANDOM_STATE,
    RAW_CATEGORICAL,
    RAW_NUMERIC,
    REQUIRED_INPUTS,
    SELECTION_METRIC,
    SELECTION_REASON,
    TEST_SIZE,
    Paths,
)
from src.loan_approval.data_loader import load_dataset
from src.loan_approval.data_validation import (
    find_target_column,
    validate_dataframe,
    write_dataset_information,
    write_validation_report,
)
from src.loan_approval.evaluate import (
    choose_threshold,
    compute_metrics,
    plot_comparison,
    plot_confusion,
    plot_importance,
    plot_precision_recall,
    plot_roc,
    write_model_reports,
)
from src.loan_approval.explainability import (
    native_importance,
    permutation_table,
    shap_importance,
    write_fairness_report,
    write_importance_report,
)
from src.loan_approval.preprocessing import build_pipeline, clean_applications, normalize_target
from src.loan_approval.utils import ValidationError, get_logger

LOGGER = get_logger("loan_approval.train")

SCORING = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if np.isnan(array).all():
        return 0.5, 0.0
    return float(np.nanmean(array)), float(np.nanstd(array))


def _candidates(mode: str) -> list[tuple[str, Any, dict[str, Any] | None, bool]]:
    """Return (name, estimator, search spec, scale numeric columns)."""
    if mode == "fast":
        # Used by unit tests. Full mode is what `python -m src.loan_approval.train` runs.
        return [
            ("DummyClassifier", DummyClassifier(strategy="most_frequent"), None, False),
            (
                "LogisticRegression",
                LogisticRegression(max_iter=200, class_weight="balanced"),
                None,
                True,
            ),
            (
                "DecisionTreeClassifier",
                DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=RANDOM_STATE),
                None,
                False,
            ),
            (
                "RandomForestClassifier",
                RandomForestClassifier(
                    n_estimators=20,
                    max_depth=4,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                None,
                False,
            ),
            (
                "ExtraTreesClassifier",
                ExtraTreesClassifier(
                    n_estimators=20,
                    max_depth=4,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                None,
                False,
            ),
        ]
    return [
        ("DummyClassifier", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE), None, False),
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000),
            {
                "kind": "grid",
                "params": {"model__C": [0.1, 1.0, 10.0]},
            },
            True,
        ),
        (
            "DecisionTreeClassifier",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "kind": "grid",
                "params": {
                    "model__max_depth": [3, 5, 8],
                    "model__min_samples_leaf": [1, 5],
                },
            },
            False,
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "kind": "random",
                "n_iter": 8,
                "params": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [4, 8, 12, None],
                    "model__min_samples_leaf": [1, 3, 5],
                },
            },
            False,
        ),
        (
            "ExtraTreesClassifier",
            ExtraTreesClassifier(random_state=RANDOM_STATE),
            {
                "kind": "random",
                "n_iter": 8,
                "params": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [4, 8, 12, None],
                    "model__min_samples_leaf": [1, 3, 5],
                },
            },
            False,
        ),
    ]


def _prepare_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series | None, dict[str, int]]:
    target = find_target_column(list(frame.columns))
    if target is None:
        raise ValidationError("Training data has no loan-status column.")
    cleaned = clean_applications(frame)
    labels = normalize_target(cleaned[target])
    bad_target = int(labels.isna().sum())
    kept = cleaned.loc[labels.notna()].copy()
    labels = labels.loc[labels.notna()]
    kept["_target"] = labels.to_numpy()
    kept = kept.drop(columns=[target])
    duplicates = int(kept.duplicated().sum())
    kept = kept.loc[~kept.duplicated()].reset_index(drop=True)
    y = (kept.pop("_target") == "Y").astype(int)
    groups = kept.pop("Gender") if "Gender" in kept.columns else None
    drop = [
        column
        for column in kept.columns
        if column in IDENTIFIER_CANDIDATES or str(column).lower().endswith("_id") or str(column).strip() == ""
    ]
    features = kept.drop(columns=drop)
    missing = [column for column in REQUIRED_INPUTS if column not in features.columns]
    if missing:
        raise ValidationError("Training data is missing: " + ", ".join(missing))
    if y.nunique() < 2:
        raise ValidationError("Training data needs both approved and rejected applications.")
    notes = {"bad_target_rows": bad_target, "duplicate_rows": duplicates, "rows_used": int(len(features))}
    return features, y, groups, notes


def _input_schema(train: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    reference: dict[str, Any] = {}
    for column in RAW_NUMERIC:
        values = pd.to_numeric(train[column], errors="coerce")
        median = 0.0 if values.dropna().empty else float(values.median())
        schema.append(
            {
                "name": column,
                "kind": "number",
                "median": median,
                "minimum": None if values.dropna().empty else float(values.min()),
                "maximum": None if values.dropna().empty else float(values.max()),
            }
        )
        reference[column] = median
    for column in RAW_CATEGORICAL:
        options = sorted(train[column].dropna().astype(str).unique().tolist())
        mode = train[column].mode(dropna=True)
        typical = None if mode.empty else str(mode.iloc[0])
        schema.append({"name": column, "kind": "category", "options": options, "mode": typical})
        reference[column] = typical
    return schema, reference


def run_training(
    frame: pd.DataFrame | None = None,
    *,
    mode: str = "full",
    paths: Paths | None = None,
) -> dict[str, Any]:
    if mode not in {"full", "fast"}:
        raise ValueError("mode must be 'full' or 'fast'.")
    paths = paths or Paths()
    paths.ensure()
    source = frame if frame is not None else load_dataset(paths.raw_csv)
    validation = validate_dataframe(source, min_rows=1, require_model_inputs=True)
    write_validation_report(validation, paths.reports)
    if paths.manifest.exists():
        write_dataset_information(
            source,
            json.loads(paths.manifest.read_text(encoding="utf-8")),
            paths.reports,
        )

    features, target, groups, notes = _prepare_rows(source)
    export = features.copy()
    export["Loan_Status"] = np.where(target.to_numpy() == 1, "Y", "N")
    if groups is not None:
        export["Gender"] = groups.to_numpy()
    export.to_csv(paths.processed / "cleaned_applications.csv", index=False)

    minority = int(min((target == 0).sum(), (target == 1).sum()))
    n_splits = min(2 if mode == "fast" else CV_FOLDS, minority)
    if n_splits < 2:
        raise ValidationError("Not enough rows in each class for cross-validation.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    index = np.arange(len(features))
    train_idx, test_idx = train_test_split(
        index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    x_train = features.iloc[train_idx]
    x_test = features.iloc[test_idx]
    y_train = target.iloc[train_idx]
    y_test = target.iloc[test_idx]
    groups_test = None if groups is None else groups.iloc[test_idx]

    fitted: dict[str, Any] = {}
    comparison: list[dict[str, Any]] = []
    # n_jobs=1: a few hundred rows do not repay Windows process startup.
    for name, estimator, search, scale in _candidates(mode):
        pipeline = build_pipeline(estimator, scale=scale)
        params: dict[str, Any] = {}
        if search is None:
            pipeline.fit(x_train, y_train)
            chosen = pipeline
        elif search["kind"] == "grid":
            grid = GridSearchCV(
                pipeline,
                search["params"],
                scoring=SELECTION_METRIC,
                cv=cv,
                n_jobs=1,
                refit=True,
            )
            grid.fit(x_train, y_train)
            chosen = grid.best_estimator_
            params = grid.best_params_
        else:
            random_search = RandomizedSearchCV(
                pipeline,
                search["params"],
                n_iter=search["n_iter"],
                scoring=SELECTION_METRIC,
                cv=cv,
                n_jobs=1,
                random_state=RANDOM_STATE,
                refit=True,
            )
            random_search.fit(x_train, y_train)
            chosen = random_search.best_estimator_
            params = random_search.best_params_
        scores = cross_validate(
            clone(chosen),
            x_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
        )
        row: dict[str, Any] = {"model": name, "params": _json_safe(params)}
        for metric in SCORING:
            mean, std = _mean_std(scores[f"test_{metric}"])
            row[f"cv_{metric}_mean"] = mean
            row[f"cv_{metric}_std"] = std
        comparison.append(row)
        fitted[name] = chosen
        LOGGER.info("%s CV accuracy %.3f", name, row["cv_accuracy_mean"])

    comparison.sort(key=lambda row: (-row["cv_accuracy_mean"], -row["cv_roc_auc_mean"], row["model"]))
    best_name = str(comparison[0]["model"])
    best = fitted[best_name]
    oof = cross_val_predict(clone(best), x_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = choose_threshold(y_train.to_numpy(), oof)
    # A second, optional cut-off that weights both classes equally. It rejects
    # more risky applications at the cost of some accuracy. Also chosen on
    # training out-of-fold scores only.
    cautious_threshold = choose_threshold(y_train.to_numpy(), oof, metric="balanced_accuracy")
    test_proba = best.predict_proba(x_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    metrics = compute_metrics(y_test.to_numpy(), test_pred, test_proba)
    cautious_metrics = compute_metrics(
        y_test.to_numpy(), (test_proba >= cautious_threshold).astype(int), test_proba
    )
    pd.DataFrame(
        {
            "actual_approved": y_test.to_numpy(),
            "approval_probability": test_proba,
            "Credit_History": x_test["Credit_History"].to_numpy(),
        }
    ).to_csv(paths.reports / "test_predictions.csv", index=False)

    class_distribution = {
        "train": {"approved": int((y_train == 1).sum()), "rejected": int((y_train == 0).sum())},
        "test": {"approved": int((y_test == 1).sum()), "rejected": int((y_test == 0).sum())},
    }
    rows_note = (
        f"Excluded {notes['bad_target_rows']} rows with a missing or unrecognised loan status "
        f"and {notes['duplicate_rows']} exact duplicate rows before the split. "
        f"{notes['rows_used']} rows remained. Validation itself did not delete anything."
    )
    write_model_reports(
        comparison=comparison,
        best_model=best_name,
        best_params=comparison[0]["params"],
        threshold=threshold,
        metrics=metrics,
        y_true=y_test.to_numpy(),
        y_pred=test_pred,
        class_distribution=class_distribution,
        rows_note=rows_note,
        report_dir=paths.reports,
    )
    plot_confusion(metrics, paths.reports / "confusion_matrix.png")
    plot_roc(y_test.to_numpy(), test_proba, paths.reports / "roc_curve.png")
    plot_precision_recall(y_test.to_numpy(), test_proba, paths.reports / "precision_recall_curve.png")
    plot_comparison(comparison, paths.reports / "model_comparison.png")

    repeats = 2 if mode == "fast" else 5
    permuted = permutation_table(best, x_train, y_train, repeats=repeats)
    internal = native_importance(best)
    permuted.to_csv(paths.reports / "permutation_importance.csv", index=False)
    plot_importance(
        permuted,
        paths.reports / "permutation_importance.png",
        "Permutation importance (training set)",
    )
    if internal is not None:
        plot_importance(
            internal.rename(columns={"importance_mean": "importance_mean"}),
            paths.reports / "model_feature_importance.png",
            "Internal model importance",
        )
    write_importance_report(permuted, internal, paths.reports)
    shap_table = shap_importance(best, x_train)  # None if SHAP is missing or the model is not a tree
    if shap_table is not None:
        shap_table.to_csv(paths.reports / "shap_importance.csv", index=False)
        plot_importance(
            shap_table,
            paths.reports / "shap_importance.png",
            "SHAP importance (training set)",
        )
    descriptive = features.copy()
    if groups is not None:
        descriptive["Gender"] = groups.to_numpy()
    write_fairness_report(
        y_test.to_numpy(),
        test_pred,
        groups_test,
        paths.reports,
        descriptive=descriptive,
    )

    schema, reference = _input_schema(x_train)
    manifest = {}
    if paths.manifest.exists():
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    metadata = {
        "model_name": best_name,
        "selection_metric": SELECTION_METRIC,
        "selection_reason": SELECTION_REASON,
        "threshold": threshold,
        "threshold_reason": (
            "Cutoff maximises accuracy on training out-of-fold probabilities. "
            "The test set was not used to choose it. "
            "pipeline.predict() still uses 0.5; the app and predict module use this threshold."
        ),
        "cautious_threshold": cautious_threshold,
        "cautious_threshold_reason": (
            "Cutoff maximises balanced accuracy on training out-of-fold probabilities, "
            "so rejected applications count as much as approved ones. It catches more "
            "risky applications but lowers overall accuracy. Optional in the app."
        ),
        "cautious_test_metrics": cautious_metrics,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": n_splits,
        "training_mode": mode,
        "training_date": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "target_column": validation["target_column"],
        "positive_label": "Y",
        "class_labels": {"1": "Approved", "0": "Rejected"},
        "raw_features": list(REQUIRED_INPUTS),
        "engineered_features": list(ENGINEERED_FEATURES),
        "feature_notes": FEATURE_NOTES,
        "excluded_features": list(EXCLUDED_FEATURES),
        "excluded_reason": (
            "Gender is kept for the fairness audit and is not a predictive input. "
            "Loan_ID is an identifier. Dropping Gender does not remove bias."
        ),
        "dataset": {
            "name": manifest.get("dataset_name"),
            "original_source_url": manifest.get("original_source_url"),
            "direct_download_url": manifest.get("direct_download_url"),
            "provider": manifest.get("repository_or_provider"),
            "licence": manifest.get("licence"),
            "download_date": manifest.get("download_date"),
            "sha256": manifest.get("sha256"),
        },
        "best_params": comparison[0]["params"],
        "comparison": comparison,
        "test_metrics": metrics,
        "class_distribution": class_distribution,
        "rows": notes,
        "input_schema": schema,
        "reference_values": reference,
        "causation_note": CAUSATION_NOTE,
        "imbalance_handling": (
            "class_weight is left off so the fit is not pushed away from accuracy. "
            "The dummy most-frequent baseline shows the accuracy you get by always "
            "predicting approval. SMOTE is not used."
        ),
    }
    joblib.dump(best, paths.model, compress=3)
    paths.metadata.write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    LOGGER.info("Saved %s (test ROC-AUC %.3f)", paths.model, metrics["roc_auc"])
    return metadata


def main() -> None:
    from src.loan_approval.utils import configure_logging

    configure_logging()
    metadata = run_training(mode="full")
    metrics = metadata["test_metrics"]
    print(f"Selected model: {metadata['model_name']}")
    print(f"Threshold: {metadata['threshold']:.2f}")
    print(
        "Test metrics: "
        f"accuracy {metrics['accuracy']:.3f}, "
        f"precision {metrics['precision']:.3f}, "
        f"recall {metrics['recall']:.3f}, "
        f"f1 {metrics['f1']:.3f}, "
        f"balanced accuracy {metrics['balanced_accuracy']:.3f}, "
        f"ROC-AUC {metrics['roc_auc']:.3f}, "
        f"PR-AUC {metrics['average_precision']:.3f}"
    )
    cautious = metadata["cautious_test_metrics"]
    print(
        f"Cautious mode (threshold {metadata['cautious_threshold']:.2f}): "
        f"accuracy {cautious['accuracy']:.3f}, precision {cautious['precision']:.3f}, "
        f"rejections caught {cautious['true_negative']} of "
        f"{cautious['true_negative'] + cautious['false_positive']}"
    )
    print("Model comparison (5-fold CV on the training set, ranked by accuracy):")
    for row in metadata["comparison"]:
        print(
            f"  {row['model']:<24} accuracy {row['cv_accuracy_mean']:.3f}   "
            f"ROC-AUC {row['cv_roc_auc_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
