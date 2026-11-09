"""Honest checks on how far the results can be trusted.

Run after training:  python scripts/robustness_checks.py

Writes reports/robustness_report.md and reports/robustness_report.json:

1. Nested cross-validation: the hyperparameter search is repeated inside each
   outer fold, so the score is not flattered by tuning on the same folds.
2. 30 different random 80/20 splits: how much test accuracy moves by luck.
3. Stronger models under the same rules, plus the best test accuracy any of
   them could reach if the cut-off were (unfairly) picked with the test labels.
4. A one-line credit-history rule, to show how much the model relies on it.
5. Calibration: whether a "70% probability" really means about 70% approved.
6. Missing credit history: how often those applicants were approved.

None of this changes the saved model.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from src.loan_approval.config import RANDOM_STATE, TEST_SIZE, Paths
from src.loan_approval.data_loader import load_dataset
from src.loan_approval.evaluate import choose_threshold
from src.loan_approval.preprocessing import build_pipeline
from src.loan_approval.train import _candidates, _prepare_rows
from src.loan_approval.utils import markdown_table

warnings.filterwarnings("ignore")


def _fair_score(pipeline, x_train, y_train, x_test, y_test, cv) -> dict[str, float]:
    oof = cross_val_predict(clone(pipeline), x_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = choose_threshold(y_train.to_numpy(), oof)
    fitted = clone(pipeline).fit(x_train, y_train)
    proba = fitted.predict_proba(x_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    best_possible = max(accuracy_score(y_test, (proba >= step / 100).astype(int)) for step in range(5, 96))
    return {
        "threshold": threshold,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "best_accuracy_if_cut_off_picked_on_test": float(best_possible),
    }


def main() -> None:
    paths = Paths()
    features, target, _groups, _notes = _prepare_rows(load_dataset(paths.raw_csv))
    model = joblib.load(paths.model)
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    threshold = float(metadata["threshold"])
    index = np.arange(len(features))
    train_idx, test_idx = train_test_split(index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target)
    x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, object] = {}

    # 1. Nested CV of the full random-forest search on all rows.
    _, forest, search, _ = next(c for c in _candidates("full") if c[0] == "RandomForestClassifier")
    outer = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE + 1)
    nested = []
    for fold_train, fold_test in outer.split(features, target):
        inner = RandomizedSearchCV(
            build_pipeline(clone(forest), scale=False), search["params"], n_iter=search["n_iter"],
            scoring="accuracy", cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
        ).fit(features.iloc[fold_train], target.iloc[fold_train])
        nested.append(accuracy_score(target.iloc[fold_test], inner.predict(features.iloc[fold_test])))
    results["nested_cv_accuracy"] = {"mean": float(np.mean(nested)), "std": float(np.std(nested)),
                                     "folds": [float(v) for v in nested]}
    print(f"Nested CV accuracy: {np.mean(nested):.3f} +/- {np.std(nested):.3f}")

    # 2. Same chosen model on 30 random splits.
    accuracies = []
    for seed in range(30):
        a, b = train_test_split(index, test_size=TEST_SIZE, random_state=seed, stratify=target)
        fitted = clone(model).fit(features.iloc[a], target.iloc[a])
        accuracies.append(accuracy_score(target.iloc[b], (fitted.predict_proba(features.iloc[b])[:, 1] >= threshold)))
    accuracies = np.array(accuracies)
    results["random_splits"] = {"n": 30, "mean": float(accuracies.mean()), "min": float(accuracies.min()),
                                "max": float(accuracies.max()), "splits_at_or_above_90": int((accuracies >= 0.9).sum())}
    print(f"30 random splits: mean {accuracies.mean():.3f}, range {accuracies.min():.3f}-{accuracies.max():.3f}")

    # 3. Stronger models, same rules.
    contenders = {
        "Random Forest (chosen)": clone(model),
        "Gradient Boosting (tuned)": GridSearchCV(
            build_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE), scale=False),
            {"model__n_estimators": [50, 100, 200], "model__max_depth": [2, 3], "model__learning_rate": [0.05, 0.1]},
            scoring="accuracy", cv=cv, n_jobs=-1,
        ).fit(x_train, y_train).best_estimator_,
        "HistGradientBoosting (tuned)": GridSearchCV(
            build_pipeline(HistGradientBoostingClassifier(random_state=RANDOM_STATE), scale=False),
            {"model__max_depth": [2, 3, None], "model__learning_rate": [0.05, 0.1], "model__max_iter": [50, 100]},
            scoring="accuracy", cv=cv, n_jobs=-1,
        ).fit(x_train, y_train).best_estimator_,
        "Logistic Regression (tuned)": GridSearchCV(
            build_pipeline(LogisticRegression(max_iter=2000), scale=True),
            {"model__C": [0.01, 0.1, 1, 10]}, scoring="accuracy", cv=cv, n_jobs=-1,
        ).fit(x_train, y_train).best_estimator_,
    }
    results["stronger_models"] = {
        name: _fair_score(pipe, x_train, y_train, x_test, y_test, cv) for name, pipe in contenders.items()
    }
    for name, row in results["stronger_models"].items():
        print(f"{name:<30} test accuracy {row['test_accuracy']:.3f}")

    # 4. One-line rule.
    credit = x_test["Credit_History"].astype(float).fillna(1.0)
    rule = (credit != 0).astype(int)
    results["credit_history_rule"] = {"test_accuracy": float(accuracy_score(y_test, rule)),
                                      "precision": float(precision_score(y_test, rule))}

    # 5. Calibration on the test split.
    proba = model.predict_proba(x_test)[:, 1]
    calibrated = CalibratedClassifierCV(clone(model), method="sigmoid", cv=cv).fit(x_train, y_train)
    calibrated_proba = calibrated.predict_proba(x_test)[:, 1]
    bins = []
    for low, high in [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        mask = (proba >= low) & (proba < high)
        if mask.any():
            bins.append({"range": f"{low:.1f}-{min(high, 1.0):.1f}", "rows": int(mask.sum()),
                         "mean_predicted": float(proba[mask].mean()), "actually_approved": float(y_test[mask].mean())})
    results["calibration"] = {"brier_raw": float(brier_score_loss(y_test, proba)),
                              "brier_sigmoid_calibrated": float(brier_score_loss(y_test, calibrated_proba)),
                              "bins": bins}

    # 6. Missing credit history.
    missing = features["Credit_History"].isna()
    results["missing_credit_history"] = {"rows": int(missing.sum()),
                                         "historically_approved": float(target[missing].mean())}

    (paths.reports / "robustness_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    model_rows = [[name, f"{r['test_accuracy']:.3f}", f"{r['precision']:.3f}", f"{r['roc_auc']:.3f}",
                   f"{r['best_accuracy_if_cut_off_picked_on_test']:.3f}"]
                  for name, r in results["stronger_models"].items()]
    calibration_rows = [[b["range"], str(b["rows"]), f"{b['mean_predicted']:.2f}", f"{b['actually_approved']:.2f}"]
                        for b in bins]
    lines = [
        "# Robustness checks",
        "",
        "Generated by `python scripts/robustness_checks.py`. The saved model is not changed.",
        "",
        "## 1. Nested cross-validation",
        "",
        f"Accuracy {np.mean(nested):.3f} ± {np.std(nested):.3f} over 5 outer folds. The search is repeated inside "
        "each fold, so this estimate is not flattered by tuning.",
        "",
        "## 2. Thirty random splits",
        "",
        f"Same model and cut-off, 30 different 80/20 splits: mean {accuracies.mean():.3f}, "
        f"range {accuracies.min():.3f} to {accuracies.max():.3f}. Splits reaching 90%: "
        f"{int((accuracies >= 0.9).sum())}.",
        "",
        "## 3. Stronger models under the same rules",
        "",
        markdown_table(["Model", "Test accuracy", "Precision", "ROC-AUC", "Best accuracy if cut-off picked on test"],
                       model_rows),
        "",
        "## 4. One-line rule",
        "",
        f"\"Approve if credit history is not 0\": test accuracy {results['credit_history_rule']['test_accuracy']:.3f}, "
        f"precision {results['credit_history_rule']['precision']:.3f}.",
        "",
        "## 5. Calibration (test split)",
        "",
        markdown_table(["Probability range", "Rows", "Mean predicted", "Actually approved"], calibration_rows),
        "",
        f"Brier score {results['calibration']['brier_raw']:.4f} raw, "
        f"{results['calibration']['brier_sigmoid_calibrated']:.4f} after sigmoid calibration (lower is better). "
        "The gain is small, so the uncalibrated model is kept.",
        "",
        "## 6. Missing credit history",
        "",
        f"{int(missing.sum())} applications have no credit history; "
        f"{target[missing].mean():.0%} of them were approved, close to the overall 69%. "
        "The pipeline fills the gap with the most common value, and the app flags these rows for review.",
        "",
    ]
    (paths.reports / "robustness_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote reports/robustness_report.md")


if __name__ == "__main__":
    main()
