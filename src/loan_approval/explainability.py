"""Associations the model uses. These are not causes and not a fairness proof."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.loan_approval.config import CAUSATION_NOTE, RANDOM_STATE, RAW_CATEGORICAL
from src.loan_approval.utils import fmt, markdown_table


def permutation_table(
    pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    repeats: int,
) -> pd.DataFrame:
    # Training rows only, so the test set stays unused. The bars can look stronger than they are on new applicants.
    result = permutation_importance(
        pipeline,
        features,
        target,
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": list(features.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def native_importance(pipeline) -> pd.DataFrame | None:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return None
    names = [str(name) for name in pipeline.named_steps["prepare"].get_feature_names_out()]
    if len(names) != len(values):
        return None
    table = pd.DataFrame({"feature": names, "importance_mean": values})
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- SHAP
# SHAP splits one prediction into a contribution per feature, so that
#   base value (average approval probability) + sum of contributions = this probability.
# The one-hot columns of a category (e.g. Property_Area_Rural/Semiurban/Urban)
# are added back together, so each contribution belongs to one form field or
# one engineered feature. Only tree models are supported (TreeExplainer).


def make_shap_explainer(pipeline):
    """Return a SHAP TreeExplainer for the fitted model, or None if unavailable."""
    try:
        import shap
    except ImportError:
        return None
    model = pipeline.named_steps["model"]
    if not hasattr(model, "estimators_") and not hasattr(model, "tree_"):
        return None
    return shap.TreeExplainer(model)


def _group_of(column: str) -> str:
    for category in RAW_CATEGORICAL:
        if column.startswith(category + "_"):
            return category
    return column


def _grouped_shap(pipeline, explainer, frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    prepared = pipeline.named_steps["prepare"].transform(pipeline.named_steps["features"].transform(frame))
    names = [str(name) for name in pipeline.named_steps["prepare"].get_feature_names_out()]
    values = np.asarray(explainer.shap_values(prepared))
    if values.ndim == 3:  # (rows, columns, classes): keep the "approved" class
        values = values[:, :, 1]
    base = np.asarray(explainer.expected_value).ravel()
    base_value = float(base[1] if base.size > 1 else base[0])
    grouped = pd.DataFrame(values, columns=names).T.groupby(_group_of).sum().T
    return grouped, base_value


def shap_explanation(pipeline, explainer, row: pd.DataFrame) -> dict[str, object] | None:
    """SHAP contributions for one applicant, largest first."""
    if explainer is None:
        return None
    grouped, base_value = _grouped_shap(pipeline, explainer, row.iloc[[0]])
    shown = pipeline.named_steps["features"].transform(row.iloc[[0]]).iloc[0]
    items = [
        {"feature": name, "contribution": float(value), "value": shown.get(name)}
        for name, value in grouped.iloc[0].items()
    ]
    items.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return {"base_value": base_value, "items": items}


def shap_importance(pipeline, features: pd.DataFrame) -> pd.DataFrame | None:
    """Mean absolute SHAP contribution per field over the given rows."""
    explainer = make_shap_explainer(pipeline)
    if explainer is None:
        return None
    grouped, _ = _grouped_shap(pipeline, explainer, features)
    table = pd.DataFrame({"feature": grouped.columns, "importance_mean": grouped.abs().mean().to_numpy()})
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def local_explanation(pipeline, row: pd.DataFrame, reference: dict[str, object]) -> list[dict[str, object]]:
    """Swap one field at a time for the training median or mode.

    A positive probability_change means the applicant's own value scores as
    more approvable than that typical value. It is still only an association.
    """
    if not reference or not hasattr(pipeline, "predict_proba"):
        return []
    base = float(pipeline.predict_proba(row)[:, 1][0])
    effects: list[dict[str, object]] = []
    for column, typical in reference.items():
        if column not in row.columns:
            continue
        altered = row.astype(object)
        altered.iloc[0, altered.columns.get_loc(column)] = typical
        changed = float(pipeline.predict_proba(altered)[:, 1][0])
        applicant_value = row.iloc[0][column]
        if pd.isna(applicant_value):
            shown: object = None
        elif isinstance(applicant_value, (int, float, np.floating)) and not isinstance(applicant_value, bool):
            shown = float(applicant_value)
        else:
            shown = str(applicant_value)
        effects.append(
            {
                "feature": column,
                "applicant_value": shown,
                "reference_value": typical,
                "probability_change": base - changed,
            }
        )
    effects.sort(key=lambda item: abs(float(item["probability_change"])), reverse=True)
    return effects


def _rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    def ratio(numerator: int, denominator: int) -> float | None:
        if denominator == 0:
            return None
        return numerator / denominator

    return {
        "rows": int(len(y_true)),
        "false_positive_rate": ratio(fp, fp + tn),
        "false_negative_rate": ratio(fn, fn + tp),
        "predicted_approval_rate": ratio(int((y_pred == 1).sum()), len(y_pred)),
        "historical_approval_rate": ratio(int((y_true == 1).sum()), len(y_true)),
    }


def write_importance_report(
    permutation: pd.DataFrame,
    native: pd.DataFrame | None,
    report_dir: Path,
) -> None:
    lines = [
        "# Feature importance",
        "",
        CAUSATION_NOTE,
        "",
        "Permutation importance shuffles one input column at a time and measures "
        "the drop in training ROC-AUC. Income columns also feed TotalIncome, "
        "LoanToIncome, and AmountPerMonth, so their bars include those engineered uses.",
        "",
        markdown_table(
            ["Feature", "Mean AUC drop", "Std"],
            [
                [row.feature, fmt(float(row.importance_mean), 4), fmt(float(row.importance_std), 4)]
                for row in permutation.itertuples(index=False)
            ],
        ),
        "",
    ]
    if native is not None:
        lines.extend(
            [
                "## Model-internal weights",
                "",
                "Tree importances and absolute logistic coefficients are on the "
                "transformed columns (scaled numbers and one-hot categories). "
                "A one-hot column can look small because the signal is split across levels.",
                "",
                markdown_table(
                    ["Transformed feature", "Importance"],
                    [
                        [row.feature, fmt(float(row.importance_mean), 4)]
                        for row in native.head(15).itertuples(index=False)
                    ],
                ),
                "",
            ]
        )
    (report_dir / "feature_importance.md").write_text("\n".join(lines), encoding="utf-8")


def write_fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series | None,
    report_dir: Path,
    descriptive: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Fairness notes",
        "",
        "Gender is not a model input. Removing it does not remove bias. "
        "Marital status, dependents, income, and property area can still carry "
        "information about gender, and the historical labels can themselves be biased.",
        "",
        "Group rates below are descriptive error counts on the held-out test set. "
        "They were not used to pick the model. Small groups move around a lot; "
        "a gap here is not proof of discrimination, and no gap is not proof of fairness.",
        "",
    ]
    if groups is None:
        lines.append("This file has no Gender column, so group rates were not computed.")
    else:
        labels = groups.fillna("Missing").astype(str).reset_index(drop=True)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        rows = []
        for label in sorted(labels.unique()):
            mask = (labels == label).to_numpy()
            stats = _rates(y_true[mask], y_pred[mask])
            note = "unstable (fewer than 20 test rows)" if int(stats["rows"]) < 20 else ""
            rows.append(
                [
                    label,
                    stats["rows"],
                    fmt(stats["historical_approval_rate"]),
                    fmt(stats["predicted_approval_rate"]),
                    fmt(stats["false_positive_rate"]),
                    fmt(stats["false_negative_rate"]),
                    note,
                ]
            )
        overall = _rates(y_true, y_pred)
        rows.append(
            [
                "All test rows",
                overall["rows"],
                fmt(overall["historical_approval_rate"]),
                fmt(overall["predicted_approval_rate"]),
                fmt(overall["false_positive_rate"]),
                fmt(overall["false_negative_rate"]),
                "",
            ]
        )
        lines.append(
            markdown_table(
                ["Group", "Rows", "Historical approval", "Predicted approval", "FPR", "FNR", "Note"],
                rows,
            )
        )
    if descriptive is not None and "Gender" in descriptive.columns and "ApplicantIncome" in descriptive.columns:
        lines.extend(["", "## Descriptive association in the cleaned table", ""])
        grouped = (
            descriptive.assign(_gender=descriptive["Gender"].fillna("Missing").astype(str))
            .groupby("_gender")["ApplicantIncome"]
            .mean()
        )
        lines.append("Mean applicant income by gender (not used as a training target):")
        lines.append("")
        lines.append(
            markdown_table(
                ["Gender", "Mean applicant income"],
                [[index, f"{value:.0f}"] for index, value in grouped.items()],
            )
        )
        lines.append("")
        lines.append("An income gap can make a model that never sees gender still score groups differently.")
    lines.append("")
    (report_dir / "fairness_report.md").write_text("\n".join(lines), encoding="utf-8")
