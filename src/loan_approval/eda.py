"""Charts and a findings report computed from the loaded table."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.loan_approval.config import Paths
from src.loan_approval.data_loader import load_dataset
from src.loan_approval.data_validation import (
    load_manifest,
    validate_dataframe,
    write_dataset_information,
    write_validation_report,
)
from src.loan_approval.preprocessing import add_engineered_features, clean_applications, normalize_target
from src.loan_approval.utils import fmt, markdown_table

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _approved_flag(frame: pd.DataFrame, target: str) -> pd.Series:
    return normalize_target(frame[target]).eq("Y")


def _iqr_outlier_count(series: pd.Series) -> int:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    spread = q3 - q1
    low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
    return int(((values < low) | (values > high)).sum())


def _approval_by(frame: pd.DataFrame, column: str, approved: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame({column: frame[column].astype("object"), "approved": approved})
    table = table.dropna()
    grouped = table.groupby(column)["approved"].agg(approval_rate="mean", rows="count")
    return grouped.sort_values("rows", ascending=False)


def save_plots(frame: pd.DataFrame, target: str, report_dir: Path) -> None:
    approved = _approved_flag(frame, target)
    known = normalize_target(frame[target]).notna()

    fig, axis = plt.subplots(figsize=(5, 4))
    axis.bar(
        ["Rejected", "Approved"],
        [int((~approved & known).sum()), int((approved & known).sum())],
        color=["#C0392B", "#1B4F72"],
    )
    axis.set_ylabel("Applications")
    axis.set_title("Loan status")
    _save(fig, report_dir / "target_distribution.png")

    missing = frame.isna().sum().sort_values(ascending=False)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(missing.index.astype(str), missing.values, color="#1B4F72")
    axis.set_ylabel("Missing rows")
    axis.set_title("Missing values")
    axis.tick_params(axis="x", rotation=45)
    _save(fig, report_dir / "missing_values.png")

    numeric = [
        column
        for column in ("ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term")
        if column in frame.columns
    ]
    if numeric:
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for axis, column in zip(axes.ravel(), numeric):
            sns.histplot(pd.to_numeric(frame[column], errors="coerce").dropna(), ax=axis, color="#1B4F72")
            axis.set_title(column)
        for axis in axes.ravel()[len(numeric) :]:
            axis.axis("off")
        _save(fig, report_dir / "histograms.png")

        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        plot_frame = frame.loc[known].copy()
        plot_frame["_status"] = np.where(approved[known], "Approved", "Rejected")
        for axis, column in zip(axes.ravel(), numeric):
            sns.boxplot(data=plot_frame, x="_status", y=column, ax=axis, color="#5DADE2")
            axis.set_title(column)
        for axis in axes.ravel()[len(numeric) :]:
            axis.axis("off")
        _save(fig, report_dir / "boxplots.png")

    if "ApplicantIncome" in frame.columns:
        fig, axis = plt.subplots(figsize=(6, 4))
        sns.histplot(pd.to_numeric(frame["ApplicantIncome"], errors="coerce").dropna(), ax=axis, color="#1B4F72")
        axis.set_title("Applicant income")
        _save(fig, report_dir / "income_distribution.png")
    if "LoanAmount" in frame.columns:
        fig, axis = plt.subplots(figsize=(6, 4))
        sns.histplot(pd.to_numeric(frame["LoanAmount"], errors="coerce").dropna(), ax=axis, color="#148F77")
        axis.set_title("Loan amount")
        _save(fig, report_dir / "loan_amount_distribution.png")

    heat = pd.DataFrame()
    for column in ("ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"):
        if column in frame.columns:
            heat[column] = pd.to_numeric(frame[column], errors="coerce")
    heat["Approved"] = approved.astype(float)
    heat = heat.dropna(how="all")
    if heat.shape[1] > 1:
        fig, axis = plt.subplots(figsize=(7, 5))
        sns.heatmap(heat.corr(numeric_only=True), annot=True, fmt=".2f", cmap="Blues", ax=axis)
        axis.set_title("Correlation (numeric fields and approval)")
        _save(fig, report_dir / "correlation_heatmap.png")

    categorical = [
        column
        for column in (
            "Education",
            "Self_Employed",
            "Dependents",
            "Property_Area",
            "Credit_History",
            "Married",
            "Gender",
        )
        if column in frame.columns
    ]
    if categorical:
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for axis, column in zip(axes.ravel(), categorical):
            rates = _approval_by(frame, column, approved)
            axis.bar(rates.index.astype(str), rates["approval_rate"], color="#1B4F72")
            axis.set_ylim(0, 1)
            axis.set_title(column)
            axis.tick_params(axis="x", rotation=30)
        for axis in axes.ravel()[len(categorical) :]:
            axis.axis("off")
        _save(fig, report_dir / "categorical_vs_status.png")
        focus = [column for column in ("Credit_History", "Property_Area", "Education", "Married") if column in frame.columns]
        if focus:
            fig, axes = plt.subplots(1, len(focus), figsize=(4 * len(focus), 4))
            if len(focus) == 1:
                axes = [axes]
            for axis, column in zip(axes, focus):
                rates = _approval_by(frame, column, approved)
                axis.bar(rates.index.astype(str), rates["approval_rate"], color="#148F77")
                axis.set_ylim(0, 1)
                axis.set_title(f"Approval rate by {column}")
                axis.tick_params(axis="x", rotation=30)
            _save(fig, report_dir / "features_vs_status.png")


def _gap_sentence(frame: pd.DataFrame, column: str, approved: pd.Series) -> str | None:
    if column not in frame.columns:
        return None
    rates = _approval_by(frame, column, approved)
    usable = rates[rates["rows"] >= 20]
    if len(usable) < 2:
        return f"{column}: not enough rows in two groups to compare approval rates."
    high = usable["approval_rate"].idxmax()
    low = usable["approval_rate"].idxmin()
    gap = float(usable.loc[high, "approval_rate"] - usable.loc[low, "approval_rate"])
    return (
        f"{column}: approval is {float(usable.loc[high, 'approval_rate']):.1%} "
        f"for {high} ({int(usable.loc[high, 'rows'])} rows) and "
        f"{float(usable.loc[low, 'approval_rate']):.1%} for {low} "
        f"({int(usable.loc[low, 'rows'])} rows), a gap of {gap:.1%}."
    )


def build_eda_markdown(frame: pd.DataFrame, validation: dict[str, object]) -> str:
    target = validation["target_column"]
    lines = [
        "# Exploratory data analysis",
        "",
        "Every number in this file was calculated from the dataset on disk.",
        "",
        "## Dataset overview",
        "",
        f"- Records: **{len(frame)}**",
        f"- Columns: **{frame.shape[1]}**",
        f"- Duplicate records: **{int(frame.duplicated().sum())}**",
        f"- Target column: **{target}**",
        "",
        "### Data types",
        "",
        markdown_table(["Column", "Type"], [[column, str(dtype)] for column, dtype in frame.dtypes.items()]),
        "",
        "### Numeric summary",
        "",
    ]
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        lines.append("No numeric columns were detected before cleaning.")
    else:
        summary = numeric.describe().T.reset_index()
        headers = [str(column) for column in summary.columns]
        lines.append(
            markdown_table(
                headers,
                [[fmt(value, 2) if isinstance(value, float) else value for value in row] for row in summary.to_numpy()],
            )
        )
    lines.extend(["", "### Missing values", ""])
    missing = frame.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        lines.append("No missing values.")
    else:
        lines.append(markdown_table(["Column", "Missing", "Percent"], [
            [column, int(count), f"{count / len(frame):.1%}"] for column, count in missing.items()
        ]))

    if target is None:
        lines.extend(["", "No target column was found, so approval comparisons were skipped.", ""])
        return "\n".join(lines)

    approved = _approved_flag(frame, target)
    known = normalize_target(frame[target]).notna()
    approved_n = int((approved & known).sum())
    rejected_n = int((~approved & known).sum())
    known_n = approved_n + rejected_n
    lines.extend(
        [
            "",
            "## Target distribution",
            "",
            f"- Approved (Y): **{approved_n}** ({approved_n / known_n:.1%} of labelled rows)" if known_n else "- No labelled rows.",
            f"- Rejected (N): **{rejected_n}** ({rejected_n / known_n:.1%} of labelled rows)" if known_n else "",
            "",
            "The classes are uneven. A model that always predicts approval can look accurate "
            "and still be useless, which is why the training script compares models with ROC-AUC "
            "and reports a most-frequent dummy baseline.",
            "",
            "## Univariate notes",
            "",
        ]
    )
    for column in ("ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        lines.append(
            f"- {column}: median {values.median():.1f}, mean {values.mean():.1f}, "
            f"IQR outliers { _iqr_outlier_count(values) } of {int(values.notna().sum())} non-missing values. "
            "Outliers are kept. Extreme incomes and loan amounts can be real applications."
        )
    lines.extend(["", "## Relationship with approval", ""])
    for column in (
        "Credit_History",
        "Education",
        "Self_Employed",
        "Dependents",
        "Property_Area",
        "Married",
        "Gender",
    ):
        sentence = _gap_sentence(frame, column, approved)
        if sentence:
            lines.append(f"- {sentence}")

    cleaned = clean_applications(frame)
    if set(("ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term")).issubset(cleaned.columns):
        engineered = add_engineered_features(cleaned)
        engineered = engineered.assign(_approved=approved.to_numpy())
        labelled = engineered.dropna(subset=["_approved"])
        lines.extend(["", "## Engineered features (descriptive only)", ""])
        for column in ("TotalIncome", "LoanToIncome", "AmountPerMonth"):
            approved_median = labelled.loc[labelled["_approved"], column].median()
            rejected_median = labelled.loc[~labelled["_approved"], column].median()
            lines.append(
                f"- {column}: median {approved_median:.3f} among approved and "
                f"{rejected_median:.3f} among rejected labelled rows."
            )
        lines.append("")
        lines.append(
            "These medians motivate keeping the three features. They are not a test-set result, "
            "and they are computed before any model is fit."
        )
    lines.extend(
        [
            "",
            "## Data quality",
            "",
            "See `data_validation_report.md` for invalid values and unexpected labels. "
            "Validation does not delete rows.",
            "",
        ]
    )
    return "\n".join(lines)


def run_eda(frame: pd.DataFrame | None = None, paths: Paths | None = None) -> Path:
    paths = paths or Paths()
    paths.ensure()
    data = frame if frame is not None else load_dataset(paths.raw_csv)
    validation = validate_dataframe(data, min_rows=1, require_model_inputs=False)
    write_validation_report(validation, paths.reports)
    manifest = {}
    if paths.manifest.exists():
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    elif load_manifest(paths):
        manifest = load_manifest(paths)
    write_dataset_information(data, manifest, paths.reports)
    if validation["target_column"] is not None:
        save_plots(data, validation["target_column"], paths.reports)
    report_path = paths.reports / "eda_report.md"
    report_path.write_text(build_eda_markdown(data, validation), encoding="utf-8")
    summary = {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "target_column": validation["target_column"],
        "target_distribution": validation["target_distribution"],
        "missing_values": validation["missing_values"],
        "duplicate_records": validation["duplicate_records"],
    }
    (paths.reports / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return report_path
