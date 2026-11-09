"""Read-only checks. This module never drops or rewrites applicant rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.loan_approval.config import (
    IDENTIFIER_CANDIDATES,
    RAW_CATEGORICAL,
    RAW_NUMERIC,
    REQUIRED_INPUTS,
    TARGET_CANDIDATES,
    Paths,
)
from src.loan_approval.preprocessing import normalize_target
from src.loan_approval.utils import ValidationError, markdown_table

EXPECTED_CATEGORIES = {
    "Gender": {"Male", "Female"},
    "Married": {"Yes", "No"},
    "Education": {"Graduate", "Not Graduate"},
    "Self_Employed": {"Yes", "No"},
    "Property_Area": {"Urban", "Semiurban", "Rural"},
    "Dependents": {"0", "1", "2", "3+"},
    "Credit_History": {"0", "1"},
}


def find_target_column(columns: list[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _identifier_columns(frame: pd.DataFrame, target: str | None) -> list[str]:
    found: list[str] = []
    for column in frame.columns:
        if column == target:
            continue
        name = column.lower()
        if column in IDENTIFIER_CANDIDATES or name.endswith("_id") or name == "id":
            found.append(column)
            continue
        if frame[column].dtype == object and frame[column].nunique(dropna=True) == len(frame):
            found.append(column)
    return found


def _invalid_numbers(frame: pd.DataFrame) -> dict[str, int]:
    invalid: dict[str, int] = {}
    for column in ("ApplicantIncome", "CoapplicantIncome"):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid[f"{column}_negative"] = int((values < 0).sum())
    if "LoanAmount" in frame.columns:
        values = pd.to_numeric(frame["LoanAmount"], errors="coerce")
        invalid["LoanAmount_not_positive"] = int((values <= 0).sum())
    if "Loan_Amount_Term" in frame.columns:
        values = pd.to_numeric(frame["Loan_Amount_Term"], errors="coerce")
        invalid["Loan_Amount_Term_outside_1_to_600"] = int(((values <= 0) | (values > 600)).sum())
    if "Credit_History" in frame.columns:
        token = frame["Credit_History"].map(lambda value: str(value).strip() if pd.notna(value) else pd.NA)
        token = token.replace({"1.0": "1", "0.0": "0"})
        bad = token.notna() & ~token.isin(["0", "1"])
        invalid["Credit_History_not_0_or_1"] = int(bad.sum())
    return {name: count for name, count in invalid.items() if count > 0}


def _unexpected_categories(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    unexpected: dict[str, dict[str, int]] = {}
    for column, allowed in EXPECTED_CATEGORIES.items():
        if column not in frame.columns:
            continue
        values = frame[column].dropna().astype(str).str.strip()
        values = values.replace({"1.0": "1", "0.0": "0"})
        bad = values[~values.isin(allowed)]
        if not bad.empty:
            unexpected[column] = {str(key): int(count) for key, count in bad.value_counts().items()}
    return unexpected


def validate_dataframe(
    frame: pd.DataFrame,
    *,
    min_rows: int = 1,
    require_model_inputs: bool = False,
) -> dict[str, Any]:
    """Describe data quality. Raises only when the table cannot be used at all."""
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError("Expected a pandas DataFrame.")
    table = frame.copy()
    table.columns = [str(column).strip() for column in table.columns]
    if table.columns.duplicated().any():
        duplicates = table.columns[table.columns.duplicated()].tolist()
        raise ValidationError(f"Duplicate column names: {duplicates}")

    target = find_target_column(list(table.columns))
    missing_inputs = [column for column in REQUIRED_INPUTS if column not in table.columns]
    if require_model_inputs and missing_inputs:
        raise ValidationError(
            "The dataset is missing required applicant columns: " + ", ".join(missing_inputs)
        )
    if target is None and require_model_inputs:
        raise ValidationError(
            "No loan-status target column was found. Looked for: " + ", ".join(TARGET_CANDIDATES)
        )
    if len(table) < min_rows:
        raise ValidationError(
            f"Dataset has {len(table)} rows; at least {min_rows} labelled rows are required."
        )
    if len(table.columns) < 2:
        raise ValidationError("Dataset does not have enough columns to be a loan table.")

    numeric = [
        column
        for column in table.columns
        if column in RAW_NUMERIC or pd.api.types.is_numeric_dtype(table[column])
    ]
    categorical = [column for column in table.columns if column not in numeric and column != target]

    missing = {column: int(table[column].isna().sum()) for column in table.columns}
    missing = {column: count for column, count in missing.items() if count > 0}

    target_distribution: dict[str, int] = {}
    unexpected_labels: dict[str, int] = {}
    missing_target = 0
    if target is not None:
        normalised = normalize_target(table[target])
        missing_target = int(normalised.isna().sum())
        target_distribution = {
            str(label): int(count) for label, count in normalised.dropna().value_counts().items()
        }
        raw_bad = table.loc[normalised.isna(), target]
        unexpected_labels = {
            str(label): int(count) for label, count in raw_bad.dropna().astype(str).value_counts().items()
        }
        recognised = int(normalised.notna().sum())
        if require_model_inputs and recognised == 0:
            raise ValidationError(
                f"Column {target} has no recognised approval labels (Y/N, 1/0, Approved/Rejected)."
            )
        if require_model_inputs and recognised < min_rows:
            raise ValidationError(
                f"Only {recognised} rows have a recognised loan status; {min_rows} are required."
            )

    report = {
        "rows": int(len(table)),
        "columns": int(len(table.columns)),
        "column_names": [str(column) for column in table.columns],
        "dtypes": {column: str(dtype) for column, dtype in table.dtypes.items()},
        "numeric_features": numeric,
        "categorical_features": categorical,
        "missing_values": missing,
        "duplicate_records": int(table.duplicated().sum()),
        "target_column": target,
        "target_distribution": target_distribution,
        "missing_target_rows": missing_target,
        "unexpected_target_labels": unexpected_labels,
        "identifier_columns": _identifier_columns(table, target),
        "invalid_values": _invalid_numbers(table),
        "unexpected_categories": _unexpected_categories(table),
        "missing_model_inputs": missing_inputs,
        "records_removed": 0,
    }
    return report


def render_validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Data validation report",
        "",
        "This check does not delete rows. Counts below are problems to know about, not rows that were removed.",
        "",
        f"- Records: **{report['rows']}**",
        f"- Columns: **{report['columns']}**",
        f"- Target: **{report['target_column']}**",
        f"- Duplicate records: **{report['duplicate_records']}**",
        f"- Rows with a missing or unrecognised target: **{report['missing_target_rows']}**",
        f"- Possible identifier columns: {', '.join(report['identifier_columns']) or 'none'}",
        "",
        "## Target distribution",
        "",
    ]
    distribution = report["target_distribution"]
    if distribution:
        lines.append(
            markdown_table(
                ["Label", "Rows"],
                [[label, count] for label, count in distribution.items()],
            )
        )
    else:
        lines.append("No recognised target labels.")
    lines.extend(["", "## Missing values", ""])
    if report["missing_values"]:
        lines.append(
            markdown_table(
                ["Column", "Missing"],
                [[column, count] for column, count in report["missing_values"].items()],
            )
        )
    else:
        lines.append("No missing values.")
    lines.extend(["", "## Invalid or unexpected values", ""])
    if report["invalid_values"] or report["unexpected_categories"] or report["unexpected_target_labels"]:
        if report["invalid_values"]:
            lines.append(
                markdown_table(
                    ["Check", "Rows"],
                    [[name, count] for name, count in report["invalid_values"].items()],
                )
            )
        if report["unexpected_target_labels"]:
            lines.extend(["", "Unrecognised target labels:", ""])
            lines.append(
                markdown_table(
                    ["Label", "Rows"],
                    [[label, count] for label, count in report["unexpected_target_labels"].items()],
                )
            )
        if report["unexpected_categories"]:
            lines.extend(["", "Category values outside the usual set:", ""])
            rows = []
            for column, counts in report["unexpected_categories"].items():
                for label, count in counts.items():
                    rows.append([column, label, count])
            lines.append(markdown_table(["Column", "Value", "Rows"], rows))
    else:
        lines.append("No invalid numbers or unexpected category labels were found.")
    if report["missing_model_inputs"]:
        lines.extend(
            [
                "",
                "## Columns the model expects but this file does not have",
                "",
                ", ".join(report["missing_model_inputs"]),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_validation_report(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "data_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (report_dir / "data_validation_report.md").write_text(
        render_validation_markdown(report), encoding="utf-8"
    )


def write_dataset_information(
    frame: pd.DataFrame,
    manifest: dict[str, Any] | None,
    report_dir: Path,
) -> None:
    manifest = manifest or {}
    dimensions = manifest.get("dimensions", {})
    lines = [
        "# Dataset information",
        "",
        f"- Dataset name: {manifest.get('dataset_name', 'not recorded')}",
        f"- Original source: {manifest.get('original_source_url', 'not recorded')}",
        f"- Direct download: {manifest.get('direct_download_url', 'not recorded')}",
        f"- Provider: {manifest.get('repository_or_provider', 'not recorded')}",
        f"- Licence: {manifest.get('licence', 'not recorded')}",
        f"- Download date: {manifest.get('download_date', 'not recorded')}",
        f"- SHA-256: {manifest.get('sha256', 'not recorded')}",
        f"- Original filename: {manifest.get('original_filename', 'not recorded')}",
        f"- Rows in the file used here: {len(frame)}",
        f"- Columns in the file used here: {frame.shape[1]}",
        f"- Dimensions recorded at download: {dimensions.get('rows', 'n/a')} rows × {dimensions.get('columns', 'n/a')} columns",
        f"- Target column: {manifest.get('target_column', 'Loan_Status')}",
        "",
        "## Description",
        "",
        str(manifest.get("description", "No description was stored in the manifest.")),
        "",
        "## Columns",
        "",
        ", ".join(str(column) for column in frame.columns),
        "",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_information.md").write_text("\n".join(lines), encoding="utf-8")


def load_manifest(paths: Paths | None = None) -> dict[str, Any]:
    paths = paths or Paths()
    if not paths.manifest.exists():
        return {}
    return json.loads(paths.manifest.read_text(encoding="utf-8"))
