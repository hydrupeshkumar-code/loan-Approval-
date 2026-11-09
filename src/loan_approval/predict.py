"""Score new applications with the saved pipeline. This module does not train."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.loan_approval.config import RAW_NUMERIC, Paths
from src.loan_approval.explainability import local_explanation
from src.loan_approval.utils import ModelNotFoundError

DISCLAIMER = (
    "This is a statistical estimate from historical applications. "
    "It is not a lending decision, a credit report, or a guarantee of approval."
)


def load_model(paths: Paths | None = None) -> tuple[Any, dict[str, Any]]:
    paths = paths or Paths()
    if not paths.model.exists() or not paths.metadata.exists():
        raise ModelNotFoundError(
            "No trained model is saved yet. From the project folder run:\n"
            "python -m src.loan_approval.train"
        )
    pipeline = joblib.load(paths.model)
    metadata = json_load(paths)
    return pipeline, metadata


def json_load(paths: Paths) -> dict[str, Any]:
    import json

    return json.loads(paths.metadata.read_text(encoding="utf-8"))


def _validate_frame(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("The applicant table is empty.")
    required = list(metadata["raw_features"])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing features: " + ", ".join(missing))
    checked = frame.copy()
    for column in required:
        if column in RAW_NUMERIC:
            converted = pd.to_numeric(checked[column], errors="coerce")
            bad = checked[column].notna() & converted.isna()
            if bad.any():
                raise ValueError(f"{column} has non-numeric values.")
            checked[column] = converted
        else:
            checked[column] = checked[column].map(
                lambda value: np.nan if pd.isna(value) else str(value).strip()
            )
    return checked


BORDERLINE_MARGIN = 0.05


def review_notes(checked: pd.DataFrame, probability: np.ndarray, threshold: float, metadata: dict[str, Any]) -> list[str]:
    """Reasons a human should double-check a prediction. Empty string = none.

    - Credit history missing: the pipeline fills it with the most common value
      (good history), so the score leans on a guess.
    - Close to the cut-off: a small change in the inputs could flip the result.
    - A number outside the range seen in training: the model never learnt
      from such values, so the score is less reliable.
    """
    ranges = {
        field["name"]: (field.get("minimum"), field.get("maximum"))
        for field in metadata.get("input_schema", [])
        if field.get("kind") == "number"
    }
    notes: list[str] = []
    for position, (_, row) in enumerate(checked.iterrows()):
        reasons: list[str] = []
        if "Credit_History" in row and pd.isna(row["Credit_History"]):
            reasons.append("credit history missing")
        if abs(float(probability[position]) - threshold) < BORDERLINE_MARGIN:
            reasons.append("close to the cut-off")
        for column, (low, high) in ranges.items():
            value = row.get(column)
            if pd.notna(value) and low is not None and high is not None and not low <= float(value) <= high:
                reasons.append(f"{column} outside training range")
        notes.append("; ".join(reasons))
    return notes


def score_frame(
    frame: pd.DataFrame,
    *,
    paths: Paths | None = None,
    pipeline: Any | None = None,
    metadata: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score rows. threshold defaults to the saved standard cut-off."""
    if pipeline is None or metadata is None:
        pipeline, metadata = load_model(paths)
    checked = _validate_frame(frame, metadata)
    if not hasattr(pipeline, "predict_proba"):
        raise ModelNotFoundError("The saved model cannot produce probabilities.")
    probability = pipeline.predict_proba(checked)[:, 1]
    cutoff = float(metadata["threshold"] if threshold is None else threshold)
    scored = checked.copy()
    scored["Approval_Probability"] = probability
    scored["Predicted_Status"] = np.where(probability >= cutoff, "Approved", "Rejected")
    scored["Review_Note"] = review_notes(checked, probability, cutoff, metadata)
    return scored, metadata


def predict_one(record: dict[str, Any], paths: Paths | None = None) -> dict[str, Any]:
    pipeline, metadata = load_model(paths)
    frame = pd.DataFrame([record])
    scored, metadata = score_frame(frame, paths=paths, pipeline=pipeline, metadata=metadata)
    explanation = local_explanation(pipeline, frame, metadata.get("reference_values") or {})
    return {
        "status": str(scored["Predicted_Status"].iloc[0]),
        "probability_approved": float(scored["Approval_Probability"].iloc[0]),
        "threshold": float(metadata["threshold"]),
        "explanation": explanation[:5],
        "disclaimer": DISCLAIMER,
    }


def predict_frame(frame: pd.DataFrame, paths: Paths | None = None) -> pd.DataFrame:
    scored, _metadata = score_frame(frame, paths=paths)
    return scored
