"""Preprocessing keeps a stable schema and does not learn from the test row."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.loan_approval.preprocessing import (
    MODEL_COLUMNS,
    add_engineered_features,
    build_pipeline,
    clean_applications,
    model_frame,
)
from tests.synthetic import make_loan_frame


def test_whitespace_and_label_cleanup() -> None:
    frame = make_loan_frame(6, seed=5)
    frame.loc[0, "Education"] = "  graduate "
    frame.loc[0, "Married"] = " y "
    cleaned = clean_applications(frame)
    assert cleaned.loc[0, "Education"] == "Graduate"
    assert cleaned.loc[0, "Married"] == "Yes"


def test_engineered_features_avoid_divide_by_zero() -> None:
    frame = make_loan_frame(4, seed=6)
    frame.loc[0, "ApplicantIncome"] = 1000
    frame.loc[0, "CoapplicantIncome"] = 500
    frame.loc[0, "LoanAmount"] = 150
    frame.loc[0, "Loan_Amount_Term"] = 10
    frame.loc[1, "ApplicantIncome"] = 0
    frame.loc[1, "CoapplicantIncome"] = 0
    frame.loc[1, "Loan_Amount_Term"] = 0
    built = add_engineered_features(clean_applications(frame))
    assert built.loc[0, "TotalIncome"] == 1500
    assert built.loc[0, "LoanToIncome"] == pytest.approx(150 / 1500)
    assert built.loc[0, "AmountPerMonth"] == pytest.approx(15)
    assert pd.isna(built.loc[1, "LoanToIncome"])
    assert pd.isna(built.loc[1, "AmountPerMonth"])


def test_schema_is_stable_with_extra_columns() -> None:
    frame = make_loan_frame(8, seed=7)
    shuffled = frame[list(reversed(frame.columns))]
    assert list(model_frame(frame).columns) == list(model_frame(shuffled).columns) == MODEL_COLUMNS
    assert "Gender" not in model_frame(frame).columns
    assert "Loan_ID" not in model_frame(frame).columns


def test_imputer_uses_training_values_only() -> None:
    train = make_loan_frame(40, seed=8)
    train["ApplicantIncome"] = 10.0
    train.loc[0, "ApplicantIncome"] = np.nan
    target = (train["Loan_Status"] == "Y").astype(int)
    pipeline = build_pipeline(LogisticRegression(max_iter=300), scale=False)
    pipeline.fit(train.drop(columns=["Loan_Status"]), target)
    row = train.iloc[[1]].copy()
    row["ApplicantIncome"] = np.nan
    transformed = pipeline.named_steps["prepare"].transform(
        pipeline.named_steps["features"].transform(row)
    )
    assert transformed[0, 0] == pytest.approx(10.0)


def test_unknown_category_does_not_crash() -> None:
    train = make_loan_frame(40, seed=9)
    target = (train["Loan_Status"] == "Y").astype(int)
    pipeline = build_pipeline(LogisticRegression(max_iter=300), scale=True)
    pipeline.fit(train.drop(columns=["Loan_Status"]), target)
    row = train.iloc[[0]].drop(columns=["Loan_Status"])
    row["Property_Area"] = "Mars"
    prediction = pipeline.predict(row)
    assert len(prediction) == 1
