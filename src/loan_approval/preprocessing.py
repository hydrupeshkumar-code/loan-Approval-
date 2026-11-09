"""Cleaning, feature engineering, and leakage-safe sklearn pipelines.

`clean_applications` and `add_engineered_features` do not look at the target
and do not learn a mean, a mode, or a scale. Imputation and scaling are inside
the sklearn pipeline, which is fit on training rows only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.loan_approval.config import (
    ENGINEERED_FEATURES,
    RAW_CATEGORICAL,
    RAW_NUMERIC,
    REQUIRED_INPUTS,
)

MODEL_COLUMNS = list(RAW_NUMERIC) + list(ENGINEERED_FEATURES) + list(RAW_CATEGORICAL)

YES_NO = {"y": "Yes", "yes": "Yes", "n": "No", "no": "No", "true": "Yes", "false": "No"}
EDUCATION = {
    "graduate": "Graduate",
    "not graduate": "Not Graduate",
    "notgraduate": "Not Graduate",
    "under graduate": "Not Graduate",
    "undergraduate": "Not Graduate",
}
PROPERTY = {
    "urban": "Urban",
    "semiurban": "Semiurban",
    "semi-urban": "Semiurban",
    "semi urban": "Semiurban",
    "rural": "Rural",
}
DEPENDENTS = {"0": "0", "1": "1", "2": "2", "3": "3+", "3+": "3+"}
GENDER = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}
CREDIT = {"1": "1", "1.0": "1", "0": "0", "0.0": "0"}

APPROVED_TOKENS = {"y", "yes", "approved", "approve", "1", "1.0", "true"}
REJECTED_TOKENS = {"n", "no", "rejected", "reject", "0", "0.0", "false"}


def _token(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return None
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        text = text[:-2]
    return text


def _map_tokens(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    cleaned = []
    for value in series.tolist():
        token = _token(value)
        if token is None:
            cleaned.append(np.nan)
        else:
            cleaned.append(mapping.get(token.lower(), mapping.get(token, token)))
    return pd.Series(cleaned, index=series.index, dtype="object")


def normalize_target(series: pd.Series) -> pd.Series:
    """Map common approval labels to Y or N. Anything else becomes missing."""
    labels = []
    for value in series.tolist():
        token = _token(value)
        if token is None:
            labels.append(np.nan)
        elif token.lower() in APPROVED_TOKENS:
            labels.append("Y")
        elif token.lower() in REJECTED_TOKENS:
            labels.append("N")
        else:
            labels.append(np.nan)
    return pd.Series(labels, index=series.index, dtype="object")


def clean_applications(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip headers and labels, and turn impossible numbers into missing values.

    Rows are not dropped here. Invalid numbers become missing so the imputer,
    which is fit on the training split only, can fill them.
    """
    out = frame.copy()
    out.columns = [str(column).strip() for column in out.columns]
    unnamed = [column for column in out.columns if column.lower().startswith("unnamed")]
    out = out.drop(columns=unnamed, errors="ignore")

    for column in out.select_dtypes(include="object").columns:
        out[column] = out[column].map(_token)

    if "Gender" in out.columns:
        out["Gender"] = _map_tokens(out["Gender"], GENDER)
    if "Married" in out.columns:
        out["Married"] = _map_tokens(out["Married"], YES_NO)
    if "Self_Employed" in out.columns:
        out["Self_Employed"] = _map_tokens(out["Self_Employed"], YES_NO)
    if "Education" in out.columns:
        out["Education"] = _map_tokens(out["Education"], EDUCATION)
    if "Property_Area" in out.columns:
        out["Property_Area"] = _map_tokens(out["Property_Area"], PROPERTY)
    if "Dependents" in out.columns:
        out["Dependents"] = _map_tokens(out["Dependents"], DEPENDENTS)
    if "Credit_History" in out.columns:
        out["Credit_History"] = _map_tokens(out["Credit_History"], CREDIT)

    for column in RAW_NUMERIC:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    for column in ("ApplicantIncome", "CoapplicantIncome"):
        if column in out.columns:
            out.loc[out[column] < 0, column] = np.nan
    if "LoanAmount" in out.columns:
        out.loc[out["LoanAmount"] <= 0, "LoanAmount"] = np.nan
    if "Loan_Amount_Term" in out.columns:
        invalid_term = (out["Loan_Amount_Term"] <= 0) | (out["Loan_Amount_Term"] > 600)
        out.loc[invalid_term, "Loan_Amount_Term"] = np.nan
    return out


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the three ratios described in ``FEATURE_NOTES``."""
    out = frame.copy()
    applicant = out["ApplicantIncome"]
    coapplicant = out["CoapplicantIncome"]
    total = applicant.fillna(0) + coapplicant.fillna(0)
    out["TotalIncome"] = total.mask(applicant.isna() & coapplicant.isna())
    safe_total = out["TotalIncome"].where(out["TotalIncome"] > 0)
    out["LoanToIncome"] = out["LoanAmount"] / safe_total
    safe_term = out["Loan_Amount_Term"].where(out["Loan_Amount_Term"] > 0)
    out["AmountPerMonth"] = out["LoanAmount"] / safe_term
    return out


def model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the columns the estimator is allowed to see, in a fixed order."""
    missing = [column for column in REQUIRED_INPUTS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Gender and Loan_ID are not model inputs."
        )
    prepared = add_engineered_features(clean_applications(frame))
    model = prepared.loc[:, MODEL_COLUMNS].copy()
    for column in list(RAW_NUMERIC) + list(ENGINEERED_FEATURES):
        model[column] = pd.to_numeric(model[column], errors="coerce").astype(float)
    for column in RAW_CATEGORICAL:
        model[column] = model[column].astype("object")
    return model


class LoanFeatureBuilder(BaseEstimator, TransformerMixin):
    """Stateless cleaning and feature engineering as the first pipeline step."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "LoanFeatureBuilder":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "LoanFeatureBuilder needs a pandas DataFrame with column names. "
                "Do not pass a plain NumPy array."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LoanFeatureBuilder needs a pandas DataFrame with column names.")
        return model_frame(X)


def _one_hot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn older than 1.2 used `sparse`.
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_column_transformer(scale: bool) -> ColumnTransformer:
    """Impute every model. Scale numeric columns only when ``scale`` is true.

    Tree models keep an unscaled numeric pipeline. Logistic regression gets
    StandardScaler because its coefficients depend on units.
    """
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(RAW_NUMERIC) + list(ENGINEERED_FEATURES)),
            ("categorical", categorical, list(RAW_CATEGORICAL)),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator: Any, *, scale: bool) -> Pipeline:
    """Full preprocess-and-predict pipeline. Fit it on the training split only."""
    return Pipeline(
        [
            ("features", LoanFeatureBuilder()),
            ("prepare", make_column_transformer(scale=scale)),
            ("model", estimator),
        ]
    )
