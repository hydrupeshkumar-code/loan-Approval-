"""SHAP contributions must add up from the base value to the predicted probability."""

from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.loan_approval.explainability import make_shap_explainer, shap_explanation, shap_importance
from src.loan_approval.preprocessing import build_pipeline, normalize_target
from tests.synthetic import make_loan_frame

pytest.importorskip("shap")


def _fit(estimator, scale: bool):
    frame = make_loan_frame(80, seed=21)
    target = (normalize_target(frame["Loan_Status"]) == "Y").astype(int)
    features = frame.drop(columns=["Loan_Status", "Loan_ID", "Gender"])
    return build_pipeline(estimator, scale=scale).fit(features, target), features


def test_shap_adds_up_and_groups_one_hot_columns() -> None:
    pipeline, features = _fit(RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0), scale=False)
    explainer = make_shap_explainer(pipeline)
    row = features.iloc[[3]]
    result = shap_explanation(pipeline, explainer, row)
    total = result["base_value"] + sum(item["contribution"] for item in result["items"])
    assert total == pytest.approx(float(pipeline.predict_proba(row)[0, 1]), abs=1e-6)
    names = {item["feature"] for item in result["items"]}
    assert "Property_Area" in names and "Credit_History" in names
    assert not any(name.startswith("Property_Area_") for name in names)
    table = shap_importance(pipeline, features)
    assert table["importance_mean"].ge(0).all()


def test_shap_skips_non_tree_models() -> None:
    pipeline, _ = _fit(LogisticRegression(max_iter=500), scale=True)
    assert make_shap_explainer(pipeline) is None
