"""Single-row and batch scoring, including bad inputs and a missing model."""

from __future__ import annotations

import pandas as pd
import pytest

from src.loan_approval.config import Paths
from src.loan_approval.predict import load_model, predict_frame, predict_one, score_frame
from src.loan_approval.utils import ModelNotFoundError
from tests.synthetic import make_loan_frame


def _record() -> dict[str, object]:
    frame = make_loan_frame(1, seed=11).drop(columns=["Loan_Status", "Loan_ID", "Gender"])
    return frame.iloc[0].to_dict()


def test_single_and_batch_prediction(trained) -> None:
    paths, _metadata = trained
    result = predict_one(_record(), paths=paths)
    assert result["status"] in {"Approved", "Rejected"}
    assert 0.0 <= result["probability_approved"] <= 1.0
    assert result["disclaimer"]
    batch = make_loan_frame(5, seed=12).drop(columns=["Loan_Status"])
    scored = predict_frame(batch, paths=paths)
    assert list(scored["Predicted_Status"]) 
    assert len(scored) == 5
    unknown = batch.copy()
    unknown["Property_Area"] = "Mars"
    assert len(predict_frame(unknown, paths=paths)) == 5


def test_cautious_mode_and_review_notes(trained) -> None:
    paths, metadata = trained
    assert 0.0 < metadata["cautious_threshold"] < 1.0
    assert "true_negative" in metadata["cautious_test_metrics"]
    batch = make_loan_frame(3, seed=13).drop(columns=["Loan_Status"])
    batch.loc[0, "Credit_History"] = None
    batch.loc[1, "ApplicantIncome"] = 10_000_000
    pipeline, saved = load_model(paths)
    strict, _ = score_frame(batch, pipeline=pipeline, metadata=saved, threshold=1.01)
    loose, _ = score_frame(batch, pipeline=pipeline, metadata=saved, threshold=0.0)
    assert (strict["Predicted_Status"] == "Rejected").all()
    assert (loose["Predicted_Status"] == "Approved").all()
    assert "credit history missing" in loose["Review_Note"].iloc[0]
    assert "ApplicantIncome outside training range" in loose["Review_Note"].iloc[1]


def test_missing_feature_and_bad_type(trained) -> None:
    paths, _metadata = trained
    with pytest.raises(ValueError, match="Missing features"):
        predict_one({"ApplicantIncome": 1000}, paths=paths)
    bad = _record()
    bad["LoanAmount"] = "large"
    with pytest.raises(ValueError, match="LoanAmount"):
        predict_one(bad, paths=paths)


def test_missing_model(tmp_path) -> None:
    with pytest.raises(ModelNotFoundError):
        load_model(Paths(tmp_path))
    with pytest.raises(ModelNotFoundError):
        predict_frame(pd.DataFrame([_record()]), paths=Paths(tmp_path))
