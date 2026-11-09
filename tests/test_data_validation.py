"""Dataset checks and the download fallback. Network is not required."""

from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import pytest

from src.loan_approval.config import Paths
from src.loan_approval.data_loader import load_dataset
from src.loan_approval.data_validation import validate_dataframe
from src.loan_approval.download import (
    SOURCES,
    download_dataset,
    extract_csv_from_zip,
    looks_like_csv,
)
from src.loan_approval.utils import DatasetNotFoundError, ValidationError
from tests.synthetic import make_loan_frame


def test_rejects_html_and_accepts_csv() -> None:
    assert looks_like_csv(b"<!DOCTYPE html><html><body>error</body></html>") is False
    assert looks_like_csv(b"<html>no</html>") is False
    assert looks_like_csv(b"Loan_Status,ApplicantIncome\nY,100\n") is True


def test_zip_extracts_csv() -> None:
    payload = b"Loan_Status,ApplicantIncome\nY,100\nN,50\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("nested/train.csv", payload)
    extracted = extract_csv_from_zip(buffer.getvalue())
    assert extracted == payload


def test_validation_does_not_drop_rows() -> None:
    frame = make_loan_frame(30, seed=1)
    before = len(frame)
    report = validate_dataframe(frame)
    assert len(frame) == before
    assert report["records_removed"] == 0
    assert report["target_column"] == "Loan_Status"
    assert report["rows"] == 30


def test_missing_file(tmp_path) -> None:
    with pytest.raises(DatasetNotFoundError):
        load_dataset(tmp_path / "missing.csv")


def test_empty_dataset() -> None:
    frame = make_loan_frame(5).iloc[0:0]
    with pytest.raises(ValidationError):
        validate_dataframe(frame, min_rows=1)


def test_missing_target_column() -> None:
    frame = make_loan_frame(20).drop(columns=["Loan_Status"])
    with pytest.raises(ValidationError, match="loan-status"):
        validate_dataframe(frame, require_model_inputs=True)


def test_missing_required_column() -> None:
    frame = make_loan_frame(20).drop(columns=["Credit_History"])
    with pytest.raises(ValidationError, match="Credit_History"):
        validate_dataframe(frame, require_model_inputs=True)


def test_invalid_target_values_are_reported_and_all_invalid_fails() -> None:
    mixed = make_loan_frame(20, seed=2)
    mixed.loc[0, "Loan_Status"] = "Maybe"
    report = validate_dataframe(mixed)
    assert "Maybe" in report["unexpected_target_labels"]
    broken = mixed.copy()
    broken["Loan_Status"] = "Maybe"
    with pytest.raises(ValidationError):
        validate_dataframe(broken, require_model_inputs=True, min_rows=1)


def test_download_skips_html_then_saves_csv(tmp_path) -> None:
    good = make_loan_frame(120, seed=3).to_csv(index=False).encode("utf-8")
    calls: list[str] = []

    def fetcher(url: str) -> bytes:
        calls.append(url)
        if len(calls) == 1:
            return b"<html><body>not a dataset</body></html>"
        return good

    paths = Paths(tmp_path)
    saved = download_dataset(paths, fetcher=fetcher, sources=SOURCES[:2])
    assert saved.exists()
    assert len(calls) == 2
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["sha256"]
    assert manifest["target_column"] == "Loan_Status"
    assert manifest["dimensions"]["rows"] == 120
    frame = pd.read_csv(saved)
    assert "Loan_Status" in frame.columns

    def fail(url: str) -> bytes:
        raise AssertionError(url)

    again = download_dataset(paths, fetcher=fail, sources=SOURCES[:2])
    assert again == saved


def test_download_accepts_zip(tmp_path) -> None:
    good = make_loan_frame(110, seed=4).to_csv(index=False).encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("train.csv", good)

    def fetcher(url: str) -> bytes:
        return buffer.getvalue()

    paths = Paths(tmp_path)
    download_dataset(paths, fetcher=fetcher, sources=SOURCES[:1])
    assert paths.raw_csv.exists()
    assert len(pd.read_csv(paths.raw_csv)) == 110
