"""Find, download, and check a public labelled loan-approval table.

The primary file is the Analytics Vidhya practice problem "Loan Prediction"
(Dream Housing Finance). Analytics Vidhya does not publish a stable anonymous
file URL, so the bytes come from a public GitHub raw file of that same table.
The GitHub repository has no licence file. That limit is stored in the manifest.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from src.loan_approval.config import Paths
from src.loan_approval.data_loader import read_table
from src.loan_approval.data_validation import (
    validate_dataframe,
    write_dataset_information,
    write_validation_report,
)
from src.loan_approval.utils import (
    DatasetNotFoundError,
    ValidationError,
    get_logger,
    sha256_file,
)

LOGGER = get_logger("loan_approval.download")
MAX_BYTES = 20 * 1024 * 1024
Fetcher = Callable[[str], bytes]

LICENCE_NOTE = (
    "The GitHub mirror does not contain a licence file (GitHub's license API "
    "returns 404). The table is the public Analytics Vidhya practice problem "
    "Loan Prediction, published for learning. This project uses it for "
    "coursework, demonstration, and portfolios. Check Analytics Vidhya's "
    "current terms before any commercial use, and do not treat this mirror "
    "as an official data licence."
)

DESCRIPTION = (
    "Home-loan applications with applicant and co-applicant income, loan amount "
    "and term, credit history, education, employment, dependents, property area, "
    "and a binary loan decision. Loan_Status Y means approved and N means rejected. "
    "Gender is present for audit and is not used as a model input."
)


@dataclass(frozen=True)
class DatasetSource:
    dataset_name: str
    provider: str
    repository: str
    original_source_url: str
    download_url: str
    original_filename: str
    licence: str
    description: str


SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        dataset_name="Analytics Vidhya Loan Prediction (Dream Housing Finance)",
        provider="GitHub mirror of the Analytics Vidhya practice problem",
        repository="https://github.com/shrikant-temburwar/Loan-Prediction-Dataset",
        original_source_url="https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/",
        download_url="https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv",
        original_filename="train.csv",
        licence=LICENCE_NOTE,
        description=DESCRIPTION,
    ),
    DatasetSource(
        dataset_name="Analytics Vidhya Loan Prediction (Dream Housing Finance)",
        provider="GitHub mirror (mridulrb / IBM Watson Studio example)",
        repository="https://github.com/mridulrb/Predict-loan-eligibility-using-IBM-Watson-Studio",
        original_source_url="https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/",
        download_url="https://raw.githubusercontent.com/mridulrb/Predict-loan-eligibility-using-IBM-Watson-Studio/master/Dataset/train_u6lujuX_CVtuZ9i.csv",
        original_filename="train_u6lujuX_CVtuZ9i.csv",
        licence=LICENCE_NOTE,
        description=DESCRIPTION + " Fallback mirror of the same practice-problem training file.",
    ),
    DatasetSource(
        dataset_name="Analytics Vidhya Loan Prediction (Dream Housing Finance)",
        provider="GitHub mirror (dphi-official Datasets)",
        repository="https://github.com/dphi-official/Datasets",
        original_source_url="https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/",
        download_url="https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv",
        original_filename="loan_train.csv",
        licence=LICENCE_NOTE,
        description=(
            DESCRIPTION
            + " Fallback copy of the same applicant table. The decision may be coded 1/0 instead of Y/N."
        ),
    ),
)

MANUAL_COMMAND = """python scripts/download_data.py

If that cannot reach GitHub, save this file as data/raw/loan_prediction.csv:

https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv

Problem page (the publisher; it may require a login to download):
https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/
"""


def looks_like_csv(content: bytes) -> bool:
    sample = content.lstrip()[:800].lower()
    if not sample or sample.startswith(b"<") or b"<html" in sample or b"<!doctype" in sample:
        return False
    first = content.splitlines()[0] if content.splitlines() else b""
    return b"," in first


def extract_csv_from_zip(content: bytes) -> bytes:
    if not zipfile.is_zipfile(io.BytesIO(content)):
        raise ValidationError("The download was not a readable ZIP file.")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv") and not name.endswith("/")]
        if not names:
            raise ValidationError("The ZIP archive does not contain a CSV file.")
        return archive.read(names[0])


def fetch_bytes(url: str) -> bytes:
    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "loan-approval-prediction-educational"},
    )
    response.raise_for_status()
    if len(response.content) > MAX_BYTES:
        raise ValidationError(f"Refusing to save {url}: file is larger than {MAX_BYTES} bytes.")
    return response.content


def verify_github_raw_url(url: str) -> dict[str, object]:
    """Confirm a raw.githubusercontent.com file against the GitHub contents API.

    A rate limit or a network error is recorded and does not by itself reject
    a file that later parses as the expected loan table.
    """
    prefix = "https://raw.githubusercontent.com/"
    if not url.startswith(prefix):
        return {"checked": False, "reason": "Not a GitHub raw URL."}
    parts = url[len(prefix) :].split("/", 3)
    if len(parts) != 4:
        return {"checked": False, "reason": "Could not parse owner, repository, branch, and path."}
    owner, repo, branch, path = parts
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    try:
        response = requests.get(
            api,
            timeout=30,
            headers={
                "User-Agent": "loan-approval-prediction-educational",
                "Accept": "application/vnd.github+json",
            },
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {"checked": False, "reason": f"GitHub API request failed: {exc}"}
    download_url = payload.get("download_url")
    size = payload.get("size") or 0
    verified = payload.get("type") == "file" and bool(download_url) and size > 100
    return {
        "checked": True,
        "verified": verified,
        "api_url": api,
        "download_url": download_url,
        "size": size,
        "sha": payload.get("sha"),
        "html_url": payload.get("html_url"),
    }


def _prepare_bytes(content: bytes) -> bytes:
    if content[:2] == b"PK":
        content = extract_csv_from_zip(content)
    if not looks_like_csv(content):
        raise ValidationError(
            "The download is not a CSV. HTML error pages are rejected and are not saved as data."
        )
    return content


def _manifest_matches(paths: Paths) -> bool:
    if not paths.manifest.exists() or not paths.raw_csv.exists():
        return False
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    recorded = manifest.get("sha256")
    return bool(recorded) and recorded == sha256_file(paths.raw_csv)


def _write_manifest(
    paths: Paths,
    source: DatasetSource,
    frame: pd.DataFrame,
    target: str,
    github_verification: dict[str, object],
) -> dict[str, object]:
    manifest = {
        "dataset_name": source.dataset_name,
        "original_source_url": source.original_source_url,
        "direct_download_url": source.download_url,
        "repository_or_provider": source.provider,
        "repository": source.repository,
        "licence": source.licence,
        "download_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256_file(paths.raw_csv),
        "original_filename": source.original_filename,
        "dimensions": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "column_names": [str(column) for column in frame.columns],
        "target_column": target,
        "description": source.description,
        "local_path": "data/raw/loan_prediction.csv",
        "github_verification": github_verification,
        "licence_limitation": (
            "Unclear SPDX licence on the mirror. Documented here instead of ignored. "
            "UCI Statlog German Credit was not used: it is a different credit-risk table."
        ),
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _write_sample(paths: Paths, frame: pd.DataFrame, target: str) -> None:
    drop = [column for column in (target, "Loan_ID") if column in frame.columns]
    sample = frame.drop(columns=drop).head(8)
    sample.to_csv(paths.sample / "sample_applicants.csv", index=False)


def download_dataset(
    paths: Paths | None = None,
    *,
    fetcher: Fetcher = fetch_bytes,
    sources: tuple[DatasetSource, ...] = SOURCES,
    force: bool = False,
) -> Path:
    """Download the first source that validates. Reuse a checksum-matched local file."""
    paths = paths or Paths()
    paths.ensure()
    if not force and _manifest_matches(paths):
        LOGGER.info("Valid dataset already present at %s. Skipping download.", paths.raw_csv)
        frame = read_table(paths.raw_csv)
        report = validate_dataframe(frame, min_rows=100, require_model_inputs=True)
        write_validation_report(report, paths.reports)
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        write_dataset_information(frame, manifest, paths.reports)
        return paths.raw_csv

    failures: list[str] = []
    for source in sources:
        LOGGER.info("Trying %s", source.download_url)
        github_verification = verify_github_raw_url(source.download_url) if fetcher is fetch_bytes else {
            "checked": False,
            "reason": "Custom fetcher; GitHub API was not called.",
        }
        try:
            content = _prepare_bytes(fetcher(source.download_url))
            temp = paths.raw_csv.with_suffix(".part")
            temp.write_bytes(content)
            frame = read_table(temp)
            report = validate_dataframe(frame, min_rows=100, require_model_inputs=True)
        except (ValidationError, DatasetNotFoundError, pd.errors.ParserError, OSError, requests.RequestException, zipfile.BadZipFile) as exc:
            failures.append(f"{source.download_url} — {exc}")
            part = paths.raw_csv.with_suffix(".part")
            if part.exists():
                part.unlink()
            LOGGER.warning("Source failed: %s", exc)
            continue
        temp.replace(paths.raw_csv)
        target = str(report["target_column"])
        manifest = _write_manifest(paths, source, frame, target, github_verification)
        write_validation_report(report, paths.reports)
        write_dataset_information(frame, manifest, paths.reports)
        _write_sample(paths, frame, target)
        LOGGER.info(
            "Saved %s rows × %s columns from %s",
            frame.shape[0],
            frame.shape[1],
            source.provider,
        )
        return paths.raw_csv

    raise DatasetNotFoundError(
        "No suitable loan-approval dataset could be downloaded.\n"
        + "\n".join(failures)
        + "\n\nManual download:\n"
        + MANUAL_COMMAND
    )


def main() -> None:
    from src.loan_approval.utils import configure_logging

    configure_logging()
    path = download_dataset()
    manifest = json.loads(Paths().manifest.read_text(encoding="utf-8"))
    print(f"Dataset: {manifest['dataset_name']}")
    print(f"Source: {manifest['original_source_url']}")
    print(f"Download: {manifest['direct_download_url']}")
    print(f"File: {path}")
    print(f"Shape: {manifest['dimensions']['rows']} rows, {manifest['dimensions']['columns']} columns")
    print(f"Target: {manifest['target_column']}")
    print(f"SHA-256: {manifest['sha256']}")
