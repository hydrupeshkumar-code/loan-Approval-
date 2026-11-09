"""Load the local CSV (or a ZIP that contains one CSV)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from src.loan_approval.config import Paths
from src.loan_approval.utils import DatasetNotFoundError

DOWNLOAD_HINT = (
    "The loan dataset is not in data/raw yet. From the project folder run:\n"
    "python scripts/download_data.py"
)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not names:
                raise DatasetNotFoundError(f"No CSV found inside {path}.")
            frame = pd.read_csv(io.BytesIO(archive.read(names[0])), encoding="utf-8-sig")
    else:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or Paths().raw_csv
    if not csv_path.exists():
        raise DatasetNotFoundError(f"{DOWNLOAD_HINT}\nExpected file: {csv_path}")
    if csv_path.stat().st_size == 0:
        raise DatasetNotFoundError(f"Dataset file is empty: {csv_path}")
    frame = read_table(csv_path)
    if frame.empty:
        raise DatasetNotFoundError(f"Dataset has no rows: {csv_path}")
    return frame
