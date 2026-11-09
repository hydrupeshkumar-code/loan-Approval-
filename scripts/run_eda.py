"""Write the EDA report and charts from the dataset on disk."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.loan_approval.eda import run_eda
from src.loan_approval.utils import configure_logging


def main() -> None:
    configure_logging()
    path = run_eda()
    print(f"EDA report: {path}")


if __name__ == "__main__":
    main()
