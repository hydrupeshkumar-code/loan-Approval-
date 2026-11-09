"""Download the public loan table. Safe to run again: a valid local file is reused."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.loan_approval.download import main

if __name__ == "__main__":
    main()
