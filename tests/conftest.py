"""Shared fixtures. The fast training run happens once per pytest session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.loan_approval.config import Paths
from src.loan_approval.train import run_training
from tests.synthetic import make_loan_frame


@pytest.fixture(scope="session")
def trained(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("loan-model")
    paths = Paths(root)
    metadata = run_training(make_loan_frame(180, seed=0), mode="fast", paths=paths)
    return paths, metadata
