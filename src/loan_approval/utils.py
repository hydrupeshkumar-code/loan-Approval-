"""Small helpers: logging, hashing, and markdown tables."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path


class DatasetNotFoundError(FileNotFoundError):
    """The raw dataset is missing or could not be downloaded."""


class ValidationError(ValueError):
    """The table is present but cannot be used for this project."""


class ModelNotFoundError(FileNotFoundError):
    """The trained pipeline has not been saved yet."""


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([head, rule, *body])


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"
