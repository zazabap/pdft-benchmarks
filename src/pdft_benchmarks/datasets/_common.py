"""Shared dataset-loading helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label} data_root does not exist: {path}\n"
            f"Place the dataset at this path or pass data_root=... to override."
        )
