"""HEVC-style content-adaptive block-size selector.

Public API
----------
adaptive_compress(image, candidates, keep_ratio, granularity, tile_size)
    Pick the candidate that minimises MSE on the target unit (whole image
    or per-CTU tile).  Returns (reconstruction, info).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

CompressFn = Callable[[np.ndarray, float], np.ndarray]
Candidate = tuple[str, CompressFn]


def adaptive_compress(
    image: np.ndarray,
    candidates: list[Candidate],
    keep_ratio: float,
    granularity: Literal["image", "tile"] = "image",
    tile_size: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pick the candidate that minimises MSE on the target unit.

    Parameters
    ----------
    image:
        2-D or 3-D float array (H, W) or (H, W, C).
    candidates:
        Non-empty list of ``(name, compress_fn)`` pairs.
        ``compress_fn(image, keep_ratio) -> reconstruction`` must return an
        array of the same shape and dtype as ``image``.
    keep_ratio:
        Fraction of coefficients to retain, forwarded unchanged to each
        ``compress_fn``.
    granularity:
        ``"image"`` — whole-image oracle; one candidate is chosen globally.
        ``"tile"``  — per-CTU oracle; each ``tile_size × tile_size`` patch
        picks its own best candidate independently.
    tile_size:
        Required when ``granularity="tile"``.  Must divide both spatial
        dimensions of ``image``.

    Returns
    -------
    reconstruction : np.ndarray
        Same shape and dtype as ``image``.
    info : dict
        Granularity-specific selection log (see module docstring for schema).

    Raises
    ------
    ValueError
        On empty candidates, missing/incompatible ``tile_size``, or unknown
        ``granularity``.
    """
    _validate_inputs(image, candidates, granularity, tile_size)

    if granularity == "image":
        return _compress_image(image, candidates, keep_ratio)
    if granularity == "tile":
        return _compress_tile(image, candidates, keep_ratio, tile_size)  # type: ignore[arg-type]
    raise ValueError(f"unknown granularity: {granularity!r}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    image: np.ndarray,
    candidates: list[Candidate],
    granularity: str,
    tile_size: int | None,
) -> None:
    if not candidates:
        raise ValueError("adaptive_compress requires at least one candidate")
    if granularity not in ("image", "tile"):
        raise ValueError(f"unknown granularity: {granularity!r}")
    if granularity == "tile":
        if tile_size is None:
            raise ValueError("granularity='tile' requires tile_size")
        H, W = image.shape[:2]
        if H % tile_size != 0 or W % tile_size != 0:
            raise ValueError(
                f"tile_size {tile_size} does not divide image shape {image.shape}"
            )


# ---------------------------------------------------------------------------
# Image-granularity
# ---------------------------------------------------------------------------

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two arrays."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def _best_candidate(
    patch: np.ndarray,
    candidates: list[Candidate],
    keep_ratio: float,
) -> tuple[str, np.ndarray, dict[str, float]]:
    """Return (chosen_name, chosen_recon, mse_per_candidate) for *patch*."""
    best_name: str | None = None
    best_recon: np.ndarray | None = None
    best_mse: float = float("inf")
    mse_map: dict[str, float] = {}

    for name, fn in candidates:
        recon = fn(patch, keep_ratio)
        err = _mse(patch, recon)
        mse_map[name] = err
        # Strict less-than: first-wins tie-break (equal MSE keeps existing best).
        if err < best_mse:
            best_mse = err
            best_name = name
            best_recon = recon

    assert best_name is not None and best_recon is not None  # guarded by validation
    return best_name, best_recon, mse_map


def _compress_image(
    image: np.ndarray,
    candidates: list[Candidate],
    keep_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    chosen_name, recon, mse_map = _best_candidate(image, candidates, keep_ratio)
    info: dict[str, Any] = {
        "granularity": "image",
        "chosen": chosen_name,
        "mse_per_candidate": mse_map,
    }
    return recon, info


# ---------------------------------------------------------------------------
# Tile-granularity
# ---------------------------------------------------------------------------

def _compress_tile(
    image: np.ndarray,
    candidates: list[Candidate],
    keep_ratio: float,
    tile_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    H, W = image.shape[:2]
    n_rows = H // tile_size
    n_cols = W // tile_size

    reconstruction = np.empty_like(image)
    chosen_grid: list[list[str]] = []
    histogram: dict[str, int] = {name: 0 for name, _ in candidates}

    for row in range(n_rows):
        row_names: list[str] = []
        r0, r1 = row * tile_size, (row + 1) * tile_size
        for col in range(n_cols):
            c0, c1 = col * tile_size, (col + 1) * tile_size
            patch = image[r0:r1, c0:c1]
            chosen_name, tile_recon, _ = _best_candidate(patch, candidates, keep_ratio)
            reconstruction[r0:r1, c0:c1] = tile_recon
            row_names.append(chosen_name)
            histogram[chosen_name] = histogram.get(chosen_name, 0) + 1
        chosen_grid.append(row_names)

    info: dict[str, Any] = {
        "granularity": "tile",
        "tile_size": tile_size,
        "chosen_grid": chosen_grid,
        "chosen_histogram": histogram,
    }
    return reconstruction, info
