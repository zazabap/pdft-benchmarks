"""TU-Berlin sketch dataset loader.

The "How Do Humans Sketch Objects?" (Eitz et al. 2012) sketch set, shipped as
a single HuggingFace-style parquet of 16000 PNG line drawings (1111x1111,
black ink on white) with an integer category `label`. Preprocessing mirrors
`load_div2k`: grayscale, centre-crop to square (a no-op for the already-square
sketches), LANCZOS resize, float32 in [0, 1].
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from ._common import ensure_dir

DEFAULT_TUBERLIN_ROOT = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/tuberlin")


def load_tuberlin(
    n_train: int,
    n_test: int,
    *,
    seed: int,
    data_root: Path = DEFAULT_TUBERLIN_ROOT,
    size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Load `n_train + n_test` TU-Berlin sketches, convert to grayscale,
    centre-crop to a square, resize to `size`x`size`, return as float32 in
    [0, 1]. Same output contract as `load_div2k`.
    """
    import pyarrow.parquet as pq

    data_root = Path(data_root)
    ensure_dir(data_root, "tuberlin")

    parquets = sorted(data_root.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no .parquet files found under {data_root}")

    # The set ships as one shard; concatenate the `image` struct column across
    # any shards present, then sample without replacement.
    blobs: list[bytes] = []
    for pqf in parquets:
        col = pq.read_table(pqf, columns=["image"]).column("image").to_pylist()
        blobs.extend(rec["bytes"] for rec in col)

    total_needed = n_train + n_test
    if len(blobs) < total_needed:
        raise ValueError(
            f"not enough sketches in {data_root}: have {len(blobs)}, need {total_needed}"
        )

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(blobs), size=total_needed, replace=False)

    out = np.empty((total_needed, size, size), dtype=np.float32)
    for i, idx in enumerate(chosen_idx):
        img = Image.open(io.BytesIO(blobs[idx])).convert("L")
        # Centre-crop to square (sketches are already square; robust to others).
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        out[i] = np.asarray(img, dtype=np.float32) / 255.0

    return out[:n_train], out[n_train:]
