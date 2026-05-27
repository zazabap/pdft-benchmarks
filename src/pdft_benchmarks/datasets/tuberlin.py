"""TU-Berlin hand-drawn sketch loader.

20,000 sketches across 250 object categories, sourced via the
HuggingFace mirror `sdiaeyu6n/tu-berlin` (CC-BY-4.0; original source
http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/ now
404). Stored as a single Parquet shard with PNG-encoded image bytes
per row.

Mirrors the `load_div2k` API: returns `(train, test)` as
`(N_train, size, size)` and `(N_test, size, size)` float32 arrays
in [0, 1], grayscale, centre-cropped + LANCZOS-resized to `size`.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from ._common import ensure_dir

DEFAULT_TUBERLIN_ROOT = Path(
    "/home/claude-user/ParametricDFT-Benchmarks.jl/data/tuberlin"
)


def load_tuberlin(
    n_train: int,
    n_test: int,
    *,
    seed: int,
    data_root: Path = DEFAULT_TUBERLIN_ROOT,
    size: int = 256,
    parquet_name: str = "train-00000-of-00001.parquet",
) -> tuple[np.ndarray, np.ndarray]:
    """Load `n_train + n_test` TU-Berlin hand sketches.

    PNG bytes are decoded from the Parquet shard (`image.bytes` per row),
    converted to grayscale, centre-cropped to a square, LANCZOS-resized
    to `size`x`size`, normalised to float32 in [0, 1].

    Note: TU-Berlin sketches are mostly white with black strokes
    (background ≈ 1.0, ink ≈ 0.0). We invert to match the QuickDraw
    convention (background ≈ 0.0, strokes ≈ 1.0) so downstream code
    sees the same polarity across sketch datasets.
    """
    import pyarrow.parquet as pq

    data_root = Path(data_root)
    ensure_dir(data_root, "tuberlin")
    parquet_path = data_root / parquet_name
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"TU-Berlin parquet not found at {parquet_path}. "
            f"Download via:\n"
            f"  curl -fLO https://huggingface.co/datasets/sdiaeyu6n/tu-berlin/"
            f"resolve/main/data/{parquet_name}"
        )

    table = pq.read_table(parquet_path, columns=["image"])
    n_avail = table.num_rows
    total_needed = n_train + n_test
    if n_avail < total_needed:
        raise ValueError(
            f"TU-Berlin parquet has {n_avail} rows, need "
            f"{total_needed} (n_train={n_train} + n_test={n_test})"
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_avail, size=total_needed, replace=False)

    out = np.empty((total_needed, size, size), dtype=np.float32)
    image_col = table.column("image")
    for i, idx in enumerate(chosen):
        row = image_col[int(idx)].as_py()
        png_bytes = row["bytes"]
        img = Image.open(io.BytesIO(png_bytes)).convert("L")
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # Invert: white-bg/black-ink -> black-bg/white-ink (QuickDraw convention).
        out[i] = 1.0 - arr

    train = out[:n_train]
    test = out[n_train:]
    return train, test
