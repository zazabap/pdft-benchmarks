"""DIV2K dataset loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ._common import ensure_dir

DEFAULT_DIV2K_ROOT = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR")


def load_div2k(
    n_train: int,
    n_test: int,
    *,
    seed: int,
    data_root: Path = DEFAULT_DIV2K_ROOT,
    size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Load `n_train + n_test` DIV2K PNGs, convert to grayscale, center-crop
    to a square, resize to `size`x`size`, return as float32 in [0, 1].
    """
    data_root = Path(data_root)
    ensure_dir(data_root, "div2k")

    pngs = sorted(data_root.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"no .png files found under {data_root}")

    total_needed = n_train + n_test
    if len(pngs) < total_needed:
        raise ValueError(f"not enough images in {data_root}: have {len(pngs)}, need {total_needed}")

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(pngs), size=total_needed, replace=False)
    chosen = [pngs[i] for i in chosen_idx]

    out = np.empty((total_needed, size, size), dtype=np.float32)
    for i, p in enumerate(chosen):
        img = Image.open(p).convert("L")
        # Center-crop to square.
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        out[i] = np.asarray(img, dtype=np.float32) / 255.0

    return out[:n_train], out[n_train:]
