"""QuickDraw dataset loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._common import ensure_dir

DEFAULT_QUICKDRAW_ROOT = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/quickdraw")


def load_quickdraw(
    n_train: int,
    n_test: int,
    *,
    seed: int,
    data_root: Path = DEFAULT_QUICKDRAW_ROOT,
    img_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Load `n_train + n_test` 28x28 QuickDraw drawings, center-pad with zeros
    to `img_size`, return as float32 in [0, 1].

    Mirrors `pad_to_power_of_two` in `ParametricDFT-Benchmarks.jl/data_loading.jl`.
    """
    data_root = Path(data_root)
    ensure_dir(data_root, "quickdraw")

    npy_files = sorted(data_root.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"no .npy files found under {data_root}")

    total_needed = n_train + n_test
    rng = np.random.default_rng(seed)

    per_cat_target = (total_needed + len(npy_files) - 1) // len(npy_files)

    picked: list[np.ndarray] = []
    for npy in npy_files:
        arr = np.load(npy)
        if arr.ndim != 2 or arr.shape[1] != 784:
            raise ValueError(f"{npy} has unexpected shape {arr.shape}; expected (N, 784)")
        n_avail = arr.shape[0]
        n_pick = min(per_cat_target, n_avail)
        idx = rng.choice(n_avail, size=n_pick, replace=False)
        picked.append(arr[idx].reshape(n_pick, 28, 28))

    pool = np.concatenate(picked, axis=0).astype(np.float32) / 255.0
    if pool.shape[0] < total_needed:
        raise ValueError(
            f"not enough images in {data_root}: have {pool.shape[0]}, need {total_needed}"
        )

    perm = rng.permutation(pool.shape[0])
    pool = pool[perm[:total_needed]]

    if img_size != 28:
        pool = _pad_batch(pool, img_size)

    train = pool[:n_train]
    test = pool[n_train:]
    return train, test


def _pad_batch(images: np.ndarray, target_size: int) -> np.ndarray:
    """Center-pad each (h, w) image with zeros to (target_size, target_size)."""
    n, h, w = images.shape
    if h > target_size or w > target_size:
        raise ValueError(
            f"can't center-pad ({h},{w}) into ({target_size},{target_size}) — image larger than target"
        )
    out = np.zeros((n, target_size, target_size), dtype=images.dtype)
    y_off = (target_size - h) // 2
    x_off = (target_size - w) // 2
    out[:, y_off:y_off + h, x_off:x_off + w] = images
    return out
