#!/usr/bin/env python3
"""Generate a tiny 3-panel illustration of 2D AR(1) random fields at
different correlation strengths, for the §2 AR(1) explanation block.

Output: results/quickdraw_pca_vs_block_dct/figures/ar1_examples.pdf — three 64×64 patches sampled
from a separable AR(1)-Gaussian process at ρ = 0.0, 0.5, 0.95.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pdft_benchmarks.plots.style import save_figure


def ar1_2d(side: int, rho: float, seed: int) -> np.ndarray:
    """Sample a 2D separable AR(1)-Gaussian field of shape (side, side)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((side, side))
    out = np.zeros_like(eps)
    out[0, 0] = eps[0, 0]
    for j in range(1, side):
        out[0, j] = rho * out[0, j - 1] + eps[0, j]
    for i in range(1, side):
        out[i, 0] = rho * out[i - 1, 0] + eps[i, 0]
    for i in range(1, side):
        for j in range(1, side):
            out[i, j] = (rho * out[i - 1, j] + rho * out[i, j - 1]
                          - rho * rho * out[i - 1, j - 1]
                          + eps[i, j])
    return out


def empirical_rho(patch: np.ndarray) -> float:
    """Estimate the lag-1 autocorrelation of a 2D patch as the average
    of the row-direction and column-direction lag-1 autocorrelations.

    ρ_AR ≈ (Σ (x_n - μ)(x_{n+1} - μ)) / (Σ (x_n - μ)²)
    averaged over both spatial axes.
    """
    p = patch.astype(np.float64)
    p = p - p.mean()
    var = float(np.mean(p * p)) + 1e-12
    rho_row = float(np.mean(p[:, :-1] * p[:, 1:])) / var
    rho_col = float(np.mean(p[:-1, :] * p[1:, :])) / var
    return 0.5 * (rho_row + rho_col)


def load_div2k_patch(side: int = 64, idx: int = 250) -> np.ndarray:
    """Load a centre-crop patch from DIV2K image #idx (default #0250)."""
    from PIL import Image
    base = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR")
    p = base / f"{idx:04d}.png"
    if not p.exists():
        # fallback to small repo fixture if dataset isn't checked out
        p = Path("tests/fixtures/div2k_stub/0001.png")
    img = Image.open(p).convert("L")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    h, w = arr.shape
    ci, cj = h // 2, w // 2
    return arr[ci - side // 2 : ci + side // 2,
               cj - side // 2 : cj + side // 2]


def load_quickdraw_single(side: int = 256, seed: int = 42, idx: int = 0):
    """Load a single 28×28 QuickDraw drawing.

    Returns (display_arr, native_arr) where:
      display_arr — nearest-neighbour upscaled to (side, side) just for
                    visual size; the upscale preserves block structure
                    so the displayed image accurately reflects the
                    underlying 28×28 pixels (no interpolation smoothing).
      native_arr — the original 28×28 drawing; ρ̂ is computed on this
                   so the autocorrelation reflects the *actual* dataset
                   statistic, not an artifact of upscaling.
    """
    import sys
    sys.path.insert(0, "src")
    from pdft_benchmarks.datasets import load_quickdraw

    train, _ = load_quickdraw(idx + 1, 1, seed=seed, img_size=28)
    native = train[idx].astype(np.float64)            # (28, 28)
    factor = side // 28                                # 256 // 28 = 9
    display = np.kron(native, np.ones((factor, factor)))  # (252, 252)
    return display, native


def main() -> None:
    side = 256
    seed = 42

    # ----- Row 1: 3 synthetic AR(1) + 1 DIV2K + 1 QuickDraw -----
    row1 = []
    for rho in [0.0, 0.5, 0.95]:
        x = ar1_2d(side, rho, seed)
        x = (x - x.min()) / (x.max() - x.min() + 1e-12)
        row1.append((x, f"ρ={rho:.2f}\nρ̂={empirical_rho(x):.2f}"))

    nat0 = load_div2k_patch(side, idx=250)
    row1.append((nat0, f"DIV2K #0250\nρ̂={empirical_rho(nat0):.2f}"))

    qd_disp0, qd_nat0 = load_quickdraw_single(side, seed=seed, idx=0)
    row1.append((qd_disp0,
                 f"QuickDraw #0\nρ̂={empirical_rho(qd_nat0):.2f}"))

    # ----- Row 2: 3 different DIV2K + 2 different QuickDraw -----
    row2 = []
    for div_idx in (1, 50, 100):
        nat = load_div2k_patch(side, idx=div_idx)
        row2.append((nat, f"DIV2K #{div_idx:04d}\nρ̂={empirical_rho(nat):.2f}"))

    for qd_idx in (1, 2):
        qd_disp, qd_nat = load_quickdraw_single(side, seed=seed, idx=qd_idx)
        row2.append((qd_disp,
                     f"QuickDraw #{qd_idx}\nρ̂={empirical_rho(qd_nat):.2f}"))

    # ----- Plot 2 × 5 -----
    rows = [row1, row2]
    n_cols = max(len(r) for r in rows)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.0 * n_cols, 5.0),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.18})
    for r_idx, row in enumerate(rows):
        for c_idx in range(n_cols):
            ax = axes[r_idx, c_idx]
            if c_idx < len(row):
                img, lab = row[c_idx]
                ax.imshow(img, cmap="gray", interpolation="nearest",
                          aspect="equal")
                ax.set_title(lab, fontsize=10, pad=4)
            else:
                ax.axis("off")
            ax.set_xticks([]); ax.set_yticks([])

    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.005)
    out = Path("results/quickdraw_pca_vs_block_dct/figures/ar1_examples.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out)
    print(f"[viz] wrote {out} (+ .svg)")


if __name__ == "__main__":
    main()
