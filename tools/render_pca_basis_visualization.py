#!/usr/bin/env python3
"""Visualise Φ for global PCA and block PCA (and DCT for reference).

Each row of Φ is an eigenvector that reshapes back into an "eigen-image"
(global PCA) or "eigen-patch" (block PCA). This script renders the top-K
rows as a grid of small images so you can see what the basis actually
looks like — analogous to the classic "eigenfaces" plot.

Outputs:
  results/quickdraw_pca_vs_block_dct/figures/pca_basis.png — 3-panel figure showing
    (top)    block_PCA top-16 eigen-patches  (8×8 each)
    (middle) DCT       top-16 basis patches  (8×8 each)  for reference
    (bottom) global PCA top-16 eigen-images  (32×32 each)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--out", default=None,
                    help="Output PNG path. None → auto-derived from --dataset.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dataset", choices=["quickdraw", "div2k_8q"],
                    default="quickdraw",
                    help="Which dataset to fit PCA against.")
    args = ap.parse_args()

    DATASET_CONFIG = {
        "quickdraw": {
            "out_default": "results/quickdraw_pca_vs_block_dct/figures/pca_basis.png",
            "img_size": 32,
        },
        "div2k_8q": {
            "out_default": "results/div2k_8q_pca_vs_block_dct/figures/pca_basis.png",
            "img_size": 256,
        },
    }
    cfg = DATASET_CONFIG[args.dataset]
    if args.out is None:
        args.out = cfg["out_default"]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.fft import idct

    from pdft_benchmarks.pca import fit_block_pca, fit_global_pca

    img_size = cfg["img_size"]
    if args.dataset == "quickdraw":
        from pdft_benchmarks.datasets import load_quickdraw
        train, _ = load_quickdraw(args.n_train, args.n_test, seed=args.seed, img_size=img_size)
    elif args.dataset == "div2k_8q":
        from pdft_benchmarks.datasets import load_div2k
        train, _ = load_div2k(args.n_train, args.n_test, seed=args.seed, size=img_size)
    else:
        raise ValueError(f"unknown dataset: {args.dataset}")

    print(f"[viz] fitting block PCA on {args.n_train} images (8×8 patches)...")
    block_basis = fit_block_pca(list(train), block=8)
    print(f"[viz] fitting global PCA on {args.n_train} images ({img_size}×{img_size})...")
    global_basis = fit_global_pca(list(train))

    # ---- Build DCT basis tiles (8×8) for reference ----
    # The k-th basis function = idct of a one-hot vector at position k.
    dct_patches = []
    for ki in range(8):
        for kj in range(8):
            # 2D basis function (ki, kj): outer product of 1D basis functions
            ei = np.zeros(8); ei[ki] = 1.0
            ej = np.zeros(8); ej[kj] = 1.0
            row = idct(ei, norm="ortho")
            col = idct(ej, norm="ortho")
            dct_patches.append(np.outer(row, col))
    dct_patches = np.stack(dct_patches, axis=0)  # (64, 8, 8)

    K = args.top_k
    rows_per_grid, cols_per_grid = 4, 4   # 4×4 = 16 cells
    assert K == rows_per_grid * cols_per_grid

    # ---- Plot ----
    fig = plt.figure(figsize=(12.0, 7.5))
    # Three column-groups side by side: block_PCA | DCT | global_PCA
    # Each is a 4×4 sub-grid.
    outer = fig.add_gridspec(1, 3, wspace=0.18, left=0.02, right=0.99, top=0.92, bottom=0.04)

    n_patches = args.n_train * (img_size // 8) ** 2
    n_pixels = img_size * img_size
    titles = [
        f"block_PCA — top-{K} eigen-patches (8×8)\n"
        f"rows of Φ_b ∈ R^{{64×64}}, fit on {n_patches} patches",
        f"block_DCT (closed-form) — first {K} basis patches (8×8)\n"
        "Φ[k,·] = cos(π(n+½)k/8) — for reference",
        f"global PCA — top-{K} eigen-images ({img_size}×{img_size})\n"
        f"rows of Φ ∈ R^{{{n_pixels}×{n_pixels}}}, fit on {args.n_train} images",
    ]
    panels = [block_basis.eigenbasis[:K], dct_patches[:K], global_basis.eigenbasis[:K]]
    sides   = [8, 8, img_size]

    for col, (title, basis_rows, side) in enumerate(zip(titles, panels, sides)):
        sub = outer[0, col].subgridspec(rows_per_grid, cols_per_grid,
                                         wspace=0.08, hspace=0.18)
        # Per-panel symmetric color range (eigenvectors can be ±)
        vmax = float(np.max(np.abs(basis_rows)))
        for k in range(K):
            ax = fig.add_subplot(sub[k // cols_per_grid, k % cols_per_grid])
            patch = basis_rows[k].reshape(side, side)
            ax.imshow(patch, cmap="seismic", vmin=-vmax, vmax=vmax,
                      interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{k}", fontsize=7, pad=1)
        # Outer panel title
        fig.text(
            0.02 + (col + 0.5) / 3 * 0.97 - 0.16,
            0.945, title, fontsize=9.5, ha="center", va="bottom",
            multialignment="center",
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[viz] wrote {out} + {out_pdf}")


if __name__ == "__main__":
    main()
