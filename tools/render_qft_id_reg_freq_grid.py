#!/usr/bin/env python3
"""Frequency-space + reconstruction grid for the qft_identity_reg writeup.

For each dataset, produce a 3-row x 3-col figure:
  rows = methods: qft_identity (no reg), block-masked reg λ=1, L1 reg λ=10
  cols = (test image, |T(x)| frequency magnitude in log10, recon @ ρ=0.20)

Outputs:
  results/training/2_direct_training/identity_l1/figures/freq_recon_<dataset>.{pdf,svg}
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["div2k_8q", "quickdraw"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--img-idx", type=int, default=0,
                    help="Test-split image index (0..n_test-1).")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pdft
    import pdft.io  # noqa: F401  -- register pdft.io for compress/recover
    from pdft_benchmarks.bases import qft_identity_basis

    if args.dataset == "div2k_8q":
        from pdft_benchmarks.datasets.div2k import load_div2k
        m = n = 8
        train, test = load_div2k(n_train=500, n_test=50, seed=42, size=2**m)
        runs_base = Path("results/training/2_direct_training/identity_l1/div2k_8q/_runs")
        method_paths = {
            "qft_identity (λ=0)": None,
            "Block-masked (λ=1)": runs_base / "reg_lambda_1e+00_W10" / "trained_qft_identity_reg.json",
            "L1 (λ=10)": runs_base / "regL1_lambda_1e+01" / "trained_qft_identity_reg.json",
        }
    else:  # quickdraw
        from pdft_benchmarks.datasets.quickdraw import load_quickdraw
        m = n = 5
        train, test = load_quickdraw(n_train=500, n_test=50, seed=42, img_size=2**m)
        runs_base = Path("results/training/2_direct_training/identity_l1/quickdraw/_runs")
        method_paths = {
            "qft_identity (λ=0)": None,
            "L1 (λ=10)": runs_base / "regL1_lambda_1e+01" / "trained_qft_identity_reg.json",
        }

    print(f"[freq_grid] dataset={args.dataset}, m=n={m}, "
          f"n_train={len(train)}, n_test={len(test)}, img_idx={args.img_idx}")
    img_np = np.asarray(test[args.img_idx], dtype=np.float64)
    img_jnp = jnp.asarray(img_np, dtype=jnp.complex128)

    def load_basis(path):
        if path is None:
            return qft_identity_basis(m, n)
        d = json.load(open(path))
        tensors = [jnp.asarray(
                       np.array(t["real"]) + 1j * np.array(t["imag"]),
                       dtype=jnp.complex128)
                   for t in d["tensors"]]
        return pdft.QFTBasis(m=m, n=n, tensors=tensors)

    rho = 0.20
    discard_ratio = 1.0 - rho

    rows = []   # (method_name, freq_log_mag, recon, psnr)
    for name, path in method_paths.items():
        basis = load_basis(path)
        freq = basis.forward_transform(img_jnp)
        freq_log = np.log10(np.abs(np.asarray(freq)) + 1e-9)
        compressed = pdft.io.compress(basis, img_np, ratio=discard_ratio)
        recovered = pdft.io.recover(basis, compressed)
        recovered = np.asarray(recovered).real
        mse = float(np.mean((img_np - recovered) ** 2))
        psnr = 10 * np.log10(1.0 / (mse + 1e-30)) if mse > 0 else float("inf")
        rows.append((name, freq_log, recovered, psnr))
        print(f"[freq_grid] {name}: PSNR @ ρ=0.20 = {psnr:.2f} dB")

    n_methods = len(rows)
    fig, axes = plt.subplots(n_methods, 3, figsize=(7.5, 2.4 * n_methods),
                              constrained_layout=True)
    if n_methods == 1:
        axes = axes[np.newaxis, :]

    # Shared vmin/vmax for the frequency-mag panels so the colour scale is comparable.
    vmin = min(r[1].min() for r in rows)
    vmax = max(r[1].max() for r in rows)

    for i, (name, freq_log, recon, psnr) in enumerate(rows):
        # Col 0: input
        axes[i, 0].imshow(img_np, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(name, fontsize=10)
        if i == 0:
            axes[i, 0].set_title("input", fontsize=10)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        # Col 1: |T(x)| log magnitude
        im = axes[i, 1].imshow(freq_log, cmap="viridis", vmin=vmin, vmax=vmax)
        if i == 0:
            axes[i, 1].set_title(r"$\log_{10}|T(x)|$", fontsize=10)
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])

        # Col 2: recon @ rho=0.20
        axes[i, 2].imshow(np.clip(recon, 0, 1), cmap="gray", vmin=0, vmax=1)
        if i == 0:
            axes[i, 2].set_title(f"reconstruction @ ρ={rho:.2f}", fontsize=10)
        axes[i, 2].set_xlabel(f"PSNR {psnr:.2f} dB", fontsize=9)
        axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])

    OUT_BASE = Path("results/training/2_direct_training/identity_l1/figures")
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        p = OUT_BASE / f"freq_recon_{args.dataset}.{ext}"
        fig.savefig(p)
        print(f"[freq_grid] wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
