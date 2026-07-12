#!/usr/bin/env python3
"""Render cross-dataset PSNR(ρ) comparison for the qft_identity_reg writeup.

Three panels (one per dataset: DIV2K-8q, QuickDraw, TU-Berlin), each
plotting PSNR vs ρ ∈ {0.05, 0.10, 0.15, 0.20} for:

  - block_dct_8  (classical JPEG-style reference, no training)
  - L1 best      (our L1 reg, best lambda per dataset)
  - L1 lam=0.1, 1, 10 (the full sweep)
  - qft_identity (init, no training — pure identity operator)

Output: results/training/2_direct_training/identity_l1/figures/cross_dataset_psnr.{pdf,svg}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
PINK   = "#CC79A7"
GRAY   = "#666666"


def _l1_psnrs(dataset, rates_str):
    base = Path(f"results/training/2_direct_training/identity_l1/{dataset}/_runs")
    rows = {}
    for d in sorted(base.glob("regL1_lambda_*")):
        m = json.load(open(d / "metrics.json"))
        cell = m["qft_identity_reg"]
        lam = cell["_pdft_py"]["lam"]
        rows[lam] = [cell["metrics"][r]["mean_psnr"] for r in rates_str]
    return rows


def _bdct8_psnrs(dataset, rates_str):
    p = Path(f"results/training/2_direct_training/identity_l1/{dataset}/baselines_block_dct_8.json")
    m = json.load(open(p))["block_dct_8"]["metrics"]
    return [m[r] for r in ["0.05", "0.10", "0.15", "0.20"]]


def main():
    rates = [0.05, 0.10, 0.15, 0.20]
    rates_str = ["0.05", "0.1", "0.15", "0.2"]

    datasets = [
        ("div2k_8q",  "DIV2K-8q (m=n=8, 256×256 natural)"),
        ("quickdraw", "QuickDraw (m=n=5, 32×32 sketch)"),
        ("tuberlin",  "TU-Berlin (m=n=8, 256×256 sketch)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8),
                              constrained_layout=True)

    for i, (ds, title) in enumerate(datasets):
        ax = axes[i]
        bdct = _bdct8_psnrs(ds, rates_str)
        l1 = _l1_psnrs(ds, rates_str)

        # block_dct_8: solid black
        ax.plot(rates, bdct, marker="o", color="black", lw=1.8, markersize=6,
                 label=r"block_dct_8 (classical)", zorder=4)

        # L1 sweep — colour by lambda
        cmap = {0.1: PINK, 1.0: ORANGE, 10.0: GREEN}
        ls_map = {0.1: ":", 1.0: "--", 10.0: "-"}
        for lam, vals in sorted(l1.items()):
            if lam == 0.0:
                continue
            ax.plot(rates, vals, marker="s", color=cmap.get(lam, BLUE),
                    lw=1.4, markersize=5, linestyle=ls_map.get(lam, "-"),
                    label=rf"L1 reg, $\lambda={lam:.1f}$" if lam != 10 else r"L1 reg, $\lambda=10$",
                    zorder=3)

        ax.set_xlabel(r"keep-ratio $\rho$")
        ax.set_ylabel("test PSNR (dB)")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3, lw=0.4)
        ax.set_xticks(rates)
        ax.set_xticklabels([f"{r:.2f}" for r in rates], fontsize=9)

    OUT = Path("results/training/2_direct_training/identity_l1/figures")
    for ext in ("pdf", "svg"):
        p = OUT / f"cross_dataset_psnr.{ext}"
        fig.savefig(p)
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
