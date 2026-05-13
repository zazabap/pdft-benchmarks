#!/usr/bin/env python3
"""Render the L1-sweep figure for the qft_identity_regularization writeup.

Two panels (mirrors render_lambda_sweep.py but adapted for L1's
non-block-aligned sparsity):
  Left  — PSNR @ rho=0.20 vs lambda (log x), reference lines for
          qft (analytic), qft_identity, blocked_8.
  Right — per-gate ||T_g - I_g||_F at end of training. Single violin
          per lambda (no inner/outer split — L1 has no block-aware
          mask). Color-codes each violin: gates inside blocked_8's
          12-inner mask (orange), gates outside (blue). If L1 finds
          blocked_8-like structure, the orange dots dominate the
          spread; if L1 finds a different basin, orange and blue may
          be mixed or the spread may be sub-12 in count.

PDF + SVG outputs per the repo's figure convention.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pdft_benchmarks.identity_reg import qft_identity_table, qft_inner_mask

BLUE   = "#0072B2"   # outer-per-blocked_8
ORANGE = "#E69F00"   # inner-per-blocked_8
GREEN  = "#009E73"

RUNS_BASE = Path("results/qft_identity_init/div2k_8q/_runs")
OUT_BASE  = Path("results/qft_identity_init/figures")


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    rows = []
    table = qft_identity_table(8, 8)
    mask = qft_inner_mask(8, 8, 3, 3)

    for d in sorted(RUNS_BASE.glob("regL1_lambda_*")):
        metrics = json.load(open(d / "metrics.json"))
        cell = metrics["qft_identity_reg"]
        lam = cell["_pdft_py"]["lam"]
        psnr = cell["metrics"]["0.2"]["mean_psnr"]
        trained = json.load(open(d / "trained_qft_identity_reg.json"))
        tensors = [jnp.asarray(
                       np.array(t["real"], dtype=np.float64) +
                       1j * np.array(t["imag"], dtype=np.float64),
                       dtype=jnp.complex128)
                   for t in trained["tensors"]]
        dists = [float(jnp.linalg.norm(t - i, ord="fro"))
                 for t, i in zip(tensors, table)]
        inner_dists = [x for x, mm in zip(dists, mask) if mm]
        outer_dists = [x for x, mm in zip(dists, mask) if not mm]
        rows.append((lam, psnr, inner_dists, outer_dists))

    if not rows:
        raise SystemExit(f"no regL1_lambda_* runs found under {RUNS_BASE}")

    rows.sort(key=lambda r: r[0])
    lams = [r[0] for r in rows]
    psnrs = [r[1] for r in rows]
    inner_per_lam = [r[2] for r in rows]
    outer_per_lam = [r[3] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.5),
                                    constrained_layout=True)

    # Left: PSNR vs lambda
    x_plot = lams
    ax1.semilogx(x_plot, psnrs, marker="o", color=BLUE, lw=1.5,
                  markersize=7, label="qft_identity + L1 reg")
    ax1.axhline(31.66, color="#888888", linestyle="-.", lw=1.0,
                 label="qft_identity (no reg)")
    ax1.axhline(31.29, color=ORANGE, linestyle="--", lw=1.0,
                 label="qft (analytic init)")
    ax1.axhline(32.26, color=GREEN, linestyle=":", lw=1.0,
                 label="blocked_8 (warm-start ceiling)")
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel("test PSNR @ ρ=0.20 (dB)")
    ax1.legend(loc="best", fontsize=8, frameon=False)
    ax1.grid(True, which="both", alpha=0.3, lw=0.4)

    # Right: per-gate distance-from-identity, split by blocked_8's inner/outer mask
    positions = np.arange(len(lams), dtype=float)
    offset = 0.2
    p_inner = ax2.violinplot(inner_per_lam, positions=positions - offset,
                              widths=0.35, showmedians=True, showextrema=True)
    p_outer = ax2.violinplot(outer_per_lam, positions=positions + offset,
                              widths=0.35, showmedians=True, showextrema=True)
    for pc in p_inner["bodies"]:
        pc.set_facecolor(ORANGE); pc.set_alpha(0.6)
    for pc in p_outer["bodies"]:
        pc.set_facecolor(BLUE); pc.set_alpha(0.6)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([f"{l:.0e}" if l > 0 else "0" for l in lams],
                          rotation=0, fontsize=9)
    ax2.set_xlabel(r"$\lambda$ (L1)")
    ax2.set_ylabel(r"$\|T_g - I_g\|_F$ (per gate, end of training)")
    ax2.grid(True, axis="y", alpha=0.3, lw=0.4)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor=ORANGE, alpha=0.6,
                                label="blocked_8 inner (12)"),
                          Patch(facecolor=BLUE, alpha=0.6,
                                label="blocked_8 outer (60)")],
                 loc="best", fontsize=8, frameon=False)

    for ext in ("pdf", "svg"):
        path = OUT_BASE / f"lambda_sweep_L1.{ext}"
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
