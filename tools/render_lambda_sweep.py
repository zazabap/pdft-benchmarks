#!/usr/bin/env python3
"""Render the lambda-sweep figure for the qft_identity_regularization writeup.

Two panels:
  Left  — PSNR @ rho=0.20 vs lambda (log x). Horizontal reference lines
          for qft (analytic init), qft_identity (lam=0), blocked_8.
  Right — per-gate ||T_g - I_g||_F at end of training, split into two
          overlaid violins per lambda: inner (12 gates) and outer (60
          gates). Block-structured prior is "working" if outer violins
          concentrate near zero while inner violins spread.

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

BLUE   = "#0072B2"   # outer gates
ORANGE = "#E69F00"   # inner gates (and qft analytic-init reference)
GREEN  = "#009E73"   # blocked_8 reference

RUNS_BASE = Path("results/training/2_direct_training/identity_l1/div2k_8q/_runs")
OUT_BASE  = Path("results/training/2_direct_training/identity_l1/figures")


def load_tensors_from_json(d: dict) -> list:
    """Reconstruct list of 2x2 complex jnp arrays from JSON-serialised tensors."""
    return [jnp.asarray(
                np.array(t["real"], dtype=np.float64) +
                1j * np.array(t["imag"], dtype=np.float64),
                dtype=jnp.complex128,
            ) for t in d["tensors"]]


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    rows = []   # (lam, psnr_20, inner_dists, outer_dists)
    table = qft_identity_table(8, 8)
    mask = qft_inner_mask(8, 8, 3, 3)

    for d in sorted(RUNS_BASE.glob("reg_lambda_*_W*")):
        try:
            metrics = json.load(open(d / "metrics.json"))
            cell = metrics["qft_identity_reg"]
            lam = cell["_pdft_py"]["lam"]
            psnr = cell["metrics"]["0.2"]["mean_psnr"]
            trained = json.load(open(d / "trained_qft_identity_reg.json"))
            tensors = load_tensors_from_json(trained)
            dists = [float(jnp.linalg.norm(t - i, ord="fro"))
                     for t, i in zip(tensors, table)]
            inner_dists = [x for x, mm in zip(dists, mask) if mm]
            outer_dists = [x for x, mm in zip(dists, mask) if not mm]
            rows.append((lam, psnr, inner_dists, outer_dists))
        except FileNotFoundError as e:
            print(f"skip {d.name}: missing file {e.filename}")
            continue

    if not rows:
        raise SystemExit(f"no reg_lambda_*_W* runs found under {RUNS_BASE}")

    rows.sort(key=lambda r: r[0])
    lams = [r[0] for r in rows]
    psnrs = [r[1] for r in rows]
    inner_per_lam = [r[2] for r in rows]
    outer_per_lam = [r[3] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.5),
                                    constrained_layout=True)

    # Left: PSNR vs lambda
    x_plot = [max(l, 1e-7) for l in lams]
    ax1.semilogx(x_plot, psnrs, marker="o", color=BLUE, lw=1.5,
                  markersize=6, label="qft_identity + block-masked reg")
    if lams[0] == 0:
        ax1.semilogx([x_plot[0]], [psnrs[0]], marker="*", color=BLUE,
                      markersize=14, markeredgecolor="black",
                      label="lam=0 (= qft_identity)")
    ax1.axhline(31.29, color=ORANGE, linestyle="--", lw=1.0,
                 label="qft (analytic init)")
    ax1.axhline(32.26, color=GREEN, linestyle=":", lw=1.0,
                 label="blocked_8 (warm-start ceiling)")
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel("test PSNR @ ρ=0.20 (dB)")
    ax1.legend(loc="best", fontsize=8, frameon=False)
    ax1.grid(True, which="both", alpha=0.3, lw=0.4)

    # Right: per-gate distance-from-identity, split inner/outer.
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
                          rotation=30, fontsize=8)
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel(r"$\|T_g - I_g\|_F$ (per gate, end of training)")
    ax2.grid(True, axis="y", alpha=0.3, lw=0.4)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor=ORANGE, alpha=0.6, label="inner (12)"),
                          Patch(facecolor=BLUE,   alpha=0.6, label="outer (60)")],
                 loc="best", fontsize=8, frameon=False)

    for ext in ("pdf", "svg"):
        path = OUT_BASE / f"lambda_sweep.{ext}"
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
