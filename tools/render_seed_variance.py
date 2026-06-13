#!/usr/bin/env python3
"""Render the random-seed unfreeze variance figure (PDF + SVG, no title).

Reads <base>/seed_sweep.json (from experiments/qft_seed_sweep.py /
tools/run_seed_sweep.py) and draws three panels:

  Left   — test PSNR vs keep ratio rho: per-ordering mean line, shaded +/-sigma
           band, and thin min-max whiskers; the block-FFT 8x8 baseline as a
           dashed black reference (the claim's bar), block-DCT 8x8 dotted.
  Middle — per-seed scatter of PSNR@rho=.20 per ordering (jittered) with the
           mean+/-sigma band overlaid: shows how tight the seed turbulence is.
  Right  — histogram of the PSNR@rho=.20 endpoints per ordering with a fitted
           normal curve (the "gaussian distribution" view), block-FFT marked.

Wong palette + one line style per ordering; linear y. Classical baselines are
optional (skipped if --classical is absent, e.g. QuickDraw pilot).

Usage:
    python tools/render_seed_variance.py \
        --base results/training/2_direct_training/random_seed/div2k_8q
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Wong palette + line style, one per ordering (matches render_qft_unfreeze).
STYLE = {
    "bg": ("#0072B2", "-",  "block-growth"),
    "lr": ("#E69F00", "--", "left→right"),
    "rl": ("#009E73", "-.", "right→left"),
}
RHOS = ["0.05", "0.1", "0.15", "0.2"]
RHO_X = [0.05, 0.10, 0.15, 0.20]
DEFAULT_CLASSICAL = Path("results/training/2_direct_training/unfreeze/"
                         "reference/classical_div2k.json")


def _normal_pdf(x, mu, sigma):
    if sigma <= 0:
        return np.zeros_like(x)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="random_seed/<dataset> dir holding seed_sweep.json.")
    ap.add_argument("--classical", default=str(DEFAULT_CLASSICAL),
                    help="classical baseline JSON (block_fft_8 / block_dct_8). "
                         "Skipped if the file is absent.")
    args = ap.parse_args()

    base = Path(args.base)
    sweep_path = base / "seed_sweep.json"
    if not sweep_path.exists():
        print(f"[render] no {sweep_path}", file=sys.stderr)
        return 2
    sweep = json.loads(sweep_path.read_text())
    per_ordering = sweep["per_ordering"]
    orderings = [o for o in ("bg", "lr", "rl") if o in per_ordering
                 and per_ordering[o]["per_seed"]]
    if not orderings:
        print("[render] no populated orderings in seed_sweep.json", file=sys.stderr)
        return 2

    classical = None
    cpath = Path(args.classical)
    if cpath.exists():
        classical = json.loads(cpath.read_text())

    fig, (axL, axM, axR) = plt.subplots(
        1, 3, figsize=(13.0, 3.8), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

    # ---- Panel L: PSNR vs rho, mean +/- sigma band + min-max whiskers. ------
    for o in orderings:
        color, ls, lab = STYLE[o]
        agg = per_ordering[o]["agg"]
        present = [(rx, r) for rx, r in zip(RHO_X, RHOS) if r in agg]
        xs = [rx for rx, _ in present]
        mean = np.array([agg[r]["mean"] for _, r in present])
        std = np.array([agg[r]["std"] for _, r in present])
        lo = np.array([agg[r]["min"] for _, r in present])
        hi = np.array([agg[r]["max"] for _, r in present])
        n = present and agg[present[0][1]]["n"] or 0
        axL.fill_between(xs, mean - std, mean + std, color=color, alpha=0.20, lw=0)
        axL.plot(xs, mean, ls, color=color, marker="o", ms=4, lw=1.8,
                 label=f"{lab}  (n={n})")
        # min-max whiskers
        axL.vlines(xs, lo, hi, color=color, lw=0.8, alpha=0.6)
    if classical:
        axL.plot(RHO_X, [classical["block_fft_8"]["psnr"][r] for r in RHOS],
                 "k--", lw=1.4, label="block-FFT 8×8", zorder=1)
        axL.plot(RHO_X, [classical["block_dct_8"]["psnr"][r] for r in RHOS],
                 ":", color="0.45", lw=1.4, label="block-DCT 8×8", zorder=1)
    axL.set_xlabel(r"keep ratio $\rho$", fontsize=8.5)
    axL.set_ylabel("test PSNR (dB)", fontsize=8.5)
    axL.set_xticks(RHO_X)
    axL.legend(frameon=False, fontsize=7, loc="upper left")
    axL.set_title("mean $\\pm$ $\\sigma$ across seeds", fontsize=9)

    # ---- Panel M: per-seed scatter @ rho=.20 + mean+/-sigma. -----------------
    rng = np.random.default_rng(0)
    fft20 = classical["block_fft_8"]["psnr"]["0.2"] if classical else None
    for i, o in enumerate(orderings):
        color, _ls, lab = STYLE[o]
        vals = np.array([v["0.2"] for v in per_ordering[o]["per_seed"].values()
                         if "0.2" in v])
        if vals.size == 0:
            continue
        jitter = (rng.random(vals.size) - 0.5) * 0.5
        axM.scatter(np.full(vals.size, i) + jitter, vals, s=9, color=color,
                    alpha=0.55, edgecolors="none")
        m, sd = float(vals.mean()), float(vals.std(ddof=1) if vals.size > 1 else 0.0)
        axM.errorbar(i, m, yerr=sd, fmt="_", color="black", ms=18, lw=1.6, capsize=4,
                     zorder=5)
    if fft20 is not None:
        axM.axhline(fft20, color="k", ls="--", lw=1.2, label="block-FFT 8×8")
        axM.legend(frameon=False, fontsize=7, loc="lower right")
    axM.set_xticks(range(len(orderings)))
    axM.set_xticklabels([STYLE[o][2] for o in orderings], fontsize=7.5)
    axM.set_ylabel(r"test PSNR @ $\rho=.20$  (dB)", fontsize=8.5)
    axM.set_title("per-seed spread @ $\\rho=.20$", fontsize=9)

    # ---- Panel R: histogram @ rho=.20 + fitted normal. ----------------------
    all_vals = []
    for o in orderings:
        vals = np.array([v["0.2"] for v in per_ordering[o]["per_seed"].values()
                         if "0.2" in v])
        all_vals.append(vals)
    flat = np.concatenate([v for v in all_vals if v.size])
    if flat.size:
        lo, hi = float(flat.min()), float(flat.max())
        pad = max((hi - lo) * 0.15, 0.02)
        bins = np.linspace(lo - pad, hi + pad, 24)
        xline = np.linspace(lo - pad, hi + pad, 200)
        for o, vals in zip(orderings, all_vals):
            if vals.size == 0:
                continue
            color, _ls, lab = STYLE[o]
            axR.hist(vals, bins=bins, density=True, color=color, alpha=0.30,
                     label=f"{lab}")
            m, sd = float(vals.mean()), float(vals.std(ddof=1) if vals.size > 1 else 0.0)
            axR.plot(xline, _normal_pdf(xline, m, sd), color=color, lw=1.6)
        if fft20 is not None and lo - pad <= fft20 <= hi + pad:
            axR.axvline(fft20, color="k", ls="--", lw=1.2)
    axR.set_xlabel(r"test PSNR @ $\rho=.20$  (dB)", fontsize=8.5)
    axR.set_ylabel("density", fontsize=8.5)
    axR.legend(frameon=False, fontsize=7, loc="upper left")
    axR.set_title("endpoint distribution", fontsize=9)

    fig.tight_layout()
    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = figdir / f"seed_variance.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[render] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
