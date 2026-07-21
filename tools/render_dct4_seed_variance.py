#!/usr/bin/env python3
"""Render the DCT-IV normal-training seed-variance figure (PDF + SVG, no title).

Single-method analog of `tools/render_seed_variance.py`: the DCT-IV study has
one method (Haar-random init + normal top-20% training), so one variance band
instead of the QFT study's three unfreeze orderings. Reads
<base>/seed_sweep.json (from experiments/dct4_seed_sweep.py) and draws three
panels:

  band    — test PSNR vs keep ratio rho: trained-DCT-IV mean line, shaded
            +/-sigma band, thin min-max whiskers, against the canonical
            exact-DCT-IV init (dashdot), block-FFT 8x8 (dashed), block-DCT 8x8
            (dotted) references.
  scatter — per-seed scatter of PSNR@rho=.20 (jittered) with the mean+/-sigma
            error bar overlaid.
  hist    — histogram of the 100 PSNR@rho=.20 endpoints + fitted normal curve,
            references marked.

Wong palette, linear y, no panel title in --separate mode (caption lives in the
typst/paper figure block).

Usage:
    python tools/render_dct4_seed_variance.py \
        --base results/training/2_direct_training/random_seed/dct_div2k_8q
    python tools/render_dct4_seed_variance.py --base <...> --separate
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

BLUE = "#0072B2"   # Wong blue — the trained DCT-IV
RHOS = ["0.01", "0.05", "0.1", "0.2"]
RHO_X = [0.01, 0.05, 0.10, 0.20]


def _normal_pdf(x, mu, sigma):
    if sigma <= 0:
        return np.zeros_like(x)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _ref_lines(ax, ref, horizontal_at=None):
    """Draw canonical-DCT-IV / block-FFT / block-DCT references. If
    horizontal_at is a rho key, draw them as horizontal lines at that rho;
    else draw the full PSNR-vs-rho curves."""
    if not ref:
        return
    series = [
        ("canonical_dct4", "DCT-IV (untrained)", "-.", BLUE, 1.3),
        ("block_fft_8", "block-FFT 8×8", "--", "k", 1.4),
        ("block_dct_8", "block-DCT 8×8", ":", "0.45", 1.4),
    ]
    for key, lab, ls, color, lw in series:
        if key not in ref:
            continue
        psnr = ref[key].get("psnr", ref[key])
        if horizontal_at is not None:
            if horizontal_at in psnr:
                ax.axhline(psnr[horizontal_at], color=color, ls=ls, lw=lw, label=lab)
        else:
            ax.plot(RHO_X, [psnr[r] for r in RHOS], ls, color=color, lw=lw,
                    label=lab, zorder=1)


def _draw_band(ax, agg, ref):
    present = [(rx, r) for rx, r in zip(RHO_X, RHOS) if r in agg]
    xs = [rx for rx, _ in present]
    mean = np.array([agg[r]["mean"] for _, r in present])
    std = np.array([agg[r]["std"] for _, r in present])
    lo = np.array([agg[r]["min"] for _, r in present])
    hi = np.array([agg[r]["max"] for _, r in present])
    n = agg[present[0][1]]["n"] if present else 0
    ax.fill_between(xs, mean - std, mean + std, color=BLUE, alpha=0.20, lw=0)
    ax.plot(xs, mean, "-", color=BLUE, marker="o", ms=4, lw=1.8,
            label=f"trained DCT-IV  (n={n})")
    ax.vlines(xs, lo, hi, color=BLUE, lw=0.8, alpha=0.6)
    _ref_lines(ax, ref)
    ax.set_xlabel(r"keep ratio $\rho$", fontsize=8.5)
    ax.set_ylabel("test PSNR (dB)", fontsize=8.5)
    ax.set_xticks(RHO_X)
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def _draw_scatter(ax, per_seed, ref):
    rng = np.random.default_rng(0)
    vals = np.array([v["0.2"] for v in per_seed.values() if "0.2" in v])
    if vals.size:
        jitter = (rng.random(vals.size) - 0.5) * 0.5
        ax.scatter(jitter, vals, s=9, color=BLUE, alpha=0.55, edgecolors="none")
        m, sd = float(vals.mean()), float(vals.std(ddof=1) if vals.size > 1 else 0.0)
        ax.errorbar(0, m, yerr=sd, fmt="_", color="black", ms=22, lw=1.6, capsize=4,
                    zorder=5)
    _ref_lines(ax, ref, horizontal_at="0.2")
    ax.set_xticks([0])
    ax.set_xticklabels(["trained DCT-IV"], fontsize=7.5)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel(r"test PSNR @ $\rho=.20$  (dB)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def _draw_hist(ax, per_seed, ref):
    vals = np.array([v["0.2"] for v in per_seed.values() if "0.2" in v])
    if vals.size:
        lo, hi = float(vals.min()), float(vals.max())
        pad = max((hi - lo) * 0.15, 0.05)
        bins = np.linspace(lo - pad, hi + pad, 24)
        xline = np.linspace(lo - pad, hi + pad, 200)
        ax.hist(vals, bins=bins, density=True, color=BLUE, alpha=0.30)
        m, sd = float(vals.mean()), float(vals.std(ddof=1) if vals.size > 1 else 0.0)
        ax.plot(xline, _normal_pdf(xline, m, sd), color=BLUE, lw=1.6,
                label=rf"$\mu={m:.2f},\ \sigma={sd:.2f}$")
    _ref_lines(ax, ref, horizontal_at=None) if False else None
    # references as vertical lines at rho=.20
    if ref:
        for key, color, ls in (("canonical_dct4", BLUE, "-."),
                               ("block_fft_8", "k", "--"),
                               ("block_dct_8", "0.45", ":")):
            if key in ref:
                psnr = ref[key].get("psnr", ref[key])
                if "0.2" in psnr:
                    ax.axvline(psnr["0.2"], color=color, ls=ls, lw=1.2)
    ax.set_xlabel(r"test PSNR @ $\rho=.20$  (dB)", fontsize=8.5)
    ax.set_ylabel("density", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def _save(fig, figdir, stem):
    for ext in ("pdf", "svg"):
        out = figdir / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[render] wrote {out}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="random_seed/dct_<dataset> dir holding seed_sweep.json.")
    ap.add_argument("--reference", default=None,
                    help="reference JSON (canonical_dct4 / block_fft_8 / "
                         "block_dct_8). Default <base>/reference/classical_dct4.json.")
    ap.add_argument("--separate", action="store_true", default=False)
    args = ap.parse_args()

    base = Path(args.base)
    sweep_path = base / "seed_sweep.json"
    if not sweep_path.exists():
        print(f"[render] no {sweep_path}", file=sys.stderr)
        return 2
    sweep = json.loads(sweep_path.read_text())
    per_seed = sweep.get("per_seed", {})
    agg = sweep.get("agg", {})
    if not per_seed:
        print("[render] seed_sweep.json has no per_seed data yet", file=sys.stderr)
        return 2

    ref = None
    rpath = Path(args.reference) if args.reference else base / "reference" / "classical_dct4.json"
    if rpath.exists():
        ref = json.loads(rpath.read_text())

    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    if args.separate:
        specs = [
            ("seed_variance_band", (4.9, 3.7), lambda ax: _draw_band(ax, agg, ref)),
            ("seed_variance_scatter", (3.6, 3.7), lambda ax: _draw_scatter(ax, per_seed, ref)),
            ("seed_variance_hist", (4.6, 3.7), lambda ax: _draw_hist(ax, per_seed, ref)),
        ]
        for stem, figsize, draw in specs:
            fig, ax = plt.subplots(figsize=figsize)
            draw(ax)
            fig.tight_layout()
            _save(fig, figdir, stem)
        return 0

    fig, (axL, axM, axR) = plt.subplots(
        1, 3, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [1.3, 0.85, 1.0]})
    _draw_band(axL, agg, ref)
    _draw_scatter(axM, per_seed, ref)
    _draw_hist(axR, per_seed, ref)
    fig.tight_layout()
    _save(fig, figdir, "seed_variance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
