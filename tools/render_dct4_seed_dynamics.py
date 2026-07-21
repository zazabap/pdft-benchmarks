#!/usr/bin/env python3
"""Render the DCT-IV study's per-seed training-dynamics figure (PDF + SVG, no
title): every seed's top-k MSE descent (one thin curve per seed) on a shared
y-range, from the per-seed loss traces written by experiments/dct4_seed_sweep.py
(`_runs/seed_<NNN>_trace.json`). Mirrors the QFT seed writeup's dynamics panel,
collapsed to one panel (single method, no orderings).

Usage:
    python tools/render_dct4_seed_dynamics.py \
        --base results/training/2_direct_training/random_seed/dct_div2k_8q
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

BLUE = "#0072B2"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--logy", action="store_true", default=False,
                    help="(diagnostic only) log y; the committed figure uses "
                         "linear y per the repo figure convention.")
    args = ap.parse_args()

    base = Path(args.base)
    runs = base / "_runs"
    traces = sorted(runs.glob("seed_*_trace.json"))
    if not traces:
        print(f"[dyn] no traces under {runs}", file=sys.stderr)
        return 2

    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    n = 0
    for tp in traces:
        d = json.loads(tp.read_text())
        lh = d.get("loss_history", [])
        if not lh:
            continue
        ax.plot(np.arange(len(lh)), lh, color=BLUE, lw=0.4, alpha=0.35)
        n += 1
    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step", fontsize=8.5)
    ax.set_ylabel(r"top-$k$ MSE loss", fontsize=8.5)
    ax.set_title("")
    print(f"[dyn] plotted {n} seed traces")

    for ext in ("pdf", "svg"):
        out = base / "figures" / f"seed_dynamics.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"[dyn] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
