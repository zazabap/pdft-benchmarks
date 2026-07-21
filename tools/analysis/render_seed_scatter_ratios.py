#!/usr/bin/env python3
"""Figure 2(b) across compression rates: per-seed PSNR scatter at several keep
ratios, one panel per rate.

Companion to render_seed_variance.py's single-rate scatter (rho=.20). Reads the
per-ordering multi-ratio re-evaluation (reference/_multiratio_<ordering>.json,
produced from the saved trained_seed_*.json operators) and draws one jittered
per-seed scatter panel per keep ratio, with the mean+/-sigma error bar overlaid
and the block-FFT 8x8 / block-DCT 8x8 references at that same rate.

Wong palette, one colour per ordering; linear y; PDF + SVG, no figure title
(each panel is labelled by its keep ratio — content, not a title).

Usage:
    python tools/analysis/render_seed_scatter_ratios.py \
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

STYLE = {
    "bg": ("#0072B2", "block-growth"),
    "lr": ("#E69F00", "left→right"),
    "rl": ("#009E73", "right→left"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="random_seed/<dataset> dir holding reference/_multiratio_*.json.")
    ap.add_argument("--ratios", default="0.01,0.05,0.1,0.2",
                    help="Keep ratios (panels), left-to-right.")
    ap.add_argument("--paper-style", action="store_true", default=False,
                    help="Publication style + paper-width figsize; PDF to figures/paper/.")
    args = ap.parse_args()
    if args.paper_style:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parent))
        from paper_style import apply_paper_style, PAPER_TEXTWIDTH
        apply_paper_style()

    base = Path(args.base)
    ratios = [r.strip() for r in args.ratios.split(",") if r.strip()]
    merged_path = base / "reference" / "seed_scatter_ratios.json"
    if not merged_path.exists():
        print(f"[render] no {merged_path} (merge the _multiratio_*.json partials "
              f"first)", file=sys.stderr)
        return 2
    merged = json.loads(merged_path.read_text())
    per_ordering = merged["per_ordering"]
    orderings = [o for o in ("bg", "lr", "rl") if o in per_ordering]
    classical = merged.get("classical")

    _w = PAPER_TEXTWIDTH if args.paper_style else 3.4 * len(ratios)
    _h = 2.4 if args.paper_style else 3.7
    fig, axes = plt.subplots(1, len(ratios), figsize=(_w, _h), squeeze=False)
    axes = axes[0]
    rng = np.random.default_rng(0)
    for ax, r in zip(axes, ratios):
        for i, o in enumerate(orderings):
            color, _lab = STYLE[o]
            vals = np.array([v[r] for v in per_ordering[o].values() if r in v])
            if vals.size == 0:
                continue
            jitter = (rng.random(vals.size) - 0.5) * 0.5
            ax.scatter(np.full(vals.size, i) + jitter, vals, s=9, color=color,
                       alpha=0.55, edgecolors="none")
            m = float(vals.mean())
            sd = float(vals.std(ddof=1) if vals.size > 1 else 0.0)
            ax.errorbar(i, m, yerr=sd, fmt="_", color="black", ms=18, lw=1.6,
                        capsize=4, zorder=5)
        if classical:
            if "block_dct_8" in classical and r in classical["block_dct_8"]:
                ax.axhline(classical["block_dct_8"][r], color="0.45", ls=":", lw=1.3,
                           label="block-DCT 8×8")
            if "block_fft_8" in classical and r in classical["block_fft_8"]:
                ax.axhline(classical["block_fft_8"][r], color="k", ls="--", lw=1.2,
                           label="block-FFT 8×8")
        ax.set_xticks(range(len(orderings)))
        ax.set_xticklabels([STYLE[o][1] for o in orderings], fontsize=7, rotation=20,
                           ha="right")
        ax.set_title(f"$\\rho = {r}$", fontsize=9)
    axes[0].set_ylabel("test PSNR (dB)", fontsize=8.5)
    if classical:
        axes[-1].legend(frameon=False, fontsize=7, loc="lower left")

    fig.tight_layout()
    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    if args.paper_style:
        pdir = figdir / "paper"; pdir.mkdir(parents=True, exist_ok=True)
        out = pdir / "seed_scatter_ratios.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"[render] wrote {out}")
    else:
        for ext in ("pdf", "svg"):
            out = figdir / f"seed_scatter_ratios.{ext}"
            fig.savefig(out, bbox_inches="tight")
            print(f"[render] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
