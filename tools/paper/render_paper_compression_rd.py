#!/usr/bin/env python3
"""Render the paper's §5.2 QuickDraw rate-distortion figure.

A single-column, paper-styled variant of ``render_compression_rd.py``. The
x-axis is compressed size as a percentage of the raw image, so the two reference
rules, a horizontal cut at the dataset's quality reading and a vertical rule at
40% of raw, both land
on grid lines and read as clean crosshairs. The crossings are marked with values:
the compressed size (%) each basis needs to reach that quality (horizontal), and each
basis's test PSNR at 40% of raw (vertical). 40% is used for the vertical reading
because both curves are well-sampled there, so the points sit on the grid line
and on the curves; the 50%-of-raw budget floats the block-DCT point off the line.

Reads results/training/6_dataset_compression/quickdraw_5q/{rd_curves,headline_50pct}.json.
Writes figures/rd_quickdraw_paper.{pdf,svg} in that dataset dir. Copy the PDF into
the paper at figures/benchmarks/compression/rd_quickdraw.pdf.

Style per CLAUDE.md: Wong palette, one colour + one line style per basis,
linear axes, no figure-level title, PDF+SVG only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from pdft_benchmarks.plots.style import WONG, save_figure, set_paper_rcparams

set_paper_rcparams()

BASE_DIR = Path("results/training/6_dataset_compression")

BLUE, GREEN = WONG["blue"], WONG["green"]  # Wong palette

# Per-dataset: the results dir, the trained contender's key, the fixed-PSNR
# reading and the fixed-size reading. DIV2K's readings sit higher because the
# whole curve does; its contender is the block-wrapped real Rich variant.
DATASETS = {
    "quickdraw_5q": dict(rich_key="real_rich", psnr_cut=35.0, v_pct=40.0,
                         label="QuickDraw", out_name="rd_quickdraw_paper"),
    "div2k_8q":     dict(rich_key="real_rich_8", psnr_cut=38.0, v_pct=40.0,
                         label="DIV2K", out_name="rd_div2k_paper"),
}


def pareto(points):
    """Max-PSNR frontier: sort by bytes, keep points that improve PSNR."""
    pts = sorted(points, key=lambda p: p["bytes_per_image"])
    front, best = [], -1e9
    for p in pts:
        if p["test"]["mean_psnr"] > best:
            best = p["test"]["mean_psnr"]
            front.append(p)
    return front


def bytes_at_psnr(front, target):
    """Linear-interpolate bytes/image needed to reach a target PSNR."""
    xs = [p["bytes_per_image"] for p in front]
    ys = [p["test"]["mean_psnr"] for p in front]
    return float(np.interp(target, ys, xs))


def psnr_at_bytes(front, target_bytes):
    """Linear-interpolate PSNR at a target bytes/image."""
    xs = [p["bytes_per_image"] for p in front]
    ys = [p["test"]["mean_psnr"] for p in front]
    return float(np.interp(target_bytes, xs, ys))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="quickdraw_5q",
                    help="Which compression run to plot.")
    ap.add_argument("--out", default=None,
                    help="optional extra path to also write the PDF to "
                         "(e.g. the paper's figures/benchmarks/compression/rd_quickdraw.pdf)")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    DDIR = BASE_DIR / args.dataset
    PSNR_CUT, V_PCT = cfg["psnr_cut"], cfg["v_pct"]

    rd = json.loads((DDIR / "rd_curves.json").read_text())
    curves, meta = rd["curves"], rd["meta"]
    raw_bpi = meta["raw_bytes_per_image"]

    rich = pareto(curves[cfg["rich_key"]])
    dct = pareto(curves["block_dct_8"])

    # Fixed-PSNR reading: bytes/image at PSNR_CUT.
    x_rich = bytes_at_psnr(rich, PSNR_CUT)
    x_dct = bytes_at_psnr(dct, PSNR_CUT)
    saved_pct = 100.0 * (1.0 - x_rich / x_dct)

    # Fixed-size vertical reading at 40% of raw -- a clean grid point where both
    # curves are well-sampled, so the two points land on the x=40 grid line and
    # on their curves (the 50% budget floats the block-DCT marker off the line,
    # since its best in-budget config sits near 47%).
    v_bytes = V_PCT / 100.0 * raw_bpi
    y_rich = psnr_at_bytes(rich, v_bytes)
    y_dct = psnr_at_bytes(dct, v_bytes)
    d_psnr = y_rich - y_dct

    print(f"{cfg['label']} @ {PSNR_CUT:.0f} dB:  rich={x_rich:.0f} B/img  "
          f"dct={x_dct:.0f} B/img  -> {saved_pct:.1f}% fewer bytes")
    print(f"{cfg['label']} @ {V_PCT:.0f}%-of-raw:  rich={y_rich:.2f} dB  "
          f"dct={y_dct:.2f} dB  -> +{d_psnr:.2f} dB")

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # x-axis is the mean compressed size expressed as a percentage of the raw
    # image size, so the 50%-of-raw budget lands exactly on a grid line.
    pct = lambda b: 100.0 * b / raw_bpi

    ax.plot([pct(p["bytes_per_image"]) for p in rich],
            [p["test"]["mean_psnr"] for p in rich],
            color=BLUE, linestyle="-", marker="o", markersize=4,
            linewidth=2.0, label="RichBasis (trained)", zorder=3)
    ax.plot([pct(p["bytes_per_image"]) for p in dct],
            [p["test"]["mean_psnr"] for p in dct],
            color=GREEN, linestyle="-.", marker="s", markersize=3.5,
            linewidth=2.0, label=r"block DCT 8$\times$8", zorder=3)

    # Grid-aligned reference lines: the 35 dB quality (horizontal) and the
    # 40%-of-raw size (vertical). With the % x-axis both fall on grid lines
    # (y=35, x=40), so they read as clean crosshairs rather than free rules.
    ax.axhline(PSNR_CUT, color="0.55", linestyle=(0, (5, 3)), linewidth=1.1,
               zorder=1)
    ax.axvline(V_PCT, color="0.55", linestyle=(0, (5, 3)), linewidth=1.1,
               zorder=1)
    ax.text(pct(100), PSNR_CUT + 0.5, f"{PSNR_CUT:.0f} dB", fontsize=8, color="0.3",
            ha="left", va="bottom")
    ax.annotate(f"{V_PCT:.0f}% of raw", xy=(V_PCT, 20.5), xytext=(-4, 0),
                textcoords="offset points", fontsize=8, rotation=90,
                ha="right", va="center", color="0.25",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.85))

    # Horizontal reading -- compressed size (% of raw) where each curve reaches
    # 35 dB. RichBasis reaches it at a smaller size; the two numbers are marked.
    xr, xd = pct(x_rich), pct(x_dct)
    ax.plot([xr, xd], [PSNR_CUT, PSNR_CUT], marker="o", markersize=4.5,
            linestyle="none", markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.0, zorder=6)
    ax.annotate(f"{xr:.0f}%", xy=(xr, PSNR_CUT), xytext=(-11, -6),
                textcoords="offset points", ha="right", va="top",
                fontsize=8, color=BLUE)
    ax.annotate(f"{xd:.0f}%", xy=(xd, PSNR_CUT), xytext=(4, -8),
                textcoords="offset points", ha="left", va="top",
                fontsize=8, color=GREEN)

    # Vertical reading -- test PSNR at 40% of raw. Both points sit on the x=40
    # grid line and on their curves; same open circles as the horizontal reading,
    # both PSNR values marked (no gap arrow -- the two values carry it).
    ax.plot([V_PCT, V_PCT], [y_rich, y_dct], marker="o", markersize=4.5,
            linestyle="none", markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.0, zorder=6)
    ax.annotate(f"{y_rich:.1f} dB", xy=(V_PCT, y_rich), xytext=(-7, 2),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=8, color=BLUE)
    ax.annotate(f"{y_dct:.1f} dB", xy=(V_PCT, y_dct), xytext=(7, -2),
                textcoords="offset points", ha="left", va="top",
                fontsize=8, color=GREEN)

    ax.set_xlabel("compressed size (% of raw)", fontsize=9)
    ax.set_ylabel("test PSNR (dB)", fontsize=9)
    ax.set_xlim(pct(90), pct(575))
    ax.set_ylim(14.5, 49.5)
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout(pad=0.4)

    figdir = DDIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    pdf = figdir / f"{cfg['out_name']}.pdf"
    save_figure(fig, pdf)
    print(f"wrote {pdf} (+ .svg)")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
