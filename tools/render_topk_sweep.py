#!/usr/bin/env python3
"""Render the QFT training top-k sweep figure.

Reads results/training/3_training_topk/<dataset>/manifest.json (written by
experiments/qft_topk_sweep.py) and emits topk_<dataset>.{pdf,svg} at
results/training/3_training_topk/figures/.

Two panels (CLAUDE.md style: Wong palette, linear y, no figure title):
  left  — PSNR vs training-k, one line per eval keep-ratio. The diagonal
          point (training-k == eval-rho) is ringed; if "train at the rate
          you deploy at" helps, each line peaks at its own ringed point.
  right — the full train-k x eval-rho PSNR matrix as a heatmap, every cell
          annotated, the best training-k per eval-rho (per column) boxed.

Usage:
    python tools/render_topk_sweep.py --dataset div2k_8q
    python tools/render_topk_sweep.py --dataset quickdraw_5q
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Wong-style colourblind-safe palette (CLAUDE.md "Style for multi-curve plots").
WONG = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]
MARKERS = ["o", "X", "^", "s", "P", "h", "D"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="div2k_8q",
                        choices=["div2k_8q", "quickdraw_5q"])
    parser.add_argument("--manifest", default=None,
                        help="Path to manifest.json. Default "
                             "results/training/3_training_topk/<dataset>/manifest.json.")
    parser.add_argument("--out-dir", default="results/training/3_training_topk/figures")
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else \
        Path(f"results/training/3_training_topk/{args.dataset}/manifest.json")
    if not manifest_path.is_file():
        print(f"[render_topk] manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["rows"]
    if not rows:
        print("[render_topk] manifest has no rows", file=sys.stderr)
        return 2

    # train-k ratios (x-axis) and eval-rho keys (one line / heatmap col each).
    train_ratios = [r["train_ratio"] for r in rows]
    eval_keys = sorted(rows[0]["psnr"], key=float)          # e.g. ["0.05","0.1","0.15","0.2"]
    eval_vals = [float(k) for k in eval_keys]
    # matrix[i, j] = PSNR for train_ratios[i], eval_keys[j]
    matrix = np.array([[row["psnr"][k] for k in eval_keys] for row in rows], dtype=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

    # --- left: PSNR vs training-k, one line per eval-rho ---
    for j, ek in enumerate(eval_keys):
        col = matrix[:, j]
        axL.plot(train_ratios, col, color=WONG[j], marker=MARKERS[j],
                 markersize=6, linewidth=1.8, label=f"eval ρ={eval_vals[j]:g}")
        # Ring the diagonal point (training-k == this eval-rho), if present.
        if eval_vals[j] in train_ratios:
            di = train_ratios.index(eval_vals[j])
            axL.plot(train_ratios[di], col[di], marker="o", markersize=13,
                     markerfacecolor="none", markeredgecolor=WONG[j],
                     markeredgewidth=1.8, zorder=5)
    axL.set_xlabel("training top-k (fraction of coefficients)")
    axL.set_ylabel("test PSNR (dB)")
    axL.set_xticks(train_ratios)
    axL.grid(True, alpha=0.3)
    axL.spines["top"].set_visible(False)
    axL.spines["right"].set_visible(False)
    axL.legend(loc="best", frameon=False, fontsize=9)

    # --- right: full matrix heatmap, best train-k per eval-rho boxed ---
    im = axR.imshow(matrix, aspect="auto", cmap="viridis", origin="lower")
    axR.set_xticks(range(len(eval_keys)))
    axR.set_xticklabels([f"{v:g}" for v in eval_vals])
    axR.set_yticks(range(len(train_ratios)))
    axR.set_yticklabels([f"{r:g}" for r in train_ratios])
    axR.set_xlabel("eval keep-ratio ρ")
    axR.set_ylabel("training top-k")
    best_per_col = matrix.argmax(axis=0)                    # best train-k row per eval col
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            txt = f"{matrix[i, j]:.2f}"
            weight = "bold" if best_per_col[j] == i else "normal"
            # Pick a legible text colour against the viridis cell.
            norm = (matrix[i, j] - matrix.min()) / (np.ptp(matrix) + 1e-12)
            axR.text(j, i, txt, ha="center", va="center", fontsize=8,
                     fontweight=weight, color="white" if norm < 0.5 else "black")
            if best_per_col[j] == i:
                axR.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor="red", linewidth=2))
    fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04, label="PSNR (dB)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = out_dir / f"topk_{args.dataset}.pdf"
    svg_out = out_dir / f"topk_{args.dataset}.svg"
    fig.tight_layout()
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(svg_out, bbox_inches="tight")
    print(f"[render_topk] wrote {pdf_out}")
    print(f"[render_topk] wrote {svg_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
