#!/usr/bin/env python3
"""Render the qft_unfreeze staircase comparison (loss + grad-norm vs step,
one curve per ordering). PDF + SVG, no figure title, Wong palette.

Usage:
    python tools/render_qft_unfreeze.py --dataset quickdraw_5q
    python tools/render_qft_unfreeze.py --in results/qft_unfreeze/quickdraw_5q
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Wong palette + line style, one per ordering.
STYLE = {
    "bg": ("#0072B2", "-",  "block-growth"),
    "lr": ("#E69F00", "--", "left→right"),
    "rl": ("#009E73", "-.", "right→left"),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--in", dest="indir", default=None)
    args = p.parse_args()

    indir = Path(args.indir) if args.indir else Path(f"results/qft_unfreeze/{args.dataset}")
    if not indir.exists():
        print(f"[render] no such dir: {indir}", file=sys.stderr)
        return 2

    fig, (ax_loss, ax_grad) = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    plotted = 0
    for name, (color, ls, label) in STYLE.items():
        tj = indir / name / "trace.json"
        if not tj.exists():
            continue
        steps = json.loads(tj.read_text())["steps"]
        xs = [r["step"] for r in steps]
        loss = [r["loss"] for r in steps]
        grad = [r["grad_norm"] for r in steps]
        l0 = loss[0] if loss and loss[0] > 0 else 1.0
        ax_loss.plot(xs, [v / l0 for v in loss], color=color, ls=ls, lw=1.4, label=label)
        ax_grad.plot(xs, grad, color=color, ls=ls, lw=1.4, label=label)
        plotted += 1

    if plotted == 0:
        print(f"[render] no trace.json under {indir}", file=sys.stderr)
        return 2

    ax_loss.set_ylabel(r"loss  $L / L_0$")
    ax_loss.legend(frameon=False, fontsize=8)
    ax_grad.set_yscale("log")
    ax_grad.set_ylabel(r"grad norm  $\|g\|$")
    ax_grad.set_xlabel("cumulative training step")
    for ax in (ax_loss, ax_grad):
        ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()

    figdir = indir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = figdir / f"staircase.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[render] wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
