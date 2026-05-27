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


# Dataset columns for the combined training-dynamics grid (content labels, not
# a figure title — see CLAUDE.md "No figure-level titles").
COMBINED_DATASETS = [
    ("quickdraw_5q", "QuickDraw  (m=n=5)"),
    ("div2k_8q", "DIV2K  (m=n=8)"),
    ("tuberlin_8q", "TU-Berlin  (m=n=8)"),
]


COMBINED_INITS = [("identity", "identity init"), ("random", "random init")]


def _final_psnr(trace: dict, ratio: str = "0.2"):
    """Final-stage test PSNR at `ratio` from a trace dict, or None if absent
    (the final stage is always evaluated, so it is present once training ends)."""
    for st in reversed(trace.get("stages", [])):
        p = (st.get("extra") or {}).get("psnr") or {}
        if ratio in p:
            return p[ratio]
    return None


def _render_combined(base: Path) -> int:
    """Training-dynamics grid: rows = init (identity, random), cols = dataset,
    one loss L/L0 curve per unfreeze ordering. The qft_unfreeze analogue of the
    other experiments' <exp>/figures/training_dynamics figure. Per-(dataset,init)
    grad-norm staircases live in <dataset>/<init>/figures/staircase.* ."""
    cols = [(tag, lab) for tag, lab in COMBINED_DATASETS
            if any((base / tag / init).exists() for init, _ in COMBINED_INITS)]
    rows = [(init, lab) for init, lab in COMBINED_INITS
            if any((base / tag / init).exists() for tag, _ in COMBINED_DATASETS)]
    if not cols or not rows:
        print(f"[render] no <dataset>/<init> subdirs under {base}", file=sys.stderr)
        return 2

    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(3.4 * len(cols), 2.5 * len(rows)),
                             squeeze=False, sharex="col")
    any_curve = False
    for r, (init, init_lab) in enumerate(rows):
        for c, (tag, ds_lab) in enumerate(cols):
            ax = axes[r][c]
            finals = []  # (color, leg, mse_end, psnr) for the corner annotation
            for name, (color, ls, leg) in STYLE.items():
                tj = base / tag / init / name / "trace.json"
                if not tj.exists():
                    continue
                trace = json.loads(tj.read_text())
                steps = trace["steps"]
                if not steps:
                    continue
                xs = [s["step"] for s in steps]
                loss = [s["loss"] for s in steps]  # absolute top-k MSE (not L/L0)
                ax.plot(xs, loss, color=color, ls=ls, lw=1.3, label=leg)
                ax.plot([xs[-1]], [loss[-1]], marker="o", ms=3.5, color=color)  # endpoint
                finals.append((color, leg, loss[-1], _final_psnr(trace)))
                any_curve = True
            # Mark the final absolute MSE and final PSNR@0.20 per ordering.
            for i, (color, leg, mse, psnr) in enumerate(finals):
                txt = f"MSE {mse:.4g}" + (f" · {psnr:.1f} dB" if psnr is not None else "")
                ax.text(0.97, 0.96 - 0.085 * i, txt, transform=ax.transAxes,
                        ha="right", va="top", fontsize=6.5, color=color)
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(ds_lab, fontsize=9)
            if r == len(rows) - 1:
                ax.set_xlabel("cumulative training step", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{init_lab}\ntop-$k$ MSE loss", fontsize=8.5)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=7.5)

    if not any_curve:
        print(f"[render] no trace.json found under {base}/<dataset>/<init>/", file=sys.stderr)
        return 2

    fig.tight_layout()
    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        out = figdir / f"training_dynamics.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[render] wrote {out}")
    plt.close(fig)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--in", dest="indir", default=None)
    p.add_argument("--combined", action="store_true",
                   help="Render a 2x3 (loss, grad-norm) x (dataset) training-"
                        "dynamics grid into <base>/figures/training_dynamics.{pdf,svg}.")
    p.add_argument("--base", default="results/qft_unfreeze",
                   help="Experiment base dir (used with --combined).")
    args = p.parse_args()

    if args.combined:
        return _render_combined(Path(args.base))

    indir = Path(args.indir) if args.indir else Path(f"results/qft_unfreeze/{args.dataset}")
    if not indir.exists():
        print(f"[render] no such dir: {indir}", file=sys.stderr)
        return 2

    fig, (ax_loss, ax_grad) = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    plotted = 0
    finals = []  # (color, label, L/L0_end, abs_loss_end, grad_end, psnr)
    for name, (color, ls, label) in STYLE.items():
        tj = indir / name / "trace.json"
        if not tj.exists():
            continue
        trace = json.loads(tj.read_text())
        steps = trace["steps"]
        xs = [r["step"] for r in steps]
        loss = [r["loss"] for r in steps]  # absolute top-k MSE
        grad = [r["grad_norm"] for r in steps]
        ax_loss.plot(xs, loss, color=color, ls=ls, lw=1.4, label=label)
        ax_grad.plot(xs, grad, color=color, ls=ls, lw=1.4, label=label)
        # mark the endpoints (final MSE / final grad norm)
        ax_loss.plot([xs[-1]], [loss[-1]], marker="o", ms=4, color=color)
        ax_grad.plot([xs[-1]], [grad[-1]], marker="o", ms=4, color=color)
        finals.append((color, label, loss[-1], grad[-1], _final_psnr(trace)))
        plotted += 1

    if plotted == 0:
        print(f"[render] no trace.json under {indir}", file=sys.stderr)
        return 2

    # Final absolute MSE + grad norm + PSNR@0.20 per ordering, marked at finish.
    for i, (color, label, mse, gend, psnr) in enumerate(finals):
        txt = f"{label}: MSE={mse:.4g}" + \
              (f"  PSNR@.20={psnr:.2f} dB" if psnr is not None else "")
        ax_loss.text(0.985, 0.95 - 0.075 * i, txt, transform=ax_loss.transAxes,
                     ha="right", va="top", fontsize=7, color=color)

    ax_loss.set_ylabel("top-$k$ MSE loss")
    ax_loss.legend(frameon=False, fontsize=8, loc="upper right")
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
