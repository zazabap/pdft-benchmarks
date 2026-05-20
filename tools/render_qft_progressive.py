#!/usr/bin/env python3
"""Render qft_progressive per-stage training-dynamics curves.

Reads results/qft_progressive/<dataset>/_runs/stage_k<k>/loss_history/
qft_progressive_k<k>_loss.json for k=1..7 and emits training_dynamics.{pdf,svg}
at results/qft_progressive/figures/.

Style follows tools/render_loss_curves.py: one curve per stage on a
single panel with per-stage colour + phase-staggered markers; legend
in upper right; full grid; no top/right spines; no fig.suptitle (per
CLAUDE.md "No figure-level titles").

Stage k=8 is intentionally omitted (the curriculum's headline story
plays out across k=1..7; the k=8 stage where the BlockedBasis wrapper
is dropped is covered in the writeup as a follow-up note, not in the
headline figure).

Usage:
    python tools/render_qft_progressive.py \\
        [--results-base results/qft_progressive/div2k_8q] \\
        [--out-dir results/qft_progressive/figures] \\
        [--mode {normalized,absolute}]   (default: normalized)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Wong-style colourblind-safe palette (CLAUDE.md "Style for multi-curve
# plots"). Seven distinct colours mapped to the seven plotted stages k=1..7.
COLOR_BY_K = {
    1: "#0072B2",  # blue
    2: "#E69F00",  # orange
    3: "#009E73",  # green
    4: "#CC79A7",  # pink
    5: "#D55E00",  # vermilion
    6: "#56B4E9",  # sky
    7: "#000000",  # black
}

# Distinct marker shapes per stage. Same set the existing loss-curves
# renderer uses (o / X / ^ / P / s / h / D).
MARKER_BY_K = {
    1: "o",
    2: "X",
    3: "^",
    4: "P",
    5: "s",
    6: "h",
    7: "D",
}

# Per-stage (period, offset) for marker placement, in epochs. Relatively
# prime periods so marker schedules don't synchronise across coincident
# curves; offsets stagger each stage's first marker. Copied from
# render_loss_curves.py SCHEDULE.
SCHEDULE = [(7, 2), (9, 4), (11, 6), (13, 8), (8, 3), (10, 5), (12, 7)]


def _load_stage_loss_history(results_base: Path, k: int) -> dict:
    cell = results_base / "_runs" / f"stage_k{k}"
    lh_path = cell / "loss_history" / f"qft_progressive_k{k}_loss.json"
    return json.loads(lh_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-base", type=str,
                        default="results/qft_progressive/div2k_8q",
                        help="Parent dir holding _runs/stage_k<k>/.")
    parser.add_argument("--out-dir", type=str,
                        default="results/qft_progressive/figures",
                        help="Where to write training_dynamics.{pdf,svg}.")
    parser.add_argument("--mode", type=str, default="normalized",
                        choices=["normalized", "absolute"],
                        help="normalized: y = val_loss / first_val_loss "
                             "(each curve starts at 1.0). "
                             "absolute: y = val MSE.")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_base = Path(args.results_base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-stage curves for k=1..7.
    curves = []
    for k in range(1, 8):
        lh = _load_stage_loss_history(results_base, k)
        step_losses = np.asarray(lh["step_losses"], dtype=np.float64)
        val_losses = np.asarray(lh["val_losses"], dtype=np.float64)
        if len(val_losses) == 0:
            continue
        L0 = float(val_losses[0])
        if L0 <= 0:
            continue
        n_epochs = len(val_losses)
        steps_per_epoch = max(1, len(step_losses) // n_epochs)
        x_val = np.arange(1, n_epochs + 1) * steps_per_epoch
        val_norm = val_losses / L0
        y_curve = val_losses if args.mode == "absolute" else val_norm
        curves.append({
            "k": k, "x": x_val, "y": y_curve,
            "val_norm": val_norm, "L0": L0,
            "Lf": float(val_losses[-1]),
        })

    if not curves:
        print("[render_qft_progressive] no stage curves found.", file=sys.stderr)
        return 2

    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    for i, c in enumerate(curves):
        k = c["k"]
        color = COLOR_BY_K[k]
        marker = MARKER_BY_K[k]
        x_val, y_curve, val_norm, L0, Lf = c["x"], c["y"], c["val_norm"], c["L0"], c["Lf"]
        n_y = len(y_curve)

        # Phase-staggered marker placement.
        period, offset = SCHEDULE[i % len(SCHEDULE)]
        idx = np.arange(offset, n_y, period)
        if len(idx) > 14:
            idx = idx[:14]
        if len(idx) == 0 or idx[-1] != n_y - 1:
            idx = np.concatenate([idx, [n_y - 1]])
        x_marker = x_val[idx]
        y_marker = y_curve[idx]

        # Legend label: include (L_0 -> L_f) absolute pair so the reader
        # can de-normalise the L/L_0 view by inspection — matches
        # render_loss_curves.py convention.
        if args.mode == "absolute":
            label = f"k={k}"
        else:
            label = f"k={k}  ({L0:.0f}$\\rightarrow${Lf:.0f})"

        ax.plot(x_val, y_curve, color=color, linewidth=1.5,
                linestyle="-", alpha=0.85, zorder=2)
        ax.plot(x_marker, y_marker, color=color, linewidth=0,
                marker=marker, markersize=4.5,
                markerfacecolor=color, markeredgecolor=color,
                markeredgewidth=1.0,
                label=label, zorder=3)

    ax.set_xlabel("training step (within stage)", fontsize=8)
    if args.mode == "absolute":
        ax.set_ylabel("absolute val MSE", fontsize=8)
    else:
        ax.set_ylabel(r"val loss / $L_0$", fontsize=8)
        ax.axhline(1.0, color="#888888", linewidth=0.7, zorder=0)
    ax.tick_params(labelsize=7)

    # Full grid (matches render_loss_curves.py).
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.30, linewidth=0.5,
            color="#999999", zorder=0)
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.3,
            color="#bbbbbb", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=7, loc="upper right", framealpha=0.85,
              handlelength=2.4)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.12)

    pdf_out = out_dir / "training_dynamics.pdf"
    svg_out = out_dir / "training_dynamics.svg"
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_qft_progressive] wrote {pdf_out} and {svg_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
