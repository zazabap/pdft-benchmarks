#!/usr/bin/env python3
"""Render the qft_blocked_replication reproduction-comparison figure.

Overlays the validation-loss training trajectories of four cells trained
at the headline 1008-step budget on DIV2K-8q:

  qft               — analytic init, all 72 gates trainable.
  qft_identity      — identity init, all 72 gates trainable.
  blocked_8         — BlockedBasis(QFTBasis(3,3), 5, 5) analytic init,
                      12 inner gates trainable (block-tiling implicit).
  qft_frozen_outer  — QFTBasis(8, 8) identity init, 12 inner gates
                      trainable, 60 outer gates frozen at identity via
                      pdft.train_basis_batched(..., frozen_indices=...).

The reproduction story: qft_frozen_outer's trajectory + final value
match blocked_8's exactly (+0.000 dB delta at every test rho), even
though their inits differ (identity vs analytic). This visualises the
operator-equivalence claim.

Style follows tools/render_loss_curves.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Wong-style palette, distinct per cell.
COLOR = {
    "qft":               "#08306b",  # navy
    "qft_identity":      "#56B4E9",  # sky
    "blocked_8":         "#D55E00",  # vermilion
    "qft_frozen_outer":  "#000000",  # black (the new reproduction)
}
MARKER = {
    "qft":               "o",
    "qft_identity":      "X",
    "blocked_8":         "s",
    "qft_frozen_outer":  "D",
}
LABEL = {
    "qft":               "qft (analytic init)",
    "qft_identity":      "qft_identity (identity init)",
    "blocked_8":         "blocked_8 (BlockedBasis analytic)",
    "qft_frozen_outer":  "qft_frozen_outer (identity + frozen_indices)",
}

# Per-cell loss_history path.
LOSS_HISTORY = {
    "qft": "results/div2k_8q_pca_vs_block_dct/by_basis/qft/loss_history/qft_loss.json",
    "qft_identity": "results/qft_identity_init/div2k_8q/_runs/run1/loss_history/qft_identity_loss.json",
    "blocked_8": "results/div2k_8q_pca_vs_block_dct/by_basis/blocked_8/loss_history/blocked_8_loss.json",
    "qft_frozen_outer": "results/qft_progressive/blocked_replication/div2k_8q/_runs/freeze_outer/loss_history/qft_freeze_outer_loss.json",
}

# Phase-staggered (period, offset) per cell — relatively-prime so adjacent
# markers don't collide on coincident curves.
SCHEDULE = [(7, 2), (9, 4), (11, 6), (13, 8)]


def _load_curve(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Returns (x_steps, y_val_mse, L_0, L_f)."""
    d = json.loads(path.read_text())
    step_losses = np.asarray(d["step_losses"], dtype=np.float64)
    val_losses = np.asarray(d["val_losses"], dtype=np.float64)
    if len(val_losses) == 0:
        raise ValueError(f"no val_losses in {path}")
    n_epochs = len(val_losses)
    steps_per_epoch = max(1, len(step_losses) // n_epochs)
    x = np.arange(1, n_epochs + 1) * steps_per_epoch
    return x, val_losses, float(val_losses[0]), float(val_losses[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=str,
                        default="results/qft_progressive/blocked_replication/div2k_8q/figures",
                        help="Where to write reproduction_comparison.{pdf,svg}.")
    parser.add_argument("--ymin", type=float, default=120.0)
    parser.add_argument("--ymax", type=float, default=320.0)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Order of plotting (and legend): qft (highest L_f), qft_identity, blocked_8,
    # qft_frozen_outer (the reproduction, drawn last so it overlays blocked_8).
    cells = ["qft", "qft_identity", "blocked_8", "qft_frozen_outer"]

    curves = []
    for c in cells:
        path = Path(LOSS_HISTORY[c])
        if not path.exists():
            print(f"[render_qft_blocked_replication] missing: {path}", file=sys.stderr)
            return 2
        x, y, L0, Lf = _load_curve(path)
        curves.append({"name": c, "x": x, "y": y, "L0": L0, "Lf": Lf})

    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    for i, c in enumerate(curves):
        name = c["name"]
        color = COLOR[name]
        marker = MARKER[name]
        x_val, y_curve = c["x"], c["y"]
        L0, Lf = c["L0"], c["Lf"]
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

        label = f"{LABEL[name]}  ({L0:.0f}$\\rightarrow${Lf:.1f})"
        ax.plot(x_val, y_curve, color=color, linewidth=1.5,
                linestyle="-", alpha=0.85, zorder=2)
        ax.plot(x_marker, y_marker, color=color, linewidth=0,
                marker=marker, markersize=5.0,
                markerfacecolor=color, markeredgecolor=color,
                markeredgewidth=1.0,
                label=label, zorder=3)

    ax.set_xlabel("training step", fontsize=8)
    ax.set_ylabel("absolute val MSE", fontsize=8)
    ax.set_ylim(args.ymin, args.ymax)
    ax.tick_params(labelsize=7)

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

    pdf_out = out_dir / "reproduction_comparison.pdf"
    svg_out = out_dir / "reproduction_comparison.svg"
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_qft_blocked_replication] wrote {pdf_out} and {svg_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
