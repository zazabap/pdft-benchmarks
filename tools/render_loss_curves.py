#!/usr/bin/env python3
"""Render per-dataset training loss curves: one figure with two panels
(unblocked / block-wrapped), per-step loss on log y, one curve per
trained basis. Reads `loss_history/{name}_loss.json` from each
`by_basis/<name>/` cell.

Outputs both PDF (paper) and SVG (typst) following the convention used
by render_freq_recon_grid + render_pca_basis_visualization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DATASET_CONFIG = {
    "quickdraw": {
        "by_basis": "results/quickdraw_pca_vs_block_dct/by_basis",
        "out": "results/quickdraw_pca_vs_block_dct/figures/loss_curves.pdf",
        # QuickDraw curves bottom out around L/L0 ≈ 0.27 (rich), much
        # lower than DIV2K which floors near 0.40. Use a wider y-range
        # so no curve gets clipped.
        "ylim": (0.20, 1.05),
    },
    "div2k_8q": {
        "by_basis": "results/div2k_8q_pca_vs_block_dct/by_basis",
        "out": "results/div2k_8q_pca_vs_block_dct/figures/loss_curves.pdf",
        "ylim": (0.35, 1.05),
    },
}

# Group + ordering. Names that don't exist on a given dataset are skipped.
UNBLOCKED_PREF = ["qft", "entangled_qft", "tebd", "mera"]
BLOCK_PREF     = ["blocked", "rich", "real_rich",
                  "blocked_8", "rich_8", "real_rich_8"]

# Distinct hues per basis (qualitative palette — colourblind-safe Wong /
# Tol "vibrant"-like). Each basis gets a unique colour so curves on the
# same panel are easy to tell apart on small print.
COLOR = {
    "qft":           "#0072B2",  # blue
    "entangled_qft": "#E69F00",  # orange
    "tebd":          "#009E73",  # green
    "mera":          "#CC79A7",  # pink
    "blocked":       "#D55E00",  # vermilion
    "rich":          "#56B4E9",  # sky blue
    "real_rich":     "#000000",  # black
    "blocked_8":     "#D55E00",
    "rich_8":        "#56B4E9",
    "real_rich_8":   "#000000",
}

# Line style per basis — gives curves a second visual axis so even when
# colours look similar after grayscale conversion / poor projection, the
# curves remain individually identifiable.
LINESTYLE = {
    "qft":           "-",
    "entangled_qft": "--",
    "tebd":          "-.",
    "mera":          ":",
    "blocked":       "-",
    "rich":          "--",
    "real_rich":     "-.",
    "blocked_8":     "-",
    "rich_8":        "--",
    "real_rich_8":   "-.",
}


def load_loss(by_basis_root: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = by_basis_root / name / "loss_history" / f"{name}_loss.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    step_losses = np.asarray(d["step_losses"], dtype=np.float64)
    val_losses = np.asarray(d["val_losses"], dtype=np.float64)
    return step_losses, val_losses


def plot_panel(ax, by_basis_root: Path, names: list[str], title: str,
               ylim: tuple[float, float] = (0.4, 1.0)):
    """Plot per-basis training-loss curves on `ax`.

    Y-axis: loss normalised by each basis's *own* initial step-loss, so all
    curves start at 1.0 — what we see is fractional reduction over training.
    A per-basis L0 means cross-basis comparison is on convergence *speed*
    and *floor*, not on raw scale (which depends on dataset and dim).
    """
    plotted_any = False
    for name in names:
        loaded = load_loss(by_basis_root, name)
        if loaded is None:
            continue
        step_losses, val_losses = loaded
        L0 = float(step_losses[0])
        if L0 <= 0:
            continue
        step_norm = step_losses / L0
        val_norm = val_losses / L0 if len(val_losses) > 0 else np.array([])
        color = COLOR.get(name, "#888888")
        ls = LINESTYLE.get(name, "-")
        x = np.arange(1, len(step_norm) + 1)
        # Faint training-step trace.
        ax.plot(x, step_norm, color=color, linewidth=0.7, alpha=0.35,
                zorder=1, linestyle=ls)
        # Per-epoch validation: bold trace with markers, used for the legend.
        if len(val_norm) > 0:
            n_epochs = len(val_norm)
            steps_per_epoch = max(1, len(step_norm) // n_epochs)
            x_val = np.arange(1, n_epochs + 1) * steps_per_epoch
            ax.plot(x_val, val_norm, color=color, linewidth=1.8,
                    linestyle=ls, marker="o", markersize=3.0,
                    label=f"`{name}`", zorder=2)
        else:
            ax.plot([], [], color=color, linestyle=ls, label=f"`{name}`")
        plotted_any = True
    if plotted_any:
        ax.set_xlabel("training step", fontsize=8)
        ax.set_ylabel(r"loss / $L_0$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.axhline(1.0, color="#cccccc", linewidth=0.6, zorder=0)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85,
                  handlelength=2.4)
    else:
        ax.set_title(f"{title} — no curves found", fontsize=9, color="#888888")
        ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(DATASET_CONFIG), required=True)
    ap.add_argument("--out", default=None,
                    help="Output PDF path. None → auto-derived from --dataset.")
    args = ap.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    by_basis_root = Path(cfg["by_basis"])
    out_pdf = Path(args.out if args.out else cfg["out"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    ylim = cfg.get("ylim", (0.4, 1.0))
    plot_panel(axes[0], by_basis_root, UNBLOCKED_PREF,
               "unblocked (full-image transform)", ylim=ylim)
    plot_panel(axes[1], by_basis_root, BLOCK_PREF,
               "block-wrapped (8×8 inner transform)", ylim=ylim)
    # Right panel inherits the y-axis from sharey=True; clear its ylabel so
    # only the left panel labels the shared axis (avoids overlap with the
    # right panel's left edge under tight inter-panel spacing).
    axes[1].set_ylabel("")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.13,
                        wspace=0.05)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    out_svg = out_pdf.with_suffix(".svg")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"[viz] wrote {out_pdf} + {out_svg}")


if __name__ == "__main__":
    main()
