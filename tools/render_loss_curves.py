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
    },
    "div2k_8q": {
        "by_basis": "results/div2k_8q_pca_vs_block_dct/by_basis",
        "out": "results/div2k_8q_pca_vs_block_dct/figures/loss_curves.pdf",
    },
}

# Group + ordering. Names that don't exist on a given dataset are skipped.
UNBLOCKED_PREF = ["qft", "entangled_qft", "tebd", "mera"]
BLOCK_PREF     = ["blocked", "rich", "real_rich",
                  "blocked_8", "rich_8", "real_rich_8"]

# Distinct hues per basis. Block-wrapped use blue tones (consistent with the
# freq_recon_grid header colour); unblocked use gray-warm tones.
COLOR = {
    "qft":           "#444444",
    "entangled_qft": "#777777",
    "tebd":          "#aa6600",
    "mera":          "#cc3300",
    "blocked":       "#08306b",
    "rich":          "#2171b5",
    "real_rich":     "#6baed6",
    "blocked_8":     "#08306b",
    "rich_8":        "#2171b5",
    "real_rich_8":   "#6baed6",
}


def load_loss(by_basis_root: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = by_basis_root / name / "loss_history" / f"{name}_loss.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    step_losses = np.asarray(d["step_losses"], dtype=np.float64)
    val_losses = np.asarray(d["val_losses"], dtype=np.float64)
    return step_losses, val_losses


def plot_panel(ax, by_basis_root: Path, names: list[str], title: str):
    plotted_any = False
    for name in names:
        loaded = load_loss(by_basis_root, name)
        if loaded is None:
            continue
        step_losses, val_losses = loaded
        x = np.arange(1, len(step_losses) + 1)
        ax.plot(x, step_losses, color=COLOR.get(name, "#888888"),
                linewidth=1.0, alpha=0.55, zorder=1)
        # validation: per-epoch — overlay at evenly-spaced step indices
        if len(val_losses) > 0:
            n_epochs = len(val_losses)
            steps_per_epoch = max(1, len(step_losses) // n_epochs)
            x_val = np.arange(1, n_epochs + 1) * steps_per_epoch
            ax.plot(x_val, val_losses, color=COLOR.get(name, "#888888"),
                    linewidth=1.6, marker="o", markersize=2.5,
                    label=f"`{name}`", zorder=2)
        else:
            ax.plot([], [], color=COLOR.get(name, "#888888"),
                    label=f"`{name}`")
        plotted_any = True
    if plotted_any:
        ax.set_yscale("log")
        ax.set_xlabel("training step", fontsize=8)
        ax.set_ylabel("loss (log scale)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85,
                  handlelength=1.6)
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
    plot_panel(axes[0], by_basis_root, UNBLOCKED_PREF,
               "unblocked (full-image transform)")
    plot_panel(axes[1], by_basis_root, BLOCK_PREF,
               "block-wrapped (8×8 inner transform)")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.13,
                        wspace=0.05)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    out_svg = out_pdf.with_suffix(".svg")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"[viz] wrote {out_pdf} + {out_svg}")


if __name__ == "__main__":
    main()
