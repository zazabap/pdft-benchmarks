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

# Within-group sequential palette: unblocked = blue family, block-wrapped
# = red/orange family. Cohesive look for the paper. When two curves end
# at near-identical values (qft ≡ entangled_qft, tebd ≡ mera, rich ≈
# real_rich), the within-pair distinguisher is the marker, not the
# colour — see MARKER below.
COLOR = {
    # unblocked group — blue family
    "qft":           "#08306b",  # navy
    "entangled_qft": "#2171b5",  # mid-blue
    "tebd":          "#0e6655",  # dark teal
    "mera":          "#26c6da",  # cyan-teal
    # block-wrapped group — red/orange family
    "blocked":       "#a50f15",  # deep red
    "rich":          "#cb6020",  # burnt orange
    "real_rich":     "#000000",  # black (anchor)
    "blocked_8":     "#a50f15",
    "rich_8":        "#cb6020",
    "real_rich_8":   "#000000",
}

# Per-basis marker shape. Each basis has a unique geometric shape; with
# phase-staggered marker positions across bases, two coincident curves
# never put a marker at the same x and the shapes interleave cleanly
# along the trajectory.
MARKER = {
    "qft":           "o",  # circle
    "entangled_qft": "X",  # filled X
    "tebd":          "^",  # up-triangle
    "mera":          "P",  # filled plus
    "blocked":       "s",  # square
    "rich":          "h",  # hexagon
    "real_rich":     "D",  # diamond
    "blocked_8":     "s",
    "rich_8":        "h",
    "real_rich_8":   "D",
}
# Per-marker size override (matplotlib defaults render some shapes a
# touch lighter than circles at the same nominal size; bump those).
MARKER_SIZE = {
    "X": 5.0,
    "P": 5.0,
}
HOLLOW = {
    "qft":           False,
    "entangled_qft": False,
    "tebd":          False,
    "mera":          False,
    "blocked":       False,
    "rich":          False,
    "real_rich":     False,
    "blocked_8":     False,
    "rich_8":        False,
    "real_rich_8":   False,
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
               ylim: tuple[float, float] = (0.4, 1.0),
               mode: str = "normalized",
               show_legend: bool = True,
               show_xlabel: bool = True,
               show_title: bool = True):
    """Plot per-basis training-loss curves on `ax` following standard
    technical-figure conventions:

    - Y axis is either L/L_0 (mode="normalized") or absolute MSE
      (mode="absolute"). Normalized mode is good for cross-basis
      convergence shape; absolute mode preserves PSNR-equivalent
      ordering (lower curve == higher PSNR).
    - One line per basis (full validation trajectory), one marker style
      per basis (subset of points; phase-staggered across bases so two
      coincident curves never put markers at the same x).
    - Legend ordered by absolute final MSE ascending — best (lowest) at
      the top so the reader maps the legend to the curve order at the
      right edge in absolute mode (and the legend stays consistent
      across both modes).
    - Subtle line alpha (0.85) so overlapping coincident curves blend
      into a slightly darker stroke instead of fighting for ink.
    """
    # Collect per-basis curves first so we can sort + stagger.
    curves = []
    for name in names:
        loaded = load_loss(by_basis_root, name)
        if loaded is None:
            continue
        step_losses, val_losses = loaded
        L0 = float(step_losses[0])
        if L0 <= 0:
            continue
        if len(val_losses) == 0:
            continue
        val_norm = val_losses / L0
        n_epochs = len(val_norm)
        steps_per_epoch = max(1, len(step_losses) // n_epochs)
        x_val = np.arange(1, n_epochs + 1) * steps_per_epoch
        # Pick which curve to plot based on mode; keep both around so
        # the legend can always reference the absolute (L_0 -> L_f) pair.
        y_curve = val_losses if mode == "absolute" else val_norm
        curves.append((name, x_val, y_curve, val_norm, L0))

    if not curves:
        ax.set_title(f"{title} — no curves found", fontsize=9, color="#888888")
        ax.set_xticks([]); ax.set_yticks([])
        return False

    # Sort by final ABSOLUTE value (val_norm[-1] * L_0) ascending: lowest
    # absolute-loss curves listed first in the legend. The figure may
    # show L/L_0 OR absolute MSE depending on mode; sorting by absolute
    # value keeps the legend faithful to which basis actually achieves
    # the lowest MSE (which is what the headline PSNR table reports).
    curves.sort(key=lambda c: float(c[3][-1]) * float(c[4]))
    # Per-basis (period, offset) for marker placement, in epochs.
    # Periods are relatively prime so the marker schedules don't
    # synchronise even when two curves coincide; offsets push each
    # basis's first marker to a different starting epoch. Result:
    # gaps between adjacent markers are slightly different per
    # basis, killing the alignment artefact that made the
    # uniform-spacing version look overlap-y on coincident curves.
    SCHEDULE = [(7, 2), (9, 4), (11, 6), (13, 8), (8, 3), (10, 5), (12, 7)]

    for i, (name, x_val, y_curve, val_norm, L0) in enumerate(curves):
        color = COLOR.get(name, "#888888")
        marker = MARKER.get(name, "o")
        hollow = HOLLOW.get(name, False)
        face = "white" if hollow else color
        n_epochs = len(val_norm)

        # Per-basis non-uniform marker schedule: epochs offset+0,
        # offset+period, offset+2*period, ... Each basis uses a
        # different (period, offset) so adjacent markers from different
        # bases never share an x — even on coincident curves.
        n_y = len(y_curve)
        period, offset = SCHEDULE[i % len(SCHEDULE)]
        idx = np.arange(offset, n_y, period)
        # Cap at ~14 markers so dense panels don't get crowded.
        if len(idx) > 14:
            idx = idx[:14]
        # Always include the final epoch as a marker (anchors the
        # convergence floor at the right edge of the panel).
        if len(idx) == 0 or idx[-1] != n_y - 1:
            idx = np.concatenate([idx, [n_y - 1]])
        x_marker = x_val[idx]
        y_marker = y_curve[idx]

        # Connecting line — slight transparency so overlapping equivalent
        # curves blend, plus they don't visually compete for ink.
        ax.plot(x_val, y_curve, color=color, linewidth=1.5,
                linestyle="-", alpha=0.85, zorder=2)
        # Markers at staggered slots.
        msize = MARKER_SIZE.get(marker, 4.5)
        # Legend label: in absolute mode the y-axis already shows MSE,
        # so the basis name alone suffices. In normalized mode we
        # include the (L_0 -> L_f) absolute pair so the reader can
        # de-normalise the L/L_0 view by inspection (a curve sitting
        # lower in L/L_0 space can still have higher absolute MSE).
        L_final_abs = float(val_norm[-1]) * float(L0)
        if mode == "absolute":
            label = f"`{name}`"
        else:
            label = f"`{name}`  ({L0:.0f}$\\rightarrow${L_final_abs:.0f})"
        ax.plot(x_marker, y_marker, color=color, linewidth=0,
                marker=marker, markersize=msize,
                markerfacecolor=face, markeredgecolor=color,
                markeredgewidth=1.0,
                label=label,
                zorder=3)
    plotted_any = True
    if plotted_any:
        if show_xlabel:
            ax.set_xlabel("training step", fontsize=8)
        if mode == "absolute":
            ax.set_ylabel("absolute val MSE", fontsize=8)
        else:
            ax.set_ylabel(r"loss / $L_0$", fontsize=8)
        ax.tick_params(labelsize=7)
        if show_title:
            ax.set_title(title, fontsize=9)
        # Full grid on both axes — major gridlines for primary readability
        # and a fainter minor grid so individual checkpoint positions are
        # easy to read off without the lines competing with the markers.
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.30, linewidth=0.5,
                color="#999999", zorder=0)
        ax.grid(True, which="minor", alpha=0.15, linewidth=0.3,
                color="#bbbbbb", zorder=0)
        if mode != "absolute":
            ax.axhline(1.0, color="#888888", linewidth=0.7, zorder=0)
            ax.set_ylim(*ylim)
        # Trim the panel borders for a less boxy look.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if show_legend:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.85,
                      handlelength=2.4)
    else:
        ax.set_title(f"{title} — no curves found", fontsize=9, color="#888888")
        ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(DATASET_CONFIG), default=None)
    ap.add_argument("--by-basis", default=None,
                    help="override by_basis root (use with --bases / --out).")
    ap.add_argument("--bases", default=None,
                    help="comma list of basis names to plot (default: all).")
    ap.add_argument("--out", default=None,
                    help="Output PDF path. None → auto-derived from --dataset.")
    args = ap.parse_args()

    if args.by_basis:
        by_basis_root = Path(args.by_basis)
        names = (args.bases.split(",") if args.bases
                 else UNBLOCKED_PREF + BLOCK_PREF)
        out_pdf = Path(args.out) if args.out else \
            by_basis_root.parent / "figures" / "loss_curves.pdf"
    else:
        if not args.dataset:
            ap.error("provide --dataset, or --by-basis (+ --bases/--out)")
        cfg = DATASET_CONFIG[args.dataset]
        by_basis_root = Path(cfg["by_basis"])
        names = UNBLOCKED_PREF + BLOCK_PREF
        out_pdf = Path(args.out if args.out else cfg["out"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Single panel, absolute val MSE, all bases together — the canonical
    # ML-paper loss curve format. Same y-axis for every basis means the
    # right-edge curve ordering equals the headline PSNR ordering
    # (max PSNR == min MSE) by inspection. The previous L/L_0
    # normalization caused that ordering to disagree with PSNR (because
    # per-basis L_0 differs by ~2x); the absolute view fixes that.
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.6))
    plot_panel(ax, by_basis_root, names,
               title="",
               mode="absolute", show_legend=True, show_title=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.10)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    out_svg = out_pdf.with_suffix(".svg")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"[viz] wrote {out_pdf} + {out_svg}")


if __name__ == "__main__":
    main()
