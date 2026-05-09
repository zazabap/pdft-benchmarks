#!/usr/bin/env python3
"""Render multi-ρ block-size sweep figure (B1 framing).

Three sub-panels in a row per dataset:
  1. block_dct   — HEVC-style classical (unfitted reference)
  2. block_bd_pca — dataset-fitted classical (paper headline classical)
  3. real_rich   — headline trained basis

Each panel shows 4 PSNR-vs-b curves (one per ρ ∈ {0.05, 0.10, 0.15,
0.20}) on a log-2 x-axis. Y-axis is shared across the three panels so
absolute PSNR can be compared across families.

Usage:
    python tools/render_block_size_sweep.py --dataset {quickdraw,div2k_8q}

Outputs PDF + SVG into results/block_size_sweep/<dataset>/figures/:
    sweep_quickdraw.{pdf,svg}
    sweep_div2k_8q.{pdf,svg}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "quickdraw": {
        "by_basis": "results/block_size_sweep/quickdraw/by_basis",
        "out_stem": "results/block_size_sweep/quickdraw/figures/sweep_quickdraw",
        # Classical b grid (block_dct, block_bd_pca)
        "classical_b": [2, 4, 8, 16],
        # Trained real_rich b grid
        "trained_b": [4, 8, 16],
        # Display name used in per-panel sub-label
        "dataset_label": "QuickDraw",
    },
    "div2k_8q": {
        "by_basis": "results/block_size_sweep/div2k_8q/by_basis",
        "out_stem": "results/block_size_sweep/div2k_8q/figures/sweep_div2k_8q",
        # Stop at 128 (drop 256 = global)
        "classical_b": [4, 8, 16, 32, 64, 128],
        # Trained real_rich b grid
        "trained_b": [4, 8, 16, 32],
        "dataset_label": "DIV2K-8q",
    },
}

# Single source of truth for retention values; the JSON metric files use
# string keys with trailing zeros stripped (so 0.10 → "0.1", 0.20 → "0.2").
RHO_VALS = (0.05, 0.10, 0.15, 0.20)


def _rho_key(r: float) -> str:
    """Match the JSON serialisation: trim trailing zeros from the float string."""
    return f"{r:g}"


RHO_KEYS = tuple(_rho_key(r) for r in RHO_VALS)

# Sequential blue palette: light → dark for ρ=0.05 → ρ=0.20
RHO_COLORS = {
    "0.05": "#9ecae1",
    "0.1":  "#6baed6",
    "0.15": "#3182bd",
    "0.2":  "#08519c",
}
RHO_LABELS = {
    "0.05": r"$\rho = 0.05$",
    "0.1":  r"$\rho = 0.10$",
    "0.15": r"$\rho = 0.15$",
    "0.2":  r"$\rho = 0.20$",
}

# Three panels per figure: (basis_family, sub_label, uses_trained_data)
# Plain text labels — matplotlib's default text renderer does not interpret
# LaTeX \_ as subscript; use raw underscores instead.
PANELS = [
    ("block_dct",    "block_dct (classical)",    False),
    ("block_bd_pca", "block_bd_pca (fitted)",    False),
    ("real_rich",    "real_rich (trained)",       True),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_baselines(by_basis: Path) -> dict:
    """Load _baselines.json, return the parsed dict."""
    p = by_basis / "_baselines.json"
    return json.loads(p.read_text())


def load_trained_metrics(by_basis: Path, family: str, b: int) -> dict | None:
    """Load metrics.json for a trained basis cell <family>_<b>.

    Returns the inner per-ρ metrics dict, or None if cell is missing.
    """
    cell_name = f"{family}_{b}"
    p = by_basis / cell_name / "metrics.json"
    if not p.exists():
        return None
    outer = json.loads(p.read_text())
    # Outer key is the basis name itself
    if cell_name not in outer:
        return None
    return outer[cell_name]["metrics"]


def get_psnr_curve(
    baselines: dict,
    by_basis: Path,
    family: str,
    b_values: list[int],
    is_trained: bool,
) -> dict[int, dict[str, float]]:
    """Return {b: {rho_key: mean_psnr}} for every b in b_values.

    Missing cells are silently dropped.
    """
    result: dict[int, dict[str, float]] = {}
    for b in b_values:
        key = f"{family}_{b}"
        if is_trained:
            metrics = load_trained_metrics(by_basis, family, b)
        else:
            if key not in baselines:
                continue
            metrics = baselines[key]["metrics"]
        if metrics is None:
            continue
        result[b] = {rk: metrics[rk]["mean_psnr"] for rk in RHO_KEYS if rk in metrics}
    return result


# ---------------------------------------------------------------------------
# Panel renderer
# ---------------------------------------------------------------------------

def render_panel(
    ax: plt.Axes,
    psnr_data: dict[int, dict[str, float]],
    panel_label: str,
    dataset_label: str,
    b_ref: int = 8,
    ymax_clip: float | None = None,
) -> None:
    """Render one PSNR-vs-b panel onto `ax`.

    psnr_data: {b: {rho_key: psnr}}
    ymax_clip: the shared y-axis ceiling (passed in so we can mark clipped
               points with an upward-arrow indicator for honesty).
    """
    b_values = sorted(psnr_data.keys())
    if not b_values:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        return

    for rk in RHO_KEYS:
        ys = []
        xs = []
        for b in b_values:
            if rk in psnr_data[b]:
                xs.append(b)
                ys.append(psnr_data[b][rk])

        if not xs:
            continue

        color = RHO_COLORS[rk]

        # Build display ys — clamp any value above ymax_clip so the curve
        # connects through a visible clamped point.  We record which points
        # were clipped to draw an upward-arrow indicator afterward.
        clipped_mask = [False] * len(ys)
        ys_display = list(ys)
        if ymax_clip is not None:
            for i, y in enumerate(ys):
                if y > ymax_clip:
                    ys_display[i] = ymax_clip
                    clipped_mask[i] = True

        ax.plot(
            xs, ys_display,
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=5,
            markerfacecolor=color,
            markeredgewidth=0,
            label=RHO_LABELS[rk],
            zorder=3,
        )

        # Peak marker — filled diamond overlaid at argmax of *actual* ys
        peak_idx = int(np.argmax(ys))
        x_peak = xs[peak_idx]
        y_peak_display = ys_display[peak_idx]
        ax.plot(
            x_peak, y_peak_display,
            marker="D",
            markersize=9,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=5,
            linestyle="none",
        )

        # Clip indicators — for any clamped point, draw a small upward
        # triangle at the clamped y-position and annotate with the true
        # value.  Using a marker (not an annotation arrow) avoids z-order
        # conflicts with the legend box.
        for i, clipped in enumerate(clipped_mask):
            if clipped:
                # Upward triangle marker at the clamped position
                ax.plot(
                    xs[i], ys_display[i],
                    marker="^",
                    markersize=9,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=7,
                    linestyle="none",
                )
                # Text annotation showing the true value, placed to the
                # right of the triangle marker so it doesn't overlap the
                # legend (which sits in the upper-left corner).
                ax.text(
                    xs[i] * 1.15, ys_display[i],
                    f"{ys[i]:.1f}",
                    ha="left", va="center",
                    fontsize=6.5,
                    color=color,
                    zorder=7,
                )

    # Vertical dashed reference at b=8 — no legend entry (position is
    # self-evident; adding it would clutter the ρ legend)
    if min(b_values) <= b_ref <= max(b_values):
        ax.axvline(b_ref, color="#888888", linewidth=0.9, linestyle="--",
                   zorder=2)

    # x-axis: log-2 scale; ticks only at actual b values
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_major_locator(mticker.FixedLocator(b_values))
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    ax.set_xlabel("block size $b$", fontsize=10)
    ax.set_ylabel("PSNR (dB)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)

    # Per-panel sub-label (content, not title)
    # Use text in axes coordinates so it sits inside the panel
    ax.text(
        0.97, 0.03,
        f"{panel_label}\n({dataset_label})",
        ha="right", va="bottom",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#333333",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
    )

    ax.grid(axis="y", linewidth=0.4, color="#cccccc", zorder=1)
    ax.grid(axis="x", linewidth=0.4, color="#cccccc", zorder=1)


# ---------------------------------------------------------------------------
# Figure orchestrator
# ---------------------------------------------------------------------------

def build_figure(dataset: str, repo_root: Path) -> None:
    cfg = DATASET_CONFIG[dataset]
    by_basis = repo_root / cfg["by_basis"]
    out_stem = repo_root / cfg["out_stem"]
    classical_b = cfg["classical_b"]
    trained_b = cfg["trained_b"]
    dataset_label = cfg["dataset_label"]

    baselines = load_baselines(by_basis)

    # Collect data for each panel
    panel_data = []
    for family, label, is_trained in PANELS:
        b_grid = trained_b if is_trained else classical_b
        psnr_data = get_psnr_curve(baselines, by_basis, family, b_grid, is_trained)
        panel_data.append((family, label, psnr_data))

    # Compute shared y-axis limits from all panels.
    # Use the 95th-percentile ceiling so one outlier point (e.g. QuickDraw
    # block_dct b=2 ρ=0.20 at 44.6 dB — near-lossless because b²=4 < m=5
    # so all block coefficients are selected) does not compress the rest of
    # the figure.  The line still enters the axis from below and the steep
    # drop from that outlier to b=4 remains visible; only the y-axis scale
    # is protected.
    all_psnrs: list[float] = []
    for _, _, pd in panel_data:
        for b_dict in pd.values():
            all_psnrs.extend(b_dict.values())
    if all_psnrs:
        arr = np.asarray(all_psnrs)
        ymin = float(arr.min()) - 1.5
        ymax = float(np.percentile(arr, 95)) + 2.0
    else:
        ymin, ymax = 15.0, 40.0

    # --- Figure layout ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.93,
                        wspace=0.10)

    for i, (ax, (family, label, psnr_data)) in enumerate(zip(axes, panel_data)):
        render_panel(
            ax=ax,
            psnr_data=psnr_data,
            panel_label=label,
            dataset_label=dataset_label,
            ymax_clip=ymax,
        )
        ax.set_ylim(ymin, ymax)
        # Only leftmost panel gets y-label; the others share the axis
        if i > 0:
            ax.set_ylabel("")

    # Single figure-level legend above the panel row. Avoids overlap with
    # curves that the per-panel placement struggled with (block_dct and
    # real_rich rising flanks at b=4-8 on both datasets).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(labels),
        frameon=False,
        fontsize=9,
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_stem) + ".svg", bbox_inches="tight")
    plt.close(fig)

    print(f"  wrote {out_stem}.pdf")
    print(f"  wrote {out_stem}.svg")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render multi-ρ block-size sweep figure (B1 framing)."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIG),
        required=True,
        help="Which dataset to render.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Repo root path (default: parent of tools/).",
    )
    args = parser.parse_args()

    repo_root = Path(args.root) if args.root else Path(__file__).parent.parent
    print(f"Dataset: {args.dataset}")
    build_figure(args.dataset, repo_root)


if __name__ == "__main__":
    main()
