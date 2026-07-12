#!/usr/bin/env python3
"""Render the per-image oracle adaptive-block-size summary figure and LaTeX
table for both datasets (DIV2K-8q and QuickDraw) in a single invocation.

Figure: 1×2 panel grid (left = DIV2K-8q, right = QuickDraw).
  Each panel: 3 family groups (BlockDCT, BlockBD-PCA, RealRich), 2 bars
  per group (best-fixed-b in grey, per-image adaptive in family colour).
  Gain annotated above adaptive bar; chosen fixed-best b labelled below
  fixed bar. Single legend banner at top, no figure-level title.

  QuickDraw panel uses a broken y-axis because the block_dct fixed-best
  at b=2 (54.9 dB) is a sparse-tile pathology that sits ~22 dB above the
  fitted/trained families. Without a break, block_bd_pca and real_rich
  bars are invisible slivers.

Table: tall LaTeX tabular (6 rows: family × dataset) written to
  results/adaptive_block_size/tables/adaptive_per_image.tex.

Outputs:
  results/adaptive_block_size/figures/adaptive_per_image.{pdf,svg}
  results/adaptive_block_size/tables/adaptive_per_image.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Palette — Wong colourblind-safe, matching family identifiers used in sweep
# ---------------------------------------------------------------------------
FAMILY_COLOR = {
    "block_dct":    "#0072B2",  # Wong blue
    "block_bd_pca": "#CC79A7",  # Wong pink
    "real_rich":    "#000000",  # black
}

FAMILY_LABEL = {
    "block_dct":    "BlockDCT\n(HEVC-style)",
    "block_bd_pca": "BlockBD-PCA\n(fitted)",
    "real_rich":    "RealRich\n(trained)",
}

FAMILIES = ["block_dct", "block_bd_pca", "real_rich"]

DATASET_METRICS = {
    "div2k_8q": "results/adaptive_block_size/div2k_8q/per_image/metrics.json",
    "quickdraw": "results/adaptive_block_size/quickdraw/per_image/metrics.json",
}

DATASET_LABEL = {
    "div2k_8q": "(DIV2K-8q)",
    "quickdraw": "(QuickDraw)",
}


def load_metrics(repo_root: Path) -> dict[str, dict]:
    """Load both dataset metrics JSON files. Returns {dataset: data}."""
    result = {}
    for ds, rel_path in DATASET_METRICS.items():
        p = repo_root / rel_path
        if not p.exists():
            raise FileNotFoundError(f"Metrics not found: {p}")
        result[ds] = json.loads(p.read_text())
    return result


def _b_label(b_str: str) -> str:
    """Extract numeric block-size from keys like 'block_dct_8' → 'b=8'."""
    parts = b_str.rsplit("_", 1)
    return f"b={parts[-1]}"


def _gain_str(gain: float) -> str:
    """Format a gain value as '+0.07', '+0.008', etc."""
    if gain < 0.01:
        return f"+{gain:.3f}"
    return f"+{gain:.2f}"


def _draw_bars_on_ax(ax, x_fixed, x_adapt, fixed_psnrs, adapt_psnrs,
                     gains, b_labels, bar_width,
                     y_min_data, y_range_data,
                     annotate_gains: bool, annotate_b: bool,
                     bar_bottom: float = 0.0,
                     ann_gain_abs: float | None = None,
                     ann_b_offset_frac: float = 0.03) -> None:
    """Draw bars and optional annotations onto a single Axes.

    annotate_gains: draw +dB text above adaptive bars.
    annotate_b: draw b= text just above the bar_bottom line.
    bar_bottom: baseline for bars (set to ylim_min for broken-axis panels).
    ann_gain_abs: absolute y-offset above bar top for gain labels; if None
        uses max(y_range_data * 0.015, 0.005) — enough for tiny ranges.
    ann_b_offset_frac: fraction of y_range_data above bar_bottom for b= label.
    """
    import numpy as np  # noqa: F401 (already imported in callers)
    if ann_gain_abs is None:
        ann_gain_abs = max(y_range_data * 0.015, 0.005)

    for i, fam in enumerate(FAMILIES):
        color = FAMILY_COLOR[fam]

        # Height = value - baseline so bars only occupy their relevant range.
        fixed_h = fixed_psnrs[i] - bar_bottom
        adapt_h = adapt_psnrs[i] - bar_bottom

        ax.bar(
            x_fixed[i], fixed_h,
            bottom=bar_bottom,
            width=bar_width,
            color="#d0d0d0",
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )
        ax.bar(
            x_adapt[i], adapt_h,
            bottom=bar_bottom,
            width=bar_width,
            color=color,
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )

        if annotate_gains:
            bar_top = adapt_psnrs[i]
            ann_y = bar_top + ann_gain_abs
            ax.text(
                x_adapt[i], ann_y, _gain_str(gains[i]),
                ha="center", va="bottom",
                fontsize=7.5, color=color, fontweight="bold",
                zorder=4,
            )

        if annotate_b:
            # Place just above the bar bottom (inside the bar).
            label_y = bar_bottom + y_range_data * ann_b_offset_frac
            ax.text(
                x_fixed[i], label_y, b_labels[i],
                ha="center", va="bottom",
                fontsize=7.0, color="#555555",
                zorder=4,
            )


def plot_panel_simple(ax, data: dict, dataset_key: str) -> None:
    """Standard (non-broken) grouped bar chart for DIV2K-8q."""
    import numpy as np

    bar_width = 0.35
    group_gap = 0.1
    group_width = 2 * bar_width + group_gap
    x_centres = np.arange(len(FAMILIES)) * (group_width + 0.3)
    x_fixed = x_centres - bar_width / 2
    x_adapt = x_centres + bar_width / 2

    fixed_psnrs, adapt_psnrs, gains, b_labels = [], [], [], []
    for fam in FAMILIES:
        fd = data[fam]
        fixed_psnrs.append(fd["fixed_best"]["psnr"])
        adapt_psnrs.append(fd["adaptive"]["psnr"])
        gains.append(fd["adaptive"]["gain_over_fixed_best_db"])
        b_labels.append(_b_label(fd["fixed_best"]["b"]))

    fixed_psnrs = np.array(fixed_psnrs)
    adapt_psnrs = np.array(adapt_psnrs)
    all_vals = np.concatenate([fixed_psnrs, adapt_psnrs])
    y_min = all_vals.min()
    y_max = all_vals.max()
    y_range = max(y_max - y_min, 1.0)

    # Absolute gain-label offset: at least 3% of y_range but floor at 0.01 dB
    # so the RealRich "+0.008" doesn't sit directly on the bar top.
    gain_abs = max(y_range * 0.03, 0.01)
    # bar_bottom is the axis lower limit so bars don't extend to y=0
    bar_btm = y_min - y_range * 0.12

    _draw_bars_on_ax(
        ax, x_fixed, x_adapt, fixed_psnrs, adapt_psnrs,
        gains, b_labels, bar_width,
        y_min_data=y_min, y_range_data=y_range,
        annotate_gains=True, annotate_b=True,
        bar_bottom=bar_btm,
        ann_gain_abs=gain_abs,
        ann_b_offset_frac=0.06,
    )

    ax.set_xticks(x_centres)
    ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES], fontsize=8)
    ax.set_ylabel("PSNR (dB)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(x_fixed[0] - bar_width, x_adapt[-1] + bar_width * 1.5)
    ax.set_ylim(bar_btm, y_max + y_range * 0.20)

    # Dataset sub-label in top-left (clear of bars)
    ax.text(
        0.02, 0.97, DATASET_LABEL[dataset_key],
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8.5, color="#444444", style="italic",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.30, linewidth=0.5, color="#999999", zorder=0)
    ax.set_axisbelow(True)


def plot_panel_broken(ax_top, ax_bot, data: dict, dataset_key: str) -> None:
    """Broken y-axis panel for QuickDraw.

    QuickDraw's block_dct fixed-best sits at b=2 / 54.9 dB due to the
    sparse-tile pathology, while the fitted/trained families live at
    28–33 dB — a ~22 dB gap. This function uses two vertically-stacked
    axes sharing x (ax_top = high range, ax_bot = low range) with
    diagonal break markers at the split.

    ax_top and ax_bot must already be positioned so ax_top sits directly
    above ax_bot (no vertical gap visible from the outside); the caller
    handles subplot layout.
    """
    import numpy as np

    bar_width = 0.35
    group_gap = 0.1
    group_width = 2 * bar_width + group_gap
    x_centres = np.arange(len(FAMILIES)) * (group_width + 0.3)
    x_fixed = x_centres - bar_width / 2
    x_adapt = x_centres + bar_width / 2

    fixed_psnrs, adapt_psnrs, gains, b_labels = [], [], [], []
    for fam in FAMILIES:
        fd = data[fam]
        fixed_psnrs.append(fd["fixed_best"]["psnr"])
        adapt_psnrs.append(fd["adaptive"]["psnr"])
        gains.append(fd["adaptive"]["gain_over_fixed_best_db"])
        b_labels.append(_b_label(fd["fixed_best"]["b"]))

    fixed_psnrs = np.array(fixed_psnrs)
    adapt_psnrs = np.array(adapt_psnrs)

    # Split: block_dct (index 0) is high, block_bd_pca + real_rich are low.
    hi_min, hi_max = 54.5, 55.4   # top axis: just the DCT outlier
    lo_vals = np.concatenate([fixed_psnrs[1:], adapt_psnrs[1:]])
    lo_min = lo_vals.min()
    lo_max = lo_vals.max()
    lo_range = max(lo_max - lo_min, 0.5)

    # Each axis gets its own baseline so bars are short segments, not
    # full columns from y=0.
    hi_bottom = hi_min
    lo_bottom = lo_min - lo_range * 0.30  # extra room so short bars are visible

    # Top axis: draw only the DCT family (index 0), baseline = hi_bottom.
    # We pass all three families but only DCT has a value in [hi_min, hi_max];
    # the others produce zero-height bars (height = value - hi_bottom < 0)
    # which matplotlib clips.  Simpler: draw explicitly per-family.
    dct_color = FAMILY_COLOR["block_dct"]
    ax_top.bar(x_fixed[0], fixed_psnrs[0] - hi_bottom, bottom=hi_bottom,
               width=bar_width, color="#d0d0d0", edgecolor=dct_color,
               linewidth=1.2, zorder=3)
    ax_top.bar(x_adapt[0], adapt_psnrs[0] - hi_bottom, bottom=hi_bottom,
               width=bar_width, color=dct_color, edgecolor=dct_color,
               linewidth=1.2, zorder=3)
    # DCT gain annotation on ax_top
    hi_range = hi_max - hi_min
    ax_top.text(x_adapt[0], adapt_psnrs[0] + hi_range * 0.05,
                _gain_str(gains[0]),
                ha="center", va="bottom",
                fontsize=7.5, color=dct_color, fontweight="bold", zorder=5)
    # DCT b= label inside bar in ax_top
    ax_top.text(x_fixed[0], hi_bottom + hi_range * 0.08, b_labels[0],
                ha="center", va="bottom",
                fontsize=7.0, color="#555555", zorder=5)

    # Bottom axis: draw block_bd_pca and real_rich (indices 1, 2).
    for i in [1, 2]:
        fam = FAMILIES[i]
        color = FAMILY_COLOR[fam]
        ax_bot.bar(x_fixed[i], fixed_psnrs[i] - lo_bottom, bottom=lo_bottom,
                   width=bar_width, color="#d0d0d0", edgecolor=color,
                   linewidth=1.2, zorder=3)
        ax_bot.bar(x_adapt[i], adapt_psnrs[i] - lo_bottom, bottom=lo_bottom,
                   width=bar_width, color=color, edgecolor=color,
                   linewidth=1.2, zorder=3)
        # Gain annotation
        ax_bot.text(x_adapt[i], adapt_psnrs[i] + lo_range * 0.06,
                    _gain_str(gains[i]),
                    ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="bold", zorder=5)
        # b= label inside bar
        ax_bot.text(x_fixed[i], lo_bottom + lo_range * 0.06, b_labels[i],
                    ha="center", va="bottom",
                    fontsize=7.0, color="#555555", zorder=5)

    # Set axis limits
    ax_top.set_ylim(hi_min, hi_max)
    ax_bot.set_ylim(lo_bottom, lo_max + lo_range * 0.30)

    x_lo = x_fixed[0] - bar_width
    x_hi = x_adapt[-1] + bar_width * 1.5
    ax_top.set_xlim(x_lo, x_hi)
    ax_bot.set_xlim(x_lo, x_hi)

    # x-axis ticks + labels only on bottom axis
    ax_bot.set_xticks(x_centres)
    ax_bot.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES], fontsize=8)
    ax_bot.tick_params(axis="x", length=0)
    ax_top.set_xticks([])

    # y-axis labels: one on the left of the whole combined panel (bot axis)
    ax_bot.set_ylabel("PSNR (dB)", fontsize=8)
    ax_bot.tick_params(axis="y", labelsize=7)
    ax_top.tick_params(axis="y", labelsize=7)

    # Remove y-ticks that sit too close to the break boundary (within 10% of
    # each axis's data range) — avoids crowding the break-mark area.
    hi_range = hi_max - hi_min
    lo_ylim_top = lo_max + lo_range * 0.30  # matches set_ylim above
    for ax, y_thresh_lo, y_thresh_hi, axis_range in [
        (ax_top, hi_min, hi_max, hi_range),
        (ax_bot, lo_bottom, lo_ylim_top, lo_range),
    ]:
        ticks = [t for t in ax.get_yticks()
                 if (t - y_thresh_lo) > axis_range * 0.08
                 and (y_thresh_hi - t) > axis_range * 0.08]
        ax.set_yticks(ticks)

    # Hide the inner spines at the break
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    # Hide other unnecessary spines
    for ax in (ax_top, ax_bot):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)  # already set; harmless

    # Grids
    for ax in (ax_top, ax_bot):
        ax.yaxis.grid(True, alpha=0.30, linewidth=0.5, color="#999999", zorder=0)
        ax.set_axisbelow(True)

    # Draw diagonal break marks on the LEFT spine only (standard convention).
    # The right spine is hidden, so right-side marks would float in mid-air.
    # Marks appear on the bottom edge of ax_top and the top edge of ax_bot,
    # both on the left side (x ≈ 0 in axes fraction).
    d = 0.022  # size in axes fraction
    # bottom edge of ax_top — left side only
    ax_top.plot((-d, +d), (-d, +d),
                transform=ax_top.transAxes, color="#666666",
                clip_on=False, linewidth=1.2)
    # top edge of ax_bot — left side only
    ax_bot.plot((-d, +d), (1 - d, 1 + d),
                transform=ax_bot.transAxes, color="#666666",
                clip_on=False, linewidth=1.2)

    # Dataset sub-label in top-left of ax_top
    ax_top.text(
        0.02, 0.97, DATASET_LABEL[dataset_key],
        transform=ax_top.transAxes,
        ha="left", va="top",
        fontsize=8.5, color="#444444", style="italic",
    )


def render_figure(metrics: dict[str, dict], out_stem: Path) -> None:
    """Render 1×2 figure and write PDF + SVG.

    Layout: left column = DIV2K-8q (single axes), right column = QuickDraw
    (broken axis: top sub-panel + bottom sub-panel, height ratio 1:2).
    We use gridspec to keep both columns at the same total height.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(9.2, 4.0))

    # Outer 1×2 gridspec divides the figure into left and right halves
    outer = gridspec.GridSpec(1, 2, figure=fig,
                              left=0.07, right=0.98,
                              top=0.87, bottom=0.14,
                              wspace=0.28)

    # Left column: single axes
    ax_left = fig.add_subplot(outer[0, 0])

    # Right column: nested gridspec with 2 rows (top=hi, bot=lo), ratio 1:2
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer[0, 1],
        hspace=0.04,         # tiny gap so break marks are visible
        height_ratios=[1, 2],
    )
    ax_right_top = fig.add_subplot(inner[0])
    ax_right_bot = fig.add_subplot(inner[1])

    # Draw panels
    plot_panel_simple(ax_left, metrics["div2k_8q"], "div2k_8q")
    plot_panel_broken(ax_right_top, ax_right_bot, metrics["quickdraw"], "quickdraw")

    # Single top-of-figure legend (two entries)
    legend_handles = [
        Patch(facecolor="#d0d0d0", edgecolor="#555555", linewidth=1.2,
              label="best fixed-$b$"),
        Patch(facecolor="#555555", edgecolor="#555555", linewidth=1.2,
              label="per-image adaptive"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
        fontsize=8.5,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_stem.with_suffix(".pdf")
    svg_path = out_stem.with_suffix(".svg")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"[viz] wrote {pdf_path}")
    print(f"[viz] wrote {svg_path}")
    plt.close(fig)


def render_table(metrics: dict[str, dict], out_path: Path) -> None:
    """Write tall LaTeX tabular to out_path. Values read from JSON."""

    rows = []
    dataset_display = {
        "div2k_8q": "DIV2K-8q",
        "quickdraw": "Quick Draw",
    }
    family_display = {
        "block_dct":    "BlockDCT",
        "block_bd_pca": "BlockBD-PCA",
        "real_rich":    "RealRich",
    }

    for ds in ["div2k_8q", "quickdraw"]:
        for fam in FAMILIES:
            fam_data = metrics[ds][fam]
            b_val = _b_label(fam_data["fixed_best"]["b"])  # "b=8" → want just "8"
            b_num = b_val.replace("b=", "")
            fixed_psnr = fam_data["fixed_best"]["psnr"]
            adapt_psnr = fam_data["adaptive"]["psnr"]
            gain = fam_data["adaptive"]["gain_over_fixed_best_db"]
            # Format gain with 3 sig figs; keep leading + sign
            if gain < 0.01:
                gain_str = f"$+{gain:.3f}$"
            else:
                gain_str = f"$+{gain:.2f}$"
            rows.append((
                dataset_display[ds],
                family_display[fam],
                b_num,
                f"{fixed_psnr:.2f}",
                f"{adapt_psnr:.2f}",
                gain_str,
            ))

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Family & fixed-best $b$ & fixed PSNR & adapt PSNR & gain (dB) \\",
        r"\midrule",
    ]
    prev_ds = None
    for i, (ds, fam, b, fp, ap, g) in enumerate(rows):
        if prev_ds is not None and ds != prev_ds:
            lines.append(r"\midrule")
        lines.append(f"{ds} & {fam} & {b} & {fp} & {ap} & {g} \\\\")
        prev_ds = ds
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[table] wrote {out_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-figure",
        default=str(repo_root / "results/adaptive_block_size/figures/adaptive_per_image"),
        help="Output figure path stem (without extension). .pdf and .svg are appended.",
    )
    ap.add_argument(
        "--out-table",
        default=str(repo_root / "results/adaptive_block_size/tables/adaptive_per_image.tex"),
        help="Output LaTeX table path.",
    )
    args = ap.parse_args()

    metrics = load_metrics(repo_root)

    render_figure(metrics, Path(args.out_figure))
    render_table(metrics, Path(args.out_table))


if __name__ == "__main__":
    main()
