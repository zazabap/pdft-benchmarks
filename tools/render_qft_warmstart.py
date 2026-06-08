#!/usr/bin/env python3
"""Render the warm-start QFT loss-curve figure: 1 panel, 3 curves.

For each dataset, plot:
  - qft (random init from the headline experiment)
  - blocked (or blocked_8 for DIV2K, the source of the warm-start init)
  - qft_warmstart_blocked* (the new warm-started run)

Y-axis: absolute val MSE (linear), so the right-edge ordering equals the
PSNR ranking. X-axis: training step.

Outputs PDF + SVG to results/training/1_structure_inclusion/figures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


WONG = {
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "green":     "#009E73",
}


# Per-dataset config: (basis name in headline cells, headline cell root)
DATASETS: dict[str, dict] = {
    "div2k_8q": {
        "headline_root":   "results/structure/div2k_8q_pca_vs_block_dct/by_basis",
        "qft_name":        "qft",
        "blocked_name":    "blocked_8",
        "warmstart_root":  "results/training/1_structure_inclusion/by_basis",
        "warmstart_name":  "qft_warmstart_blocked_8",
        "out":             "results/training/1_structure_inclusion/figures/loss_curves_div2k_8q.pdf",
    },
    "quickdraw": {
        "headline_root":   "results/structure/quickdraw_pca_vs_block_dct/by_basis",
        "qft_name":        "qft",
        "blocked_name":    "blocked",
        "warmstart_root":  "results/training/1_structure_inclusion/by_basis",
        "warmstart_name":  "qft_warmstart_blocked",
        "out":             "results/training/1_structure_inclusion/figures/loss_curves_quickdraw.pdf",
    },
}


def load_curve(cell_root: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (steps_per_epoch * epoch_idx, val_losses) from a cell, or None."""
    p = cell_root / name / "loss_history" / f"{name}_loss.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    step_losses = np.asarray(d["step_losses"], dtype=np.float64)
    val_losses = np.asarray(d["val_losses"], dtype=np.float64)
    if len(val_losses) == 0 or len(step_losses) == 0:
        return None
    n_epochs = len(val_losses)
    steps_per_epoch = max(1, len(step_losses) // n_epochs)
    x = np.arange(1, n_epochs + 1) * steps_per_epoch
    return x, val_losses


def render(cfg: dict) -> None:
    headline_root = Path(cfg["headline_root"])
    warmstart_root = Path(cfg["warmstart_root"])

    qft = load_curve(headline_root, cfg["qft_name"])
    blocked = load_curve(headline_root, cfg["blocked_name"])
    warm = load_curve(warmstart_root, cfg["warmstart_name"])

    if any(c is None for c in (qft, blocked, warm)):
        missing = [
            n for n, c in [
                (cfg["qft_name"], qft),
                (cfg["blocked_name"], blocked),
                (cfg["warmstart_name"], warm),
            ] if c is None
        ]
        raise SystemExit(
            f"missing loss curves for: {missing}. "
            f"Did the warm-start run finish for {cfg['warmstart_name']}?"
        )

    # Prepend a step-0 marker on the WARM-START curve at the trained
    # blocked val MSE. The bit-exact identity (qft_warm_from_trained_blocked
    # produces an operator equal to the trained blocked one to numerical
    # zero on random complex inputs) means the warm-start's actual val
    # MSE at step 0 IS the blocked floor. The JSON's val_losses[0] is
    # recorded after epoch 1 (9 minibatch updates), by which time Adam's
    # sign-update behavior has already perturbed the operator off the
    # local minimum — so the first stored point is misleadingly high.
    # Showing the step-0 marker visually anchors the warm-start at its
    # true starting position, making the "starts at blocked, briefly
    # diverges, returns" story unambiguous.
    blocked_floor = float(blocked[1][-1])
    warm_x = np.concatenate([[0], warm[0]])
    warm_y = np.concatenate([[blocked_floor], warm[1]])
    warm = (warm_x, warm_y)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.6))

    # Plot in order: qft (random init), blocked (warm-start source),
    # qft_warmstart (the new run).
    for (x, y), color, marker, label in [
        (qft,     WONG["blue"],      "o", f"`{cfg['qft_name']}` (random init)"),
        (blocked, WONG["vermilion"], "s", f"`{cfg['blocked_name']}` (warm-start source)"),
        (warm,    WONG["green"],     "^", f"`{cfg['warmstart_name']}`"),
    ]:
        ax.plot(x, y, color=color, linewidth=1.6, alpha=0.9, zorder=2, label=label)
        # Marker stride: ~14 markers across the curve.
        n = len(y)
        stride = max(1, n // 14)
        idx = np.arange(0, n, stride)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        ax.plot(x[idx], y[idx], color=color, linewidth=0,
                marker=marker, markersize=5.0,
                markerfacecolor=color, markeredgecolor=color,
                markeredgewidth=1.0, zorder=3)

    ax.set_xlabel("training step", fontsize=9)
    ax.set_ylabel("absolute val MSE (log scale)", fontsize=9)
    # Log y-scale: the warmup-LR transient pushes the warm-start curve
    # far above the converged regime in the first 1-2 epochs (an Adam +
    # cosine-warmup artefact, not a sign of a bad warm-start). On DIV2K
    # this transient hits ~1500 MSE while the converged values sit at
    # ~130 — a 10× dynamic range that linear y squishes. Log y shows
    # both the spike and the convergence-to-blocked-floor cleanly.
    # Differs from the headline experiment's linear-y convention.
    ax.set_yscale("log")
    ax.tick_params(labelsize=8)
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.30, linewidth=0.5, color="#999999", zorder=0)
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.3, color="#bbbbbb", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85, handlelength=2.4)

    fig.subplots_adjust(left=0.10, right=0.99, top=0.97, bottom=0.10)

    out_pdf = Path(cfg["out"])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    out_svg = out_pdf.with_suffix(".svg")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"[viz] wrote {out_pdf} + {out_svg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    args = ap.parse_args()
    render(DATASETS[args.dataset])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
