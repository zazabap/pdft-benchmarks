#!/usr/bin/env python3
"""Render the paper's topology-comparison loss curve (Figure 4).

Absolute validation MSE vs training step for the six learned bases (RichBasis,
DCT-IV relaxed, QFT, Entangled QFT, TEBD, MERA), read from the shared seed-42
topology run. A plain machine-learning loss curve:
the y-axis is the loss itself (absolute MSE), not a per-basis L/L0
normalisation, so the floors here are the same numbers reported in the
Val-MSE column of the main results table.

The four basic variants collapse into two near-coincident pairs
(QFT/Entangled QFT, TEBD/MERA); distinct colour + line style per basis keeps
the overlapping members legible.

Reads results/structure/div2k_8q_pca_vs_block_dct/by_basis/<basis>/loss_history/*.json.
Writes .../figures/topology_loss_curve.{pdf,svg}. Use --out to also drop the
PDF into the paper at figures/benchmarks/structure/topology_loss_curve.pdf.

Style per CLAUDE.md: Wong palette, one colour + one line style per basis,
linear axes, no figure-level title, PDF+SVG only. Rendered at column size so
the fonts stay legible without LaTeX shrinking a landscape figure.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # avoid Type-3 (arXiv flags it)
matplotlib.rcParams["ps.fonttype"] = 42
import numpy as np
import matplotlib.pyplot as plt

DDIR = Path("results/structure/div2k_8q_pca_vs_block_dct")

# (basis key, pretty label, Wong colour, line style, marker). Ordering places
# each near-coincident pair adjacently so the legend reads the collapse.
# DCT-IV (relaxed) is the real-orthogonal analogue and a headline competitor to
# RichBasis, so it sits second and gets a high-contrast black dash-dot to stand
# apart from the unitary family's coloured curves.
SERIES = [
    ("rich",          "RichBasis",        "#0072B2", "-",               "o"),
    ("dct4_ctl",      "DCT-IV (relaxed)", "#000000", (0, (3, 1, 1, 1)), "P"),
    ("qft",           "QFT",              "#E69F00", (0, (5, 2)),       "s"),
    ("entangled_qft", "Entangled QFT",    "#D55E00", (0, (1, 1)),       "^"),
    ("tebd",          "TEBD",             "#009E73", (0, (5, 2)),       "D"),
    ("mera",          "MERA",             "#56B4E9", (0, (1, 1)),       "v"),
]


def load_val(basis):
    """Validation-MSE trajectory and its per-epoch training-step x-axis."""
    f = glob.glob(str(DDIR / "by_basis" / basis / "loss_history" / "*.json"))[0]
    j = json.loads(Path(f).read_text())
    val = np.asarray(j["val_losses"], dtype=float)
    steps_per_epoch = max(1, j["steps"] // max(1, j["epochs_completed"]))
    x = (np.arange(len(val)) + 1) * steps_per_epoch
    return x, val


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help="optional extra path to also write the PDF to (e.g. the "
                         "paper's figures/benchmarks/structure/topology_loss_curve.pdf)")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    finals = {}
    for basis, label, color, ls, mk in SERIES:
        x, val = load_val(basis)
        finals[label] = val[-1]
        ax.plot(x, val, color=color, linestyle=ls, linewidth=1.8,
                marker=mk, markersize=4, markevery=14, markeredgecolor="white",
                markeredgewidth=0.4, label=label, zorder=3)

    for label, v in finals.items():
        print(f"{label:14s} final val MSE = {v:.1f}")

    ax.set_xlabel("training step", fontsize=9)
    ax.set_ylabel("validation MSE", fontsize=9)
    ax.set_xlim(0, 1010)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right",
              handlelength=2.4, labelspacing=0.3)
    fig.tight_layout(pad=0.4)

    figdir = DDIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    pdf = figdir / "topology_loss_curve.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(figdir / "topology_loss_curve.svg", bbox_inches="tight")
    print(f"wrote {pdf} (+ .svg)")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
