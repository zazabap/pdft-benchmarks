#!/usr/bin/env python3
"""Render the paper's topology-comparison loss curve (Figure 4).

Absolute validation MSE vs training step for the six learned bases (RichBasis,
DCT-IV relaxed, TEBD, MERA, QFT, Entangled QFT), read from the same seed-42
run series that produces the headline DIV2K table, so the curves and the table
describe one set of trainings. A plain machine-learning loss curve: the y-axis
is the loss itself (absolute MSE), not a per-basis L/L0 normalisation.

The split is by gate, not by wiring. The four bases carrying a general
two-qubit tensor (RichBasis, DCT-IV, TEBD, MERA) floor within ~6 MSE of one
another, while the diagonal-phase QFT pair sits ~45 higher; QFT and Entangled
QFT coincide to the plotted precision, the matched-axis coupling contributing
nothing. The y-axis is broken so that band and the QFT plateau each get their own
scale; a single linear axis renders the band as a smear.

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
import numpy as np
import matplotlib.pyplot as plt

from pdft_benchmarks.plots.style import WONG, save_figure, set_paper_rcparams

set_paper_rcparams()

DDIR = Path("results/structure/div2k_8q_pca_vs_block_dct")

# (basis key, pretty label, Wong colour, line style, marker). Ordering places
# each near-coincident pair adjacently so the legend reads the collapse.
# DCT-IV (relaxed) is the real-orthogonal analogue and a headline competitor to
# RichBasis, so it sits second and gets a high-contrast black dash-dot to stand
# apart from the unitary family's coloured curves.
SERIES = [
    ("rich_full",     "RichBasis",        WONG["blue"],      "-",               "o"),
    ("dct4_ctl",      "DCT-IV (relaxed)", WONG["black"],     (0, (3, 1, 1, 1)), "P"),
    ("tebd_u4",       "TEBD",             WONG["green"],     (0, (5, 2)),       "D"),
    ("mera_u4",       "MERA",             WONG["sky"],       (0, (1, 1)),       "v"),
    ("qft",           "QFT",              WONG["orange"],    (0, (5, 2)),       "s"),
    ("entangled_qft", "Entangled QFT",    WONG["vermilion"], (0, (1, 1)),       "^"),
]

# The four U(4)/O(2) bases land within ~6 MSE of one another while the QFT pair
# sits ~45 above, so a single linear axis renders the interesting band as a
# smear. The y-axis is broken instead: the upper panel holds the descent and
# the QFT plateau, the lower one the band, each at its own scale. This beats an
# inset here because the band gets the full plot width and hides no data.
UPPER_YLIM = (138.0, 280.0)
LOWER_YLIM = (99.0, 117.0)
UPPER_TICKS = (150, 200, 250)
LOWER_TICKS = (100, 105, 110, 115)


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

    fig, (hi, lo) = plt.subplots(
        2, 1, sharex=True, figsize=(3.9, 3.0),
        gridspec_kw=dict(height_ratios=[1, 1.15], hspace=0.08))

    finals = {}
    for basis, label, color, ls, mk in SERIES:
        x, val = load_val(basis)
        finals[label] = val[-1]
        for a in (hi, lo):
            a.plot(x, val, color=color, linestyle=ls, linewidth=1.8,
                   marker=mk, markersize=4, markevery=14, markeredgecolor="white",
                   markeredgewidth=0.4, label=label if a is hi else None, zorder=3)

    for label, v in finals.items():
        print(f"{label:14s} final val MSE = {v:.1f}")

    hi.set_ylim(*UPPER_YLIM)
    lo.set_ylim(*LOWER_YLIM)
    hi.set_yticks(UPPER_TICKS)
    lo.set_yticks(LOWER_TICKS)

    # Drop the spines facing the break, then mark it with the usual diagonals.
    hi.spines["bottom"].set_visible(False)
    lo.spines["top"].set_visible(False)
    hi.tick_params(bottom=False, labelsize=8.5)
    lo.tick_params(labelsize=8.5)
    brk = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=3.6, linestyle="none",
               color="k", mec="k", mew=0.8, clip_on=False)
    hi.plot([0, 1], [0, 0], transform=hi.transAxes, **brk)
    lo.plot([0, 1], [1, 1], transform=lo.transAxes, **brk)

    for a in (hi, lo):
        a.grid(alpha=0.25, linewidth=0.5)
        for sp in a.spines.values():
            sp.set_linewidth(0.8)

    lo.set_xlabel("training step", fontsize=9.5)
    lo.set_xlim(0, 1010)
    hi.legend(fontsize=8, frameon=False, loc="upper right",
              handlelength=2.2, labelspacing=0.22, borderaxespad=0.2)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.985)
    fig.text(0.008, 0.55, "validation MSE", rotation=90, va="center",
             ha="left", fontsize=9.5)

    figdir = DDIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    pdf = figdir / "topology_loss_curve.pdf"
    save_figure(fig, pdf)
    print(f"wrote {pdf} (+ .svg)")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
