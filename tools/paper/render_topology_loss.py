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
nothing. An inset resolves the tight band, which the full axis cannot show.

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

# The four U(4)/O(2) bases land within ~6 MSE of one another, which is
# invisible on an axis that also has to hold the QFT pair at ~148. The inset
# repeats that band at readable scale.
INSET_YLIM = (99.0, 110.0)
INSET_XLIM = (400, 1010)


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

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    axin = ax.inset_axes([0.40, 0.25, 0.57, 0.31])

    finals = {}
    for basis, label, color, ls, mk in SERIES:
        x, val = load_val(basis)
        finals[label] = val[-1]
        for a, me in ((ax, 14), (axin, 10)):
            a.plot(x, val, color=color, linestyle=ls, linewidth=1.8,
                   marker=mk, markersize=4, markevery=me, markeredgecolor="white",
                   markeredgewidth=0.4, label=label if a is ax else None, zorder=3)

    for label, v in finals.items():
        print(f"{label:14s} final val MSE = {v:.1f}")

    # Inset over the tight band; the QFT pair simply falls outside it.
    axin.set_xlim(*INSET_XLIM)
    axin.set_ylim(*INSET_YLIM)
    axin.tick_params(labelsize=6, length=2, pad=1)
    axin.grid(alpha=0.25, linewidth=0.4)
    for s in ("top", "right"):
        axin.spines[s].set_visible(False)
    ax.indicate_inset_zoom(axin, edgecolor="0.45", linewidth=0.6, alpha=0.8)

    ax.set_xlabel("training step", fontsize=9)
    ax.set_ylabel("validation MSE", fontsize=9)
    ax.set_xlim(0, 1010)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right",
              handlelength=2.2, labelspacing=0.22, borderaxespad=0.2)
    fig.tight_layout(pad=0.4)

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
