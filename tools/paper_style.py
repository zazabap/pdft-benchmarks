#!/usr/bin/env python3
"""Shared matplotlib style for paper-publication figures.

Matches the two-column `quantumarticle` look: Computer-Modern serif (via mathtext
`cm`, so no LaTeX install is required), small absolute font sizes, thin spines,
TrueType-embedded PDF. Renderers author at PAPER_TEXTWIDTH so that, included at
`width=\\textwidth`, the fonts land at true size. The Wong palette in each
renderer is left untouched.
"""
from __future__ import annotations

import matplotlib as mpl

# quantumarticle (a4, two-column): \textwidth ~ 7.0in (figure*), \columnwidth ~ 3.4in.
PAPER_TEXTWIDTH = 7.0
PAPER_COLUMNWIDTH = 3.4


def apply_paper_style() -> None:
    """Set rcParams for paper figures. Call before creating any figure."""
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
    })


__all__ = ["apply_paper_style", "PAPER_TEXTWIDTH", "PAPER_COLUMNWIDTH"]
