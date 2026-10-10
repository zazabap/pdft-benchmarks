#!/usr/bin/env python3
"""Shared matplotlib style for paper-publication figures.

Targets the manuscript's Springer Nature layout (sn-jnl, two-column `iicol`,
submitted to Quantum Machine Intelligence) and QMI's artwork guidance:
lettering in Helvetica/Arial-style sans serif at 8-12 pt at final size, with
minimal size variance inside a figure. Renderers author every figure at the
exact width it is printed (PAPER_TEXTWIDTH, PAPER_COLUMNWIDTH, or a stated
fraction of them) and save without tight cropping, so the point sizes set here
are the printed sizes.

The face is TeX Gyre Heros (a free Helvetica clone shipped with TeX Live),
registered from the TeX tree when kpsewhich finds it; otherwise the first of
Helvetica, Arial, Liberation Sans, DejaVu Sans that matplotlib knows. Mathtext
maps onto the same face, with STIX sans as the fallback for the few symbols it
lacks. The Wong palette in each renderer is left untouched.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

# sn-jnl iicol: \textwidth = 160 mm, \columnwidth = 76 mm (455.24 and 215.43
# TeX pt); in inches for figsize.
PAPER_TEXTWIDTH = 160 / 25.4
PAPER_COLUMNWIDTH = 76 / 25.4

FONT_SIZE = 8.0        # ticks, legends, annotations, panel text
LABEL_SIZE = 8.5       # axis labels and panel titles

_SANS_FALLBACKS = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
_HEROS = "TeX Gyre Heros"


def _register_heros() -> bool:
    """Register the TeX Gyre Heros OTF faces from the local TeX tree."""
    if any(f.name == _HEROS for f in font_manager.fontManager.ttflist):
        return True
    if shutil.which("kpsewhich") is None:
        return False
    try:
        regular = subprocess.run(["kpsewhich", "texgyreheros-regular.otf"],
                                 capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return False
    if not regular:
        return False
    for face in Path(regular).parent.glob("texgyreheros-*.otf"):
        font_manager.fontManager.addfont(str(face))
    return any(f.name == _HEROS for f in font_manager.fontManager.ttflist)


def sans_family() -> str:
    """The sans-serif face the figures use on this machine."""
    if _register_heros():
        return _HEROS
    known = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in _SANS_FALLBACKS if n in known), "DejaVu Sans")


def apply_paper_style() -> None:
    """Set rcParams for paper figures. Call before creating any figure."""
    family = sans_family()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family] + [n for n in _SANS_FALLBACKS if n != family],
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.sf": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        # Calligraphic capitals (\mathcal{L}) from Computer Modern's symbol
        # font, which matplotlib ships; the sans faces have none.
        "mathtext.cal": "cmsy10",
        "mathtext.fallback": "stixsans",
        "font.size": FONT_SIZE,
        "axes.titlesize": LABEL_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE,
        "figure.titlesize": LABEL_SIZE,
        "figure.labelsize": LABEL_SIZE,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        # Exact-width output: no tight crop, so the saved page is the figsize
        # the renderer chose (its printed width) and the point sizes above
        # are the printed sizes.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


__all__ = ["apply_paper_style", "sans_family", "PAPER_TEXTWIDTH",
           "PAPER_COLUMNWIDTH", "FONT_SIZE", "LABEL_SIZE"]
