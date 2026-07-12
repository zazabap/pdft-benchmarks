"""Shared plotting style for the figure renderers under ``tools/``.

A single source of truth for the three things renderers otherwise re-implement:
the colourblind-safe Wong palette, the paper rcParams (TrueType-embedded
fonts), and the PDF+SVG dual-save convention. Importing these keeps colours and
output formats consistent and lets each renderer stay thin.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Colourblind-safe Wong palette (per the figure conventions in CLAUDE.md).
WONG: dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "vermilion": "#D55E00",
    "sky": "#56B4E9",
    "black": "#000000",
}


def set_paper_rcparams() -> None:
    """Embed TrueType (Type-42) fonts in PDF/PS output (arXiv flags Type-3)."""
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def save_figure(fig, base_path, *, formats=("pdf", "svg"), bbox_inches="tight"):
    """Save ``fig`` to ``base_path`` in each format (default PDF + SVG) -- the
    repo's figure convention (no PNG). Any suffix on ``base_path`` is replaced
    per format, and the parent directory is created. Returns the paths written.
    """
    base = Path(base_path)
    written = []
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format=fmt, bbox_inches=bbox_inches)
        written.append(out)
    return written
