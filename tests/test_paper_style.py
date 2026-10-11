import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl

_spec = importlib.util.spec_from_file_location(
    "paper_style", Path(__file__).resolve().parents[1] / "experiments" / "_paper_style.py")
paper_style = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paper_style)


def test_constants_present():
    # sn-jnl iicol text block: 160 mm text width, 76 mm column width.
    assert paper_style.PAPER_TEXTWIDTH == 160 / 25.4
    assert paper_style.PAPER_COLUMNWIDTH == 76 / 25.4
    # QMI lettering floor: 8 pt text, 8.5 pt labels whose scripts print at 8 pt.
    assert paper_style.FONT_SIZE == 8.0
    assert paper_style.LABEL_SIZE == 8.5
    assert paper_style.SCRIPT_SHRINK == paper_style.FONT_SIZE / paper_style.LABEL_SIZE


def test_apply_paper_style_sets_rcparams():
    mpl.rcParams["font.family"] = ["serif"]
    mpl.rcParams["pdf.fonttype"] = 3
    paper_style.apply_paper_style()
    family = paper_style.sans_family()
    assert mpl.rcParams["font.family"] == ["sans-serif"]
    assert mpl.rcParams["font.sans-serif"][0] == family
    assert mpl.rcParams["mathtext.fontset"] == "custom"
    assert mpl.rcParams["mathtext.rm"] == family
    assert mpl.rcParams["pdf.fonttype"] == 42
    # Exact-width output: no tight crop, so point sizes are printed sizes.
    assert mpl.rcParams["savefig.bbox"] is None
    assert mpl.rcParams["axes.titlesize"] == paper_style.LABEL_SIZE
    assert mpl.rcParams["font.size"] == paper_style.FONT_SIZE
