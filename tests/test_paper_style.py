import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl

_spec = importlib.util.spec_from_file_location(
    "paper_style", Path(__file__).resolve().parents[1] / "tools" / "analysis" / "paper_style.py")
paper_style = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paper_style)


def test_constants_present():
    assert paper_style.PAPER_TEXTWIDTH == 7.0
    assert paper_style.PAPER_COLUMNWIDTH == 3.4


def test_apply_paper_style_sets_rcparams():
    mpl.rcParams["font.family"] = ["sans-serif"]
    mpl.rcParams["pdf.fonttype"] = 3
    paper_style.apply_paper_style()
    assert mpl.rcParams["font.family"] == ["serif"]
    assert mpl.rcParams["mathtext.fontset"] == "cm"
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["savefig.bbox"] == "tight"
    assert mpl.rcParams["axes.titlesize"] == 8.5
