"""Smoke test for the block-structure compute path."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path("results/training/2_direct_training/random_seed/div2k_8q")


@pytest.mark.skipif(not (BASE / "_runs" / "bg").exists(), reason="no trained seeds")
def test_compute_writes_json(tmp_path):
    out = tmp_path / "block_structure.json"
    r = subprocess.run(
        [sys.executable, "tools/render_qft_block_structure.py", "--compute-only",
         "--base", str(BASE), "--orderings", "bg", "--seeds", "50,51",
         "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert d["orderings"]["bg"]["n"] == 2
    assert "16" in d["orderings"]["bg"]["sweep"]
    assert d["orderings"]["bg"]["eff_leakage"]["mean"] < 0.1  # near-perfect block


def test_render_figs_emits_pdf_and_svg(tmp_path):
    # Minimal synthetic agg with one representative seed present on disk.
    if not (BASE / "_runs" / "bg" / "trained_seed_050.json").exists():
        pytest.skip("no representative seed")
    import sys as _sys
    _sys.path.insert(0, "tools")
    from render_qft_block_structure_figs import render_all
    agg = {
        "block_sizes": [2, 4, 8, 16, 32, 64, 128],
        "orderings": {"bg": {
            "n": 1, "freeze_prob": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            "n_mix_row": {"mean": 4, "std": 0, "min": 4, "max": 4},
            "n_mix_col": {"mean": 4, "std": 0, "min": 4, "max": 4},
            "block_size_hist": {"16": 2},
            "sweep": {str(b): {"mean": v, "std": 0.0} for b, v in
                      zip([2, 4, 8, 16, 32, 64, 128],
                          [0.875, 0.75, 0.5, 0.0, 0.0, 0.0, 0.0])},
            "eff_leakage": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "cp_active_frac": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
            "representative_seed": 50, "representative_ordering": "bg"}},
        "pooled": {"n": 1, "block_size_hist": {"16": 2},
                   "eff_leakage": {"mean": 0.0, "std": 0.0}},
    }
    render_all(agg, tmp_path, source_base=BASE)
    figdir = tmp_path / "figures"
    for name in ("block_gate_collapse", "block_operator_heatmap", "block_leakage_sweep"):
        assert (figdir / f"{name}.pdf").exists(), name
        assert (figdir / f"{name}.svg").exists(), name
