"""Smoke tests for the single-operator block-structure compute + figure paths."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# The canonical published trained QFT (a proper 72-gate QFTBasis(8,8)).
BASIS = Path("results/final/div2k_8q/by_basis/qft/trained_qft.json")
DIV2K = Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/DIV2K_train_HR")


@pytest.mark.skipif(not BASIS.exists(), reason="canonical trained QFT not present")
def test_compute_writes_metrics_json(tmp_path):
    out = tmp_path / "block_structure.json"
    r = subprocess.run(
        [sys.executable, "tools/render_qft_block_structure.py", "--compute-only",
         "--basis", str(BASIS), "--out", str(out), "--out-base", str(tmp_path)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert 0 <= d["n_mix_row"] <= 8
    assert d["block_row"] == 2 ** d["n_mix_row"]
    assert "16" in d["leakage_sweep"]
    assert d["eff_leakage"] < 0.1                            # near-perfect block
    assert d["cp_active_frac"] > 0.9


@pytest.mark.skipif(not BASIS.exists(), reason="canonical trained QFT not present")
def test_render_core_emits_pdf_and_svg(tmp_path):
    sys.path.insert(0, "tools")
    from render_qft_block_structure_figs import render_core
    d = json.loads(BASIS.read_text())
    render_core(d, tmp_path)
    figdir = tmp_path / "figures"
    for name in ("block_gate_collapse", "block_operator_heatmap",
                 "block_io_demo", "block_leakage_sweep"):
        assert (figdir / f"{name}.pdf").exists(), name
        assert (figdir / f"{name}.svg").exists(), name


def test_render_freq_spectrum_emits_pdf_and_svg(tmp_path):
    if not BASIS.exists():
        pytest.skip("canonical trained QFT not present")
    if not DIV2K.exists():
        pytest.skip("DIV2K dataset not present")
    sys.path.insert(0, "tools")
    from render_qft_block_structure_figs import render_freq_spectrum
    d = json.loads(BASIS.read_text())
    render_freq_spectrum(d, tmp_path, n_test=4)
    figdir = tmp_path / "figures"
    assert (figdir / "block_freq_spectrum.pdf").exists()
    assert (figdir / "block_freq_spectrum.svg").exists()
