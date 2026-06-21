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
