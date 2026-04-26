"""Layer B (opt-in): quickdraw experiment smoke end-to-end. CPU-allowed for CI runners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdft_benchmarks.datasets.quickdraw import DEFAULT_QUICKDRAW_ROOT
from pdft_benchmarks.pipeline import run_experiment


@pytest.mark.integration
def test_quickdraw_smoke_e2e(tmp_path: Path):
    if not DEFAULT_QUICKDRAW_ROOT.is_dir():
        pytest.skip(f"QuickDraw not available at {DEFAULT_QUICKDRAW_ROOT}")

    out_dir = tmp_path / "quickdraw_smoke"
    res = run_experiment(
        dataset="quickdraw",
        m=5,
        n=5,
        bases=["qft", "entangled_qft", "tebd", "mera"],
        baselines=["fft", "dct", "block_fft_8", "block_dct_8"],
        preset="smoke",
        output_dir=out_dir,
        device="cpu",
    )
    assert (out_dir / "metrics.json").is_file()
    metrics = json.loads((out_dir / "metrics.json").read_text())
    # 4 quantum bases + 4 baselines = 8 keys.
    assert set(metrics.keys()) == {
        "qft",
        "entangled_qft",
        "tebd",
        "mera",
        "fft",
        "dct",
        "block_fft_8",
        "block_dct_8",
    }
    # MERA skipped on m=n=5 (m+n=10 not power of 2).
    assert metrics["mera"].get("skipped") == "incompatible_qubits"
    assert res.duration_s > 0
