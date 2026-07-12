"""End-to-end smoke test for the qft_progressive driver.

Runs experiments/qft_progressive.py with a tiny budget (smoke preset,
1 epoch per stage) and verifies the per-stage cell layout + manifest
are produced correctly.
"""
import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/opt/conda/envs/pdft/bin/python"
DRIVER = REPO_ROOT / "experiments" / "qft" / "qft_progressive.py"


@pytest.mark.slow
def test_driver_runs_8_stages_smoke(tmp_path):
    """Drive all 8 stages k=1..8 with --epochs-per-stage 1, --preset smoke."""
    out_runs = tmp_path / "runs"
    cmd = [
        PYTHON,
        str(DRIVER),
        "--epochs-per-stage", "1",
        "--out-base", str(out_runs),
        "--preset", "smoke",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, (
        f"driver exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    expected_files_per_stage = [
        "metrics.json",
        "env.json",
        "loss_history/qft_progressive_k{k}_loss.json",
        "trained_qft_progressive_k{k}.json",
    ]
    for k in range(1, 9):
        stage_dir = out_runs / f"stage_k{k}"
        assert stage_dir.is_dir(), f"missing stage directory: {stage_dir}"
        for templ in expected_files_per_stage:
            f = stage_dir / templ.format(k=k)
            assert f.is_file(), f"missing cell file: {f}"

    for k in range(1, 9):
        m = json.loads((out_runs / f"stage_k{k}" / "metrics.json").read_text())
        basis_key = f"qft_progressive_k{k}"
        assert basis_key in m, f"metrics.json missing key {basis_key}"
        psnr = m[basis_key]["metrics"]["0.2"]["mean_psnr"]
        assert isinstance(psnr, (int, float)), \
            f"k={k}: mean_psnr is not numeric ({psnr!r})"

    manifest = out_runs.parent / "manifest.json"
    assert manifest.is_file(), f"missing manifest: {manifest}"
    mf = json.loads(manifest.read_text())
    assert mf["experiment"] == "qft_progressive"
    assert mf["epochs_per_stage"] == 1
    assert len(mf["stages"]) == 8
    for k, stage in enumerate(mf["stages"], start=1):
        assert stage["k"] == k
        assert stage["cell"] == f"stage_k{k}"

    # init_policy: each stage trains independently from identity init;
    # env.json records this as a positive provenance signal.
    for k in range(1, 9):
        env = json.loads((out_runs / f"stage_k{k}" / "env.json").read_text())
        assert env["init_policy"] == "identity", \
            f"stage_k{k} env.json init_policy != 'identity'"
