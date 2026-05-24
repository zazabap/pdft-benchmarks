"""Smoke test the progressive renderer with synthesized cells, per family."""
import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/opt/conda/envs/pdft/bin/python"
RENDERER = REPO_ROOT / "tools" / "render_qft_progressive.py"


def _synthesize_cells(results_base: Path, family: str) -> None:
    """Populate results_base with a manifest + 8 fake stage cells."""
    runs = results_base / "_runs"
    runs.mkdir(parents=True, exist_ok=True)
    gate_counts = [2, 6, 12, 20, 30, 42, 56, 72]
    psnrs = [10.0, 18.0, 28.0, 30.0, 31.0, 31.5, 31.7, 31.8]
    stages = []
    cumulative = 0
    steps_per_stage = 9
    for k, (g, p) in enumerate(zip(gate_counts, psnrs), start=1):
        cell = runs / f"stage_k{k}"
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "loss_history").mkdir(parents=True, exist_ok=True)
        step_losses = [100.0 - 0.5 * i for i in range(steps_per_stage)]
        val_losses = [step_losses[-1] + 1.0]
        (cell / "loss_history" / f"{family}_progressive_k{k}_loss.json").write_text(json.dumps({
            "step_losses": step_losses,
            "val_losses": val_losses,
            "epochs_completed": 1,
            "steps": steps_per_stage,
        }))
        cumulative += steps_per_stage
        stages.append({
            "k": k, "n_trainable": g, "block_size": 2**k,
            "cell": f"stage_k{k}", "psnr_rho_020": p,
            "steps": steps_per_stage, "elapsed_seconds": 1.0,
        })
    (results_base / "manifest.json").write_text(json.dumps({
        "experiment": f"{family}_progressive",
        "family": family,
        "dataset": "div2k_8q",
        "epochs_per_stage": 1,
        "total_epochs": 8,
        "stages": stages,
        "anchors": {"qft": 31.29, "blocked_8": 32.26},
        "git_sha": "synthetic",
    }))


@pytest.mark.parametrize("family", ["qft", "rich", "real_rich"])
def test_renderer_produces_pdf_and_svg(tmp_path, family):
    results_base = tmp_path / "results"
    _synthesize_cells(results_base, family)
    out_dir = tmp_path / "figures"
    cmd = [
        PYTHON, str(RENDERER),
        "--family", family,
        "--results-base", str(results_base),
        "--out-dir", str(out_dir),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, (
        f"renderer failed:\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
    )
    pdf = out_dir / "training_dynamics.pdf"
    svg = out_dir / "training_dynamics.svg"
    assert pdf.is_file() and pdf.stat().st_size > 1024, \
        f"PDF output missing or too small: {pdf}"
    assert svg.is_file() and svg.stat().st_size > 1024, \
        f"SVG output missing or too small: {svg}"
