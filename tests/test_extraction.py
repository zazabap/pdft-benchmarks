"""Unit tests for _extraction module."""

from __future__ import annotations

import pytest

from pdft_benchmarks._extraction import (
    rename_basis_key,
)


def test_identity_mapping_for_circuit_bases():
    assert rename_basis_key("qft") == "qft"
    assert rename_basis_key("entangled_qft") == "entangled_qft"
    assert rename_basis_key("tebd") == "tebd"
    assert rename_basis_key("mera") == "mera"


def test_block_bases_get_renamed():
    assert rename_basis_key("blocked_qft") == "blocked"
    assert rename_basis_key("blocked_rich") == "rich"
    assert rename_basis_key("blocked_real") == "real_rich"


def test_unknown_key_raises():
    with pytest.raises(KeyError, match="unknown source basis key"):
        rename_basis_key("nonsense_basis")


from pdft_benchmarks._extraction import filter_metrics_for_cell


def _fake_baseline_block():
    return {"metrics": {"0.05": {"mean_psnr": 20.0, "std_psnr": 0.0,
                                  "mean_mse": 0.01, "std_mse": 0.0,
                                  "mean_ssim": 0.5, "std_ssim": 0.0,
                                  "nan_count": 0}},
            "time": 0.1}


def _fake_basis_block(psnr=27.0):
    return {"metrics": {"0.05": {"mean_psnr": psnr, "std_psnr": 0.5,
                                  "mean_mse": 0.001, "std_mse": 0.0,
                                  "mean_ssim": 0.8, "std_ssim": 0.0,
                                  "nan_count": 0}},
            "time": 100.0,
            "_pdft_py": {"warmup_s": 5.0, "device": "cuda:0",
                         "epochs_completed": 60, "steps": 600,
                         "n_test": 50, "eval_failed_count": {"0.05": 0}}}


def test_filter_keeps_one_basis_plus_4_baselines():
    src = {
        "qft": _fake_basis_block(27.0),
        "entangled_qft": _fake_basis_block(28.0),
        "tebd": _fake_basis_block(29.0),
        "fft": _fake_baseline_block(),
        "dct": _fake_baseline_block(),
        "block_fft_8": _fake_baseline_block(),
        "block_dct_8": _fake_baseline_block(),
    }
    out = filter_metrics_for_cell(src, source_basis_key="qft")
    assert set(out) == {"qft", "fft", "dct", "block_fft_8", "block_dct_8"}
    assert "entangled_qft" not in out
    assert "tebd" not in out


def test_filter_renames_blocked_qft_to_blocked():
    src = {
        "blocked_qft": _fake_basis_block(),
        "fft": _fake_baseline_block(),
        "dct": _fake_baseline_block(),
        "block_fft_8": _fake_baseline_block(),
        "block_dct_8": _fake_baseline_block(),
    }
    out = filter_metrics_for_cell(src, source_basis_key="blocked_qft")
    assert "blocked" in out
    assert "blocked_qft" not in out


def test_filter_raises_when_source_key_missing():
    src = {"fft": _fake_baseline_block()}
    with pytest.raises(KeyError, match="not in source metrics"):
        filter_metrics_for_cell(src, source_basis_key="qft")


from pdft_benchmarks._extraction import build_config_json


def test_build_config_json_extracts_preset_fields():
    env = {
        "preset": "generalized",
        "preset_dataclass": {
            "epochs": 60,
            "n_train": 500,
            "n_test": 100,
            "optimizer": "adam",
            "batch_size": 8,
            "warmup_frac": 0.05,
            "lr_peak": 0.3,
            "lr_final": 0.0003,
            "max_grad_norm": 1.0,
            "validation_split": 0.15,
            "early_stopping_patience": 5,
            "seed": 0,
            "keep_ratios": [0.05, 0.1, 0.15, 0.2],
        },
    }
    cfg = build_config_json(env, m=8, n=8, basis="qft")
    assert cfg["m"] == 8
    assert cfg["n"] == 8
    assert cfg["basis"] == "qft"
    assert cfg["preset"] == "generalized"
    assert cfg["epochs"] == 60
    assert cfg["batch_size"] == 8
    assert cfg["lr_peak"] == 0.3
    assert cfg["seed"] == 0
    assert cfg["keep_ratios"] == [0.05, 0.1, 0.15, 0.2]


def test_build_config_json_raises_on_missing_preset_dataclass():
    with pytest.raises(KeyError, match="preset_dataclass"):
        build_config_json({}, m=8, n=8, basis="qft")


import json
import os
from pathlib import Path

# Ensure matplotlib uses a non-interactive backend (for plot generation in tests)
os.environ.setdefault("MPLBACKEND", "Agg")

from pdft_benchmarks._extraction import extract_cell, write_skipped_cell


def _multi_kr_metrics_block(psnr=27.0):
    """Like _fake_basis_block but covers all 4 keep ratios for plot/CSV happiness."""
    return {"metrics": {kr: {"mean_psnr": psnr + 0.1 * i, "std_psnr": 0.5,
                              "mean_mse": 0.001, "std_mse": 0.0,
                              "mean_ssim": 0.8, "std_ssim": 0.0,
                              "nan_count": 0}
                        for i, kr in enumerate(["0.05", "0.1", "0.15", "0.2"])},
            "time": 100.0,
            "_pdft_py": {"warmup_s": 5.0, "device": "cuda:0",
                         "epochs_completed": 60, "steps": 600,
                         "n_test": 50,
                         "eval_failed_count": {kr: 0 for kr in ["0.05", "0.1", "0.15", "0.2"]}}}


def _multi_kr_baseline_block():
    return {"metrics": {kr: {"mean_psnr": 20.0, "std_psnr": 0.0,
                              "mean_mse": 0.01, "std_mse": 0.0,
                              "mean_ssim": 0.5, "std_ssim": 0.0,
                              "nan_count": 0}
                        for kr in ["0.05", "0.1", "0.15", "0.2"]},
            "time": 0.1}


def _make_fake_source(tmp_path: Path) -> Path:
    """Build a minimal valid source run dir."""
    src = tmp_path / "src_run"
    (src / "loss_history").mkdir(parents=True)
    metrics = {
        "qft": _multi_kr_metrics_block(27.0),
        "fft": _multi_kr_baseline_block(),
    }
    (src / "metrics.json").write_text(json.dumps(metrics))
    (src / "env.json").write_text(json.dumps({
        "preset": "generalized",
        "preset_dataclass": {
            "epochs": 60, "n_train": 500, "n_test": 100, "optimizer": "adam",
            "batch_size": 8, "warmup_frac": 0.05, "lr_peak": 0.3,
            "lr_final": 0.0003, "max_grad_norm": 1.0,
            "validation_split": 0.15, "early_stopping_patience": 5,
            "seed": 0, "keep_ratios": [0.05, 0.1, 0.15, 0.2],
        },
    }))
    (src / "trained_qft.json").write_text(json.dumps({"type": "QFTBasis", "m": 8, "n": 8, "tensors": []}))
    (src / "loss_history" / "qft_loss.json").write_text(json.dumps({"step_losses": [[1.0, 0.5]]}))
    (src / "run.log").write_text("fake log")
    return src


def test_extract_cell_writes_all_required_files(tmp_path):
    src = _make_fake_source(tmp_path)
    dest = tmp_path / "div2k_8q__qft"
    extract_cell(
        source_run=src,
        cell_dir=dest,
        source_basis_key="qft",
        m=8, n=8,
    )
    assert (dest / "metrics.json").is_file()
    assert (dest / "config.json").is_file()
    assert (dest / "env.json").is_file()
    assert (dest / "trained_qft.json").is_file()
    assert (dest / "loss_history" / "qft_loss.json").is_file()
    assert (dest / "run.log").is_file()


def test_extract_cell_renames_blocked_qft_to_blocked(tmp_path):
    src = tmp_path / "blocked_src"
    (src / "loss_history").mkdir(parents=True)
    metrics = {
        "blocked_qft": _multi_kr_metrics_block(28.0),
        "fft": _multi_kr_baseline_block(),
    }
    (src / "metrics.json").write_text(json.dumps(metrics))
    (src / "env.json").write_text(json.dumps({
        "preset": "generalized",
        "preset_dataclass": {
            "epochs": 60, "n_train": 500, "n_test": 100, "optimizer": "adam",
            "batch_size": 8, "warmup_frac": 0.05, "lr_peak": 0.3,
            "lr_final": 0.0003, "max_grad_norm": 1.0,
            "validation_split": 0.15, "early_stopping_patience": 5,
            "seed": 0, "keep_ratios": [0.05, 0.1, 0.15, 0.2],
        },
    }))
    (src / "trained_blocked_qft.json").write_text("{}")
    (src / "loss_history" / "blocked_qft_loss.json").write_text("{}")

    dest = tmp_path / "div2k_8q__blocked"
    extract_cell(source_run=src, cell_dir=dest, source_basis_key="blocked_qft", m=8, n=8)
    assert (dest / "trained_blocked.json").is_file()
    assert (dest / "loss_history" / "blocked_loss.json").is_file()
    assert not (dest / "trained_blocked_qft.json").exists()
    out_metrics = json.loads((dest / "metrics.json").read_text())
    assert "blocked" in out_metrics
    assert "blocked_qft" not in out_metrics


def test_extract_cell_writes_skipped_json_for_skipped_basis(tmp_path):
    dest = tmp_path / "div2k_10q__mera"
    write_skipped_cell(dest, m=10, n=10, basis="mera")
    payload = json.loads((dest / "SKIPPED.json").read_text())
    assert payload["reason"] == "incompatible_qubits"
    assert payload["m"] == 10
    assert payload["n"] == 10
    assert payload["basis"] == "mera"
    assert sorted(p.name for p in dest.iterdir()) == ["SKIPPED.json"]
