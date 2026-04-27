"""Unit tests for _extraction module."""

from __future__ import annotations

import pytest

from pdft_benchmarks._extraction import (
    SOURCE_TO_REGISTRY,
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
