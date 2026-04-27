"""Unit tests for _manifest module."""

from __future__ import annotations

from pathlib import Path

from pdft_benchmarks._manifest import (
    SCHEMA_VERSION,
    DATASETS,
    BASES,
    CLASSICAL_BASELINES,
    MERA_INCOMPATIBLE_DATASETS,
)


def test_schema_version_is_string():
    assert SCHEMA_VERSION == "1.0"


def test_datasets_table_has_three_rows():
    assert set(DATASETS) == {"div2k_8q", "div2k_10q", "quickdraw"}
    for name, row in DATASETS.items():
        assert "m" in row and "n" in row
        assert "image_size" in row
        assert row["image_size"] == [2 ** row["m"], 2 ** row["n"]]


def test_bases_table_has_seven_keys():
    assert set(BASES) == {"qft", "entangled_qft", "tebd", "mera",
                          "blocked", "rich", "real_rich"}
    assert BASES["mera"]["constraint"] == "m+n must be power of 2"


def test_classical_baselines_constant():
    assert CLASSICAL_BASELINES == ["fft", "dct", "block_fft_8", "block_dct_8"]


def test_mera_incompatible_datasets():
    assert MERA_INCOMPATIBLE_DATASETS == {"div2k_10q", "quickdraw"}


from pdft_benchmarks._manifest import summarize_metrics


def _make_metrics(psnrs: dict[str, float], time_s: float = 100.0):
    return {
        "qft": {
            "metrics": {
                kr: {"mean_mse": 0.0, "std_mse": 0.0,
                     "mean_psnr": p, "std_psnr": 0.0,
                     "mean_ssim": 0.5, "std_ssim": 0.0,
                     "nan_count": 0}
                for kr, p in psnrs.items()
            },
            "time": time_s,
            "_pdft_py": {"warmup_s": 5.0, "device": "cuda:0",
                         "epochs_completed": 60, "steps": 600,
                         "n_test": 50,
                         "eval_failed_count": {kr: 0 for kr in psnrs}},
        },
        "fft": {"metrics": {"0.05": {"mean_psnr": 20.0, "std_psnr": 0.0,
                                      "mean_mse": 0.0, "std_mse": 0.0,
                                      "mean_ssim": 0.0, "std_ssim": 0.0,
                                      "nan_count": 0}}, "time": 0.1},
    }


def test_summarize_extracts_psnr_per_keep_ratio():
    metrics = _make_metrics({"0.05": 24.5, "0.1": 27.0, "0.15": 29.0, "0.2": 30.5})
    summary = summarize_metrics(metrics, basis_key="qft")
    assert summary["psnr_at_keep_0.05"] == 24.5
    assert summary["psnr_at_keep_0.1"] == 27.0
    assert summary["psnr_at_keep_0.15"] == 29.0
    assert summary["psnr_at_keep_0.2"] == 30.5
    assert summary["train_time_s"] == 100.0


def test_summarize_returns_nan_for_missing_keep_ratio():
    import math
    metrics = _make_metrics({"0.05": 24.5})
    summary = summarize_metrics(metrics, basis_key="qft")
    assert math.isnan(summary["psnr_at_keep_0.1"])


def test_summarize_raises_if_basis_missing():
    import pytest
    metrics = {"fft": {"metrics": {}, "time": 0.0}}
    with pytest.raises(KeyError, match="basis 'qft' not in metrics"):
        summarize_metrics(metrics, basis_key="qft")
