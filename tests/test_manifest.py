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
    assert CLASSICAL_BASELINES == [
        "fft", "dct", "block_fft_8", "block_dct_8",
        "pca", "block_pca_8",
        "dct_rank", "block_dct_8_rank", "pca_rank", "block_pca_8_rank",
    ]


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


import json
from pathlib import Path

from pdft_benchmarks._manifest import build_manifest


def _write_active_cell(root: Path, dataset: str, basis: str, psnr_at_0_1: float):
    cell = root / f"{dataset}__{basis}"
    cell.mkdir(parents=True)
    (cell / "metrics.json").write_text(json.dumps({
        basis: {"metrics": {kr: {"mean_psnr": psnr_at_0_1 + i,
                                  "std_psnr": 0.0, "mean_mse": 0.0,
                                  "std_mse": 0.0, "mean_ssim": 0.5,
                                  "std_ssim": 0.0, "nan_count": 0}
                            for i, kr in enumerate(["0.05", "0.1", "0.15", "0.2"])},
                "time": 50.0,
                "_pdft_py": {"warmup_s": 5.0, "device": "cuda:0",
                             "epochs_completed": 60, "steps": 600,
                             "n_test": 50,
                             "eval_failed_count": {kr: 0 for kr in ["0.05", "0.1", "0.15", "0.2"]}}},
        "fft": {"metrics": {}, "time": 0.0},
    }))
    (cell / "config.json").write_text(json.dumps({"epochs": 60, "n_train": 500, "n_test": 50,
                                                   "lr_peak": 0.3, "batch_size": 8, "seed": 0,
                                                   "preset": "generalized"}))


def _write_skipped_cell_helper(root: Path, dataset: str, basis: str, m: int, n: int):
    cell = root / f"{dataset}__{basis}"
    cell.mkdir(parents=True)
    (cell / "SKIPPED.json").write_text(json.dumps({
        "reason": "incompatible_qubits", "m": m, "n": n, "basis": basis,
        "constraint": "m+n must be a power of 2",
    }))


def test_build_manifest_produces_21_cells(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir()
    for ds in ("div2k_8q", "div2k_10q", "quickdraw"):
        for b in ("qft", "entangled_qft", "tebd", "blocked", "rich", "real_rich"):
            _write_active_cell(pub, ds, b, 27.0)
    _write_active_cell(pub, "div2k_8q", "mera", 27.0)
    _write_skipped_cell_helper(pub, "div2k_10q", "mera", 10, 10)
    _write_skipped_cell_helper(pub, "quickdraw", "mera", 5, 5)

    manifest = build_manifest(pub, git_sha="deadbeef", pdft_version="0.2.1")
    assert len(manifest["cells"]) == 21
    statuses = [c["status"] for c in manifest["cells"]]
    assert statuses.count("active") == 19
    assert statuses.count("skipped") == 2
    assert manifest["schema_version"] == "1.0"
    assert manifest["pdft_version"] == "0.2.1"
    assert manifest["git_sha"] == "deadbeef"


def test_build_manifest_includes_psnr_summary(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir()
    _write_active_cell(pub, "div2k_8q", "qft", 27.0)
    _write_active_cell(pub, "div2k_8q", "entangled_qft", 28.0)
    _write_active_cell(pub, "div2k_8q", "tebd", 28.0)
    _write_active_cell(pub, "div2k_8q", "mera", 28.0)
    _write_active_cell(pub, "div2k_8q", "blocked", 28.0)
    _write_active_cell(pub, "div2k_8q", "rich", 28.0)
    _write_active_cell(pub, "div2k_8q", "real_rich", 28.0)
    _write_skipped_cell_helper(pub, "div2k_10q", "mera", 10, 10)
    for b in ("qft", "entangled_qft", "tebd", "blocked", "rich", "real_rich"):
        _write_active_cell(pub, "div2k_10q", b, 25.0)
    _write_skipped_cell_helper(pub, "quickdraw", "mera", 5, 5)
    for b in ("qft", "entangled_qft", "tebd", "blocked", "rich", "real_rich"):
        _write_active_cell(pub, "quickdraw", b, 22.0)

    manifest = build_manifest(pub, git_sha="abc", pdft_version="0.2.1")
    qft_cell = next(c for c in manifest["cells"] if c["id"] == "div2k_8q__qft")
    assert qft_cell["metrics_summary"]["psnr_at_keep_0.1"] == 28.0  # 27.0 + i=1
    skipped = next(c for c in manifest["cells"] if c["id"] == "div2k_10q__mera")
    assert "metrics_summary" not in skipped
    assert skipped["skip_reason"].startswith("incompatible_qubits")


import pytest
from pdft_benchmarks._manifest import validate_manifest, ManifestValidationError


def _populate_required_files(cell_dir: Path):
    (cell_dir / "env.json").write_text("{}")
    for fname in ("rate_distortion_mse.csv", "rate_distortion_psnr.csv",
                  "rate_distortion_ssim.csv", "timing_summary.csv"):
        (cell_dir / fname).write_text("basis,keep_ratio,mean,std\n")
    (cell_dir / "loss_history").mkdir(exist_ok=True)


def test_validate_passes_on_well_formed_tree(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir()
    for ds in ("div2k_8q", "div2k_10q", "quickdraw"):
        for b in ("qft", "entangled_qft", "tebd", "blocked", "rich", "real_rich"):
            _write_active_cell(pub, ds, b, 27.0)
            _populate_required_files(pub / f"{ds}__{b}")
    _write_active_cell(pub, "div2k_8q", "mera", 27.0)
    _populate_required_files(pub / "div2k_8q__mera")
    _write_skipped_cell_helper(pub, "div2k_10q", "mera", 10, 10)
    _write_skipped_cell_helper(pub, "quickdraw", "mera", 5, 5)

    manifest = build_manifest(pub, git_sha="abc", pdft_version="0.2.1")
    (pub / "MANIFEST.json").write_text(json.dumps(manifest))

    validate_manifest(pub)


def test_validate_fails_when_active_cell_missing_required_file(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir()
    _write_active_cell(pub, "div2k_8q", "qft", 27.0)
    # Intentionally do NOT call _populate_required_files
    manifest = {
        "schema_version": "1.0",
        "datasets": DATASETS, "bases": BASES,
        "classical_baselines": CLASSICAL_BASELINES,
        "cells": [{"id": "div2k_8q__qft", "dataset": "div2k_8q",
                    "basis": "qft", "status": "active",
                    "path": "div2k_8q__qft/", "preset": "generalized",
                    "config": {"epochs": 60, "n_train": 500, "n_test": 100,
                               "lr_peak": 0.3, "batch_size": 8, "seed": 0},
                    "metrics_summary": {"psnr_at_keep_0.1": 28.0,
                                         "psnr_at_keep_0.05": 27.0,
                                         "psnr_at_keep_0.15": 29.0,
                                         "psnr_at_keep_0.2": 30.0,
                                         "train_time_s": 50.0}}],
        "git_sha": "abc", "pdft_version": "0.2.1", "generated_at": "x",
    }
    (pub / "MANIFEST.json").write_text(json.dumps(manifest))

    with pytest.raises(ManifestValidationError, match="missing required file"):
        validate_manifest(pub)


def test_validate_fails_on_skipped_cell_with_extra_files(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir()
    _write_skipped_cell_helper(pub, "div2k_10q", "mera", 10, 10)
    (pub / "div2k_10q__mera" / "stowaway.json").write_text("{}")
    manifest = {
        "schema_version": "1.0",
        "datasets": DATASETS, "bases": BASES,
        "classical_baselines": CLASSICAL_BASELINES,
        "cells": [{"id": "div2k_10q__mera", "dataset": "div2k_10q",
                    "basis": "mera", "status": "skipped",
                    "path": "div2k_10q__mera/",
                    "skip_reason": "incompatible_qubits: m+n=20 is not a power of 2"}],
        "git_sha": "abc", "pdft_version": "0.2.1", "generated_at": "x",
    }
    (pub / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifestValidationError, match="extra files in skipped cell"):
        validate_manifest(pub)


def test_validate_fails_when_metrics_summary_disagrees(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir()
    _write_active_cell(pub, "div2k_8q", "qft", 27.0)
    _populate_required_files(pub / "div2k_8q__qft")
    manifest = build_manifest(pub, git_sha="x", pdft_version="0.2.1")
    for c in manifest["cells"]:
        if c["id"] == "div2k_8q__qft":
            c["metrics_summary"]["psnr_at_keep_0.1"] = 99.0
    (pub / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifestValidationError, match="metrics_summary mismatch"):
        validate_manifest(pub)
