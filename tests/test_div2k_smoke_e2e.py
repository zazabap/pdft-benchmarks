"""Layer B (opt-in): div2k 8q experiment smoke end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdft_benchmarks.datasets.div2k import DEFAULT_DIV2K_ROOT
from pdft_benchmarks.pipeline import run_experiment


@pytest.mark.integration
def test_div2k_smoke_e2e(tmp_path: Path):
    if not DEFAULT_DIV2K_ROOT.is_dir():
        pytest.skip(f"DIV2K not available at {DEFAULT_DIV2K_ROOT}")

    out_dir = tmp_path / "div2k_8q_smoke"
    run_experiment(
        dataset="div2k",
        m=8,
        n=8,
        bases=["qft", "entangled_qft", "tebd", "mera"],
        baselines=["fft", "dct", "block_fft_8", "block_dct_8", "pca", "block_pca_8"],
        preset="smoke",
        output_dir=out_dir,
        device="cpu",
    )
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert set(metrics.keys()) == {
        "qft", "entangled_qft", "tebd", "mera",
        "fft", "dct", "block_fft_8", "block_dct_8",
        "pca", "block_pca_8",
    }
    # MERA either runs successfully OR fails due to memory constraints — both
    # are valid for a smoke test. m+n=16 is a power of 2, so MERA is not
    # skipped; it will attempt to train and may OOM on CPU.
    assert "metrics" in metrics["mera"] or "failed" in metrics["mera"], (
        f"mera should attempt on DIV2K-8q; got {metrics['mera']}"
    )
    # Both PCA baselines produce metrics; pca on DIV2K-8q is rank-deficient
    # but still functional (k <= n_train).
    for name in ("pca", "block_pca_8"):
        assert "metrics" in metrics[name]
        fp = metrics[name]["_pdft_py"]["pca_fingerprint"]
        assert fp["d"] == (256 * 256 if name == "pca" else 64)
    assert (out_dir / "block_pca_8_eigenbasis.npz").is_file()


@pytest.mark.integration
def test_div2k_10q_skips_global_pca(tmp_path: Path):
    """At m=n=10 the global PCA is intractable; pipeline must skip it cleanly."""
    if not DEFAULT_DIV2K_ROOT.is_dir():
        pytest.skip(f"DIV2K not available at {DEFAULT_DIV2K_ROOT}")

    out_dir = tmp_path / "div2k_10q_skip"
    run_experiment(
        dataset="div2k",
        m=10,
        n=10,
        bases=[],
        baselines=["pca", "block_pca_8"],
        preset="smoke",
        output_dir=out_dir,
        device="cpu",
    )
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["pca"] == {"skipped": "pca_intractable_at_1m_dim"}
    # block_pca_8 still runs at m=10 (the block is 8x8 regardless).
    assert "metrics" in metrics["block_pca_8"]
