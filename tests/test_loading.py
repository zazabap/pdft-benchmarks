"""Layer A: trained-basis loader smoke tests. No training, no GPU.

Uses the existing committed cells from results/block_size_sweep/quickdraw/
to verify load + roundtrip works.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def quickdraw_blocked_8_path():
    """Path to a pre-trained blocked_8 cell. Skip if not present (CI env)."""
    candidates = [
        Path("results/block_size_sweep/quickdraw/by_basis/blocked_8/trained_blocked_8.json"),
        Path("results/block_size_sweep/quickdraw/by_basis/blocked_16/trained_blocked_16.json"),
        Path("results/quickdraw_pca_vs_block_dct/by_basis/blocked/trained_blocked.json"),
    ]
    for p in candidates:
        if p.exists():
            return p
    pytest.skip("no committed trained cell available for the smoke test")


def test_load_trained_basis_returns_callable_basis(quickdraw_blocked_8_path):
    from pdft_benchmarks._loading import load_trained_basis

    basis = load_trained_basis(quickdraw_blocked_8_path)
    assert hasattr(basis, "forward_transform")
    assert hasattr(basis, "inverse_transform")


def test_make_compress_fn_roundtrip(quickdraw_blocked_8_path):
    """At keep_ratio=1.0, make_compress_fn must agree with pdft.io.compress/recover
    (ratio=0.0, keep all coefficients).

    Note: trained circuit bases are NOT unitary (T^{-1}∘T ≠ identity); the MSE
    at keep_ratio=1.0 is the reconstruction floor of the fitted circuit, not zero.
    The correct check is that our dense top-k path matches pdft.io's sparse path
    when k=total (all coefficients kept).
    """
    import jax
    import pdft.io as pio

    from pdft_benchmarks._loading import load_trained_basis, make_compress_fn

    basis = load_trained_basis(quickdraw_blocked_8_path)
    fn = make_compress_fn(basis)
    rng = np.random.default_rng(0)
    img = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)

    recon = fn(img, keep_ratio=1.0)
    assert recon.shape == img.shape

    # Compare with pdft.io (ratio=0.0 = keep all) — should be identical.
    cpu_basis = jax.tree_util.tree_map(jax.device_get, basis)
    compressed = pio.compress(cpu_basis, img, ratio=0.0)
    recon_io = pio.recover(cpu_basis, compressed)
    np.testing.assert_allclose(recon, recon_io, atol=1e-12)


def test_make_compress_fn_partial_compression(quickdraw_blocked_8_path):
    """At keep_ratio=0.2, the compress fn should produce a real-valued image of the same shape."""
    from pdft_benchmarks._loading import load_trained_basis, make_compress_fn

    basis = load_trained_basis(quickdraw_blocked_8_path)
    fn = make_compress_fn(basis)
    rng = np.random.default_rng(1)
    img = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)
    recon = fn(img, keep_ratio=0.2)
    assert recon.shape == img.shape
    assert np.isrealobj(recon)
    # MSE shouldn't be ridiculous.
    mse = float(np.mean((img - recon) ** 2))
    assert mse < 0.5  # generous bound
