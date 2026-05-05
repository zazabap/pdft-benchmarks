"""Layer A: pca.py unit tests.

The fit functions run SVD on the JAX default device (GPU when available,
CPU otherwise), then host-roundtrip the result. Tests work in both
environments — the conftest's `import pdft` enables x64 mode before any
test runs. Compress/recover are pure numpy.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest


def test_pca_basis_dataclass_frozen():
    from pdft_benchmarks.pca import PcaBasis

    basis = PcaBasis(
        eigenbasis=np.eye(4),
        mean=np.zeros(4),
        eigenvalues=np.ones(4),
        n_samples_fit=10,
        d=4,
        block=None,
    )
    with pytest.raises(FrozenInstanceError):
        basis.eigenbasis = np.zeros((4, 4))  # type: ignore[misc]


def test_block_pca_full_keep_is_identity_when_full_rank():
    """Full-rank fit + keep_ratio=1.0 recovers the input exactly."""
    from pdft_benchmarks.pca import fit_block_pca, pca_compress, pca_recover

    rng = np.random.default_rng(0)
    # 100 images of shape (32, 32) → 100 * 16 = 1600 patches in 64-dim → full rank.
    train = rng.uniform(0.0, 1.0, size=(100, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)

    test = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)
    coefs = pca_compress(basis, test, keep_ratio=1.0)
    recovered = pca_recover(basis, coefs)
    np.testing.assert_allclose(recovered, test, atol=1e-10)


def test_block_pca_eigenbasis_orthonormal():
    from pdft_benchmarks.pca import fit_block_pca

    rng = np.random.default_rng(1)
    train = rng.uniform(0.0, 1.0, size=(100, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    k = basis.eigenbasis.shape[0]
    np.testing.assert_allclose(basis.eigenbasis @ basis.eigenbasis.T, np.eye(k), atol=1e-10)


def test_block_pca_eigenvalues_descending_and_nonneg():
    from pdft_benchmarks.pca import fit_block_pca

    rng = np.random.default_rng(2)
    train = rng.uniform(0.0, 1.0, size=(100, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    assert np.all(basis.eigenvalues >= 0.0)
    assert np.all(np.diff(basis.eigenvalues) <= 1e-12)


def test_block_pca_sign_canonical_repeatable():
    """Two fits on the same data give bit-identical eigenbasis (sign-stable)."""
    from pdft_benchmarks.pca import fit_block_pca

    rng = np.random.default_rng(3)
    train = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    a = fit_block_pca(train, block=8)
    b = fit_block_pca(train, block=8)
    np.testing.assert_array_equal(a.eigenbasis, b.eigenbasis)
    for row in a.eigenbasis:
        argmax = int(np.argmax(np.abs(row)))
        assert row[argmax] > 0


def test_block_pca_keep_ratio_global_count():
    """At kr=0.5, exactly floor(0.5 * total) coefficients are non-zero."""
    from pdft_benchmarks.pca import fit_block_pca, pca_compress

    rng = np.random.default_rng(4)
    train = rng.uniform(0.0, 1.0, size=(100, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    test = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)
    coefs = pca_compress(basis, test, keep_ratio=0.5)
    nonzero = int(np.sum(coefs != 0))
    expected = int(np.floor(0.5 * coefs.size))
    assert nonzero == expected, f"expected {expected} non-zero coefs, got {nonzero}"


def test_block_pca_zero_keep_ratio_keeps_one():
    """keep_ratio that floors to zero is coerced to 1 (largest single coefficient)."""
    from pdft_benchmarks.pca import fit_block_pca, pca_compress

    rng = np.random.default_rng(5)
    train = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    test = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)
    coefs = pca_compress(basis, test, keep_ratio=0.0)
    assert int(np.sum(coefs != 0)) == 1


def test_fit_block_pca_empty_train_raises():
    from pdft_benchmarks.pca import fit_block_pca

    with pytest.raises(ValueError, match="at least 1 training image"):
        fit_block_pca([])


def test_fit_block_pca_non_divisible_raises():
    from pdft_benchmarks.pca import fit_block_pca

    bad = np.zeros((10, 10))
    with pytest.raises(ValueError, match="block size"):
        fit_block_pca([bad], block=8)


def test_fit_block_pca_inhomogeneous_shapes_ok():
    """Block PCA pools patches; per-image H,W can vary as long as each individually divides by block."""
    from pdft_benchmarks.pca import fit_block_pca

    rng = np.random.default_rng(6)
    a = rng.uniform(0.0, 1.0, size=(32, 32))
    b = rng.uniform(0.0, 1.0, size=(64, 32))
    c = rng.uniform(0.0, 1.0, size=(16, 24))
    basis = fit_block_pca([a, b, c], block=8)
    assert basis.eigenbasis.shape[1] == 64
    # 16 + 32 + 6 = 54 patches.
    assert basis.n_samples_fit == 16 + 32 + 6


def test_pca_compress_block_size_check():
    """Compress-time block-divisibility check."""
    from pdft_benchmarks.pca import fit_block_pca, pca_compress

    rng = np.random.default_rng(7)
    train = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    bad = rng.uniform(0.0, 1.0, size=(10, 10))
    with pytest.raises(ValueError, match="block size"):
        pca_compress(basis, bad, keep_ratio=0.5)


def test_global_pca_full_keep_is_identity_when_full_rank():
    """n=128 train samples in 64-dim ambient (8x8 images) → fully covers the space at kr=1.0."""
    from pdft_benchmarks.pca import fit_global_pca, pca_compress, pca_recover

    rng = np.random.default_rng(10)
    train = rng.uniform(0.0, 1.0, size=(128, 8, 8)).astype(np.float64)
    basis = fit_global_pca(train)
    assert basis.block is None
    assert basis.d == 64

    test = rng.uniform(0.0, 1.0, size=(8, 8)).astype(np.float64)
    coefs = pca_compress(basis, test, keep_ratio=1.0)
    recovered = pca_recover(basis, coefs)
    np.testing.assert_allclose(recovered, test, atol=1e-10)


def test_fit_global_pca_inhomogeneous_shapes_raises():
    from pdft_benchmarks.pca import fit_global_pca

    rng = np.random.default_rng(11)
    a = rng.uniform(0.0, 1.0, size=(8, 8))
    b = rng.uniform(0.0, 1.0, size=(16, 16))
    with pytest.raises(ValueError, match="identical shape"):
        fit_global_pca([a, b])


def test_global_pca_compress_shape_mismatch_raises():
    from pdft_benchmarks.pca import fit_global_pca, pca_compress

    rng = np.random.default_rng(12)
    train = rng.uniform(0.0, 1.0, size=(20, 8, 8)).astype(np.float64)
    basis = fit_global_pca(train)
    wrong = rng.uniform(0.0, 1.0, size=(16, 16))
    with pytest.raises(ValueError, match="fit on shape"):
        pca_compress(basis, wrong, keep_ratio=0.5)


def test_global_pca_rank_deficient_full_keep_is_rank_k_projection():
    """n=4 train samples in 256-dim ambient → rank-4 fit; kr=1.0 returns rank-4 projection."""
    from pdft_benchmarks.pca import fit_global_pca, pca_compress, pca_recover

    rng = np.random.default_rng(13)
    train = rng.uniform(0.0, 1.0, size=(4, 16, 16)).astype(np.float64)
    basis = fit_global_pca(train)
    assert basis.eigenbasis.shape[0] <= 4

    test = rng.uniform(0.0, 1.0, size=(16, 16)).astype(np.float64)
    coefs = pca_compress(basis, test, keep_ratio=1.0)
    recovered = pca_recover(basis, coefs)
    assert np.linalg.norm(recovered - test) > 0.1
    flat = test.ravel() - basis.mean
    projected = (flat @ basis.eigenbasis.T) @ basis.eigenbasis + basis.mean
    np.testing.assert_allclose(recovered.ravel(), projected, atol=1e-10)


def test_global_pca_rank_deficient_compress_at_high_kr():
    """When `keep` would exceed available rank `k`, all k coefs are kept; no error."""
    from pdft_benchmarks.pca import fit_global_pca, pca_compress

    rng = np.random.default_rng(14)
    train = rng.uniform(0.0, 1.0, size=(4, 16, 16)).astype(np.float64)
    basis = fit_global_pca(train)
    test = rng.uniform(0.0, 1.0, size=(16, 16)).astype(np.float64)
    coefs = pca_compress(basis, test, keep_ratio=0.5)
    nonzero = int(np.sum(coefs != 0))
    assert nonzero == basis.eigenbasis.shape[0]


def test_fingerprint_deterministic():
    from pdft_benchmarks.pca import fit_block_pca, fingerprint

    rng = np.random.default_rng(20)
    train = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    a = fingerprint(fit_block_pca(train, block=8))
    b = fingerprint(fit_block_pca(train, block=8))
    assert a["spectrum_sha256"] == b["spectrum_sha256"]
    assert a["k_effective"] == b["k_effective"]
    assert a["n_samples_fit"] == b["n_samples_fit"]


def test_fingerprint_changes_with_data():
    from pdft_benchmarks.pca import fit_block_pca, fingerprint

    rng = np.random.default_rng(21)
    train_a = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    train_b = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    a = fingerprint(fit_block_pca(train_a, block=8))
    b = fingerprint(fit_block_pca(train_b, block=8))
    assert a["spectrum_sha256"] != b["spectrum_sha256"]


def test_fingerprint_fields_complete():
    from pdft_benchmarks.pca import fit_block_pca, fingerprint

    rng = np.random.default_rng(22)
    train = rng.uniform(0.0, 1.0, size=(50, 32, 32)).astype(np.float64)
    fp = fingerprint(fit_block_pca(train, block=8))
    expected_keys = {
        "n_samples_fit", "d", "k_effective", "block", "mean_norm",
        "eigenvalue_top10", "eigenvalue_sum", "spectrum_sha256",
    }
    assert set(fp.keys()) == expected_keys
    assert fp["d"] == 64
    assert fp["block"] == 8
    assert isinstance(fp["spectrum_sha256"], str) and len(fp["spectrum_sha256"]) == 64
    assert len(fp["eigenvalue_top10"]) == 10


def test_pca_compress_rank_block_per_block_count():
    """Block PCA-rank keeps exactly floor(64*kr) coefs PER BLOCK (uniform budget)."""
    from pdft_benchmarks.pca import fit_block_pca, pca_compress_rank

    rng = np.random.default_rng(40)
    train = rng.uniform(0.0, 1.0, size=(100, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    test = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)
    coefs = pca_compress_rank(basis, test, keep_ratio=0.25)
    keep_per_block = int(np.floor(0.25 * 64))  # 16
    n_blocks = (32 // 8) ** 2
    assert coefs.shape == (n_blocks, 64)
    # Every block keeps exactly `keep_per_block` non-zero coefs (positions 0..k-1).
    nonzero_per_block = np.sum(coefs != 0, axis=1)
    assert np.all(nonzero_per_block == keep_per_block)
    # Specifically positions 0..15 are kept; 16..63 zeroed.
    assert np.all(coefs[:, keep_per_block:] == 0)


def test_pca_compress_rank_full_keep_is_identity_when_full_rank():
    """At keep_ratio=1.0 with full-rank fit, rank-truncation recovers exactly."""
    from pdft_benchmarks.pca import fit_block_pca, pca_compress_rank, pca_recover

    rng = np.random.default_rng(41)
    train = rng.uniform(0.0, 1.0, size=(100, 32, 32)).astype(np.float64)
    basis = fit_block_pca(train, block=8)
    test = rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)
    rec = pca_recover(basis, pca_compress_rank(basis, test, keep_ratio=1.0))
    np.testing.assert_allclose(rec, test, atol=1e-10)


def test_global_pca_rank_caps_at_k_effective():
    """Rank-deficient global PCA-rank: keep saturates at k_effective."""
    from pdft_benchmarks.pca import fit_global_pca, pca_compress_rank

    rng = np.random.default_rng(42)
    train = rng.uniform(0.0, 1.0, size=(4, 16, 16)).astype(np.float64)
    basis = fit_global_pca(train)  # k_effective <= 4
    k_eff = basis.eigenbasis.shape[0]
    test = rng.uniform(0.0, 1.0, size=(16, 16)).astype(np.float64)
    coefs = pca_compress_rank(basis, test, keep_ratio=1.0)  # wants 256, has <=4
    assert int(np.sum(coefs != 0)) == k_eff


def test_block_pca_top_eigenvector_is_dc_for_smooth_images():
    """For natural-image-like (smooth gradient + small noise) corpus, the top
    block eigenvector should be approximately the DC vector ones/sqrt(64)."""
    from pdft_benchmarks.pca import fit_block_pca

    rng = np.random.default_rng(30)
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, 32), np.linspace(0.0, 1.0, 32), indexing="ij")
    base = 0.5 + 0.3 * (xx + yy) / 2.0
    train = np.stack([base + 0.05 * rng.standard_normal((32, 32)) for _ in range(50)], axis=0)
    basis = fit_block_pca(train, block=8)
    dc = np.ones(64) / np.sqrt(64)
    inner = abs(float(basis.eigenbasis[0] @ dc))
    assert inner > 0.95, f"top eigenvector ⟨·, DC⟩ = {inner:.3f}; expected > 0.95"
