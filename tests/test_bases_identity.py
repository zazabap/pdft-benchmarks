"""Tests for identity_basis_for: every registered basis topology pinned at
its identity element so forward_transform(x) == x.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from pdft_benchmarks.bases import identity_basis_for


# Unblocked bases: at m=n=4 to keep tests fast on CPU. m+n=8 is a power of 2
# so MERA is included.
UNBLOCKED = ("qft", "entangled_qft", "tebd", "mera")
# Blocked bases: only valid at outer m,n >= inner_m=3 + block_log >= 0; use m=n=8.
BLOCKED = ("blocked_8", "rich_8", "real_rich_8")


@pytest.fixture
def x_4x4():
    """Random 16x16 complex image for the m=n=4 unblocked test."""
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((16, 16)).astype(np.float64)
    return jnp.asarray(x_np, dtype=jnp.complex128)


@pytest.fixture
def x_8x8():
    """Random 256x256 complex image for the m=n=8 blocked test."""
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((256, 256)).astype(np.float64)
    return jnp.asarray(x_np, dtype=jnp.complex128)


@pytest.mark.parametrize("name", UNBLOCKED)
def test_identity_basis_unblocked_is_identity(name, x_4x4):
    """forward_transform(x) == x bit-exactly for unblocked identity-init bases."""
    basis = identity_basis_for(name, m=4, n=4)
    y = basis.forward_transform(x_4x4)
    max_err = float(jnp.max(jnp.abs(y - x_4x4)))
    assert max_err < 1e-12, f"{name}: T(x) - x has max-abs {max_err}, expected 0"


@pytest.mark.parametrize("name", BLOCKED)
def test_identity_basis_blocked_is_identity(name, x_8x8):
    """forward_transform(x) == x for BlockedBasis variants with identity inner."""
    basis = identity_basis_for(name, m=8, n=8)
    y = basis.forward_transform(x_8x8)
    max_err = float(jnp.max(jnp.abs(y - x_8x8)))
    assert max_err < 1e-12, f"{name}: T(x) - x has max-abs {max_err}, expected 0"


def test_identity_basis_qft_matches_qft_identity_basis():
    """For QFT, identity_basis_for and qft_identity_basis must produce
    tensors that are equal element-wise (both yield T == identity by
    different code paths)."""
    from pdft_benchmarks.bases import qft_identity_basis

    a = identity_basis_for("qft", m=5, n=5)
    b = qft_identity_basis(m=5, n=5)
    assert len(a.tensors) == len(b.tensors)
    for i, (ta, tb) in enumerate(zip(a.tensors, b.tensors)):
        assert jnp.allclose(ta, tb), f"tensor {i} differs between paths"
