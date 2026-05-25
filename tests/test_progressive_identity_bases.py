"""Unit tests for the progressive-curriculum identity-init basis helpers.

Each family's per-stage init must be the literal identity operator (so all
stages start from the same neutral point), with gates laid out H-first in
the family's storage order.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from pdft_benchmarks.bases import (
    entangled_qft_identity_basis,
    entangled_qft_random_basis,
    mera_identity_basis,
    mera_random_basis,
    qft_identity_basis,
    qft_random_basis,
    real_rich_identity_basis,
    rich_identity_basis,
    tebd_identity_basis,
    tebd_random_basis,
)


def _random_complex_image(n: int, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)),
        dtype=jnp.complex128,
    )


@pytest.mark.parametrize(
    "builder",
    [qft_identity_basis, rich_identity_basis, real_rich_identity_basis],
    ids=["qft", "rich", "real_rich"],
)
@pytest.mark.parametrize("m", [1, 2, 3])
def test_identity_basis_is_identity_operator(builder, m):
    """forward_transform of an identity-init basis is bit-exactly the input."""
    b = builder(m=m, n=m)
    x = _random_complex_image(2**m, seed=m)
    y = b.forward_transform(x)
    max_abs_diff = float(jnp.max(jnp.abs(y - x)))
    assert max_abs_diff == 0.0, (
        f"{builder.__name__}(m={m}) is not the identity operator: "
        f"max|fwd(x)-x| = {max_abs_diff:.3e}"
    )


@pytest.mark.parametrize(
    "builder",
    [rich_identity_basis, real_rich_identity_basis],
    ids=["rich", "real_rich"],
)
def test_circuit_identity_gate_shapes(builder):
    """Rich/RealRich identity bases use H (2,2) and U4 (2,2,2,2) gates only,
    stored Hadamard-first (all 2x2 H slots precede the 2x2x2x2 U4 slots)."""
    b = builder(m=3, n=3)
    shapes = [tuple(t.shape) for t in b.tensors]
    h_slots = [s for s in shapes if s == (2, 2)]
    u4_slots = [s for s in shapes if s == (2, 2, 2, 2)]
    assert len(h_slots) + len(u4_slots) == len(shapes), \
        f"unexpected gate shapes present: {set(shapes)}"
    # H-first: no U4 slot appears before the last H slot.
    last_h = max(i for i, s in enumerate(shapes) if s == (2, 2))
    first_u4 = min(i for i, s in enumerate(shapes) if s == (2, 2, 2, 2))
    assert last_h < first_u4, f"gates not Hadamard-first: shapes={shapes}"
    # Every H slot is I_2 and every U4 slot is I_4.
    eye2 = jnp.eye(2, dtype=jnp.complex128)
    eye4 = jnp.eye(4, dtype=jnp.complex128).reshape(2, 2, 2, 2)
    for t in b.tensors:
        ref = eye2 if t.shape == (2, 2) else eye4
        assert jnp.allclose(t, ref, atol=1e-12), "gate is not at its identity element"


@pytest.mark.parametrize("m", [1, 2, 3])
def test_tebd_random_basis_reproducible_and_distinct(m):
    """tebd_random_basis is seed-deterministic: same seed -> identical gates,
    different seed -> different gates. (TEBD has no identity init; its
    progressive stages use this native seeded brick-wall.)"""
    a = tebd_random_basis(m=m, n=m, seed=7)
    b = tebd_random_basis(m=m, n=m, seed=7)
    c = tebd_random_basis(m=m, n=m, seed=8)
    assert len(a.tensors) == len(b.tensors) == len(c.tensors)
    assert all(jnp.allclose(x, y) for x, y in zip(a.tensors, b.tensors)), \
        "same seed produced different gates"
    assert not all(jnp.allclose(x, y) for x, y in zip(a.tensors, c.tensors)), \
        "different seeds produced identical gates"


def test_tebd_random_basis_is_unitary_transform():
    """The TEBD circuit is a unitary transform: forward_transform preserves
    the L2 norm (Parseval). TEBD's .tensors store a gate parametrization, not
    the literal gate matrices, so unitarity is checked at the transform level."""
    b = tebd_random_basis(m=3, n=3, seed=3)
    x = _random_complex_image(2**3, seed=3)
    y = b.forward_transform(x)
    nx = float(jnp.linalg.norm(x))
    ny = float(jnp.linalg.norm(y))
    assert abs(nx - ny) < 1e-9, (
        f"TEBD forward_transform is not norm-preserving: |x|={nx:.6f} |y|={ny:.6f}"
    )


# --- phase-family (qft/entangled_qft/tebd/mera) value-based identity init ---
# These detect gate kind by value (H -> I_2, CP -> phase 0) rather than the
# rich/real_rich shape-based scheme, so identity holds to float tolerance
# (~1e-16) rather than bit-exactly.
@pytest.mark.parametrize(
    "builder, m",
    [
        (entangled_qft_identity_basis, 2),
        (entangled_qft_identity_basis, 3),
        (tebd_identity_basis, 2),
        (tebd_identity_basis, 3),
        (mera_identity_basis, 2),   # mera: m must be a power of 2
        (mera_identity_basis, 4),
    ],
)
def test_phase_family_identity_basis_is_identity_operator(builder, m):
    b = builder(m=m, n=m)
    x = _random_complex_image(2**m, seed=m)
    y = b.forward_transform(x)
    max_abs_diff = float(jnp.max(jnp.abs(y - x)))
    assert max_abs_diff < 1e-9, (
        f"{builder.__name__}(m={m}) is not the identity operator: "
        f"max|fwd(x)-x| = {max_abs_diff:.3e}"
    )


@pytest.mark.parametrize(
    "builder, m",
    [
        (qft_random_basis, 2),
        (qft_random_basis, 3),
        (entangled_qft_random_basis, 3),
        (mera_random_basis, 2),
        (mera_random_basis, 4),
    ],
)
def test_random_basis_unitary_reproducible_distinct(builder, m):
    """Each random-init builder is norm-preserving (unitary transform),
    seed-deterministic, and seed-distinct."""
    a = builder(m=m, n=m, seed=11)
    b = builder(m=m, n=m, seed=11)
    c = builder(m=m, n=m, seed=12)
    x = _random_complex_image(2**m, seed=5)
    nx = float(jnp.linalg.norm(x))
    ny = float(jnp.linalg.norm(a.forward_transform(x)))
    assert abs(nx - ny) < 1e-9, f"{builder.__name__} not norm-preserving"
    assert all(jnp.allclose(jnp.asarray(p), jnp.asarray(q))
               for p, q in zip(a.tensors, b.tensors)), "same seed differs"
    assert not all(jnp.allclose(jnp.asarray(p), jnp.asarray(q))
                   for p, q in zip(a.tensors, c.tensors)), "seeds not distinct"
