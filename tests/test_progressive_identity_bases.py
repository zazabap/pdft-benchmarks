"""Unit tests for the progressive-curriculum family init helpers.

Each family's per-stage identity init must be the literal identity operator (so
all stages start from the same neutral point); the random inits must be
norm-preserving, seed-deterministic, and seed-distinct. Both are exercised
through the generic dispatchers `family_identity_basis` / `family_random_basis`.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from pdft_benchmarks.bases import family_identity_basis, family_random_basis


def _random_complex_image(n: int, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)),
        dtype=jnp.complex128,
    )


# --- identity init: literal identity operator ------------------------------
# rich/real_rich use shape-based identity (bit-exact); qft uses value-based
# identity that is also bit-exact here; the other phase families hold to
# float tolerance (~1e-16).
@pytest.mark.parametrize("family", ["qft", "rich", "real_rich"])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_identity_basis_is_identity_operator(family, m):
    b = family_identity_basis(family, m=m, n=m)
    x = _random_complex_image(2**m, seed=m)
    max_abs_diff = float(jnp.max(jnp.abs(b.forward_transform(x) - x)))
    assert max_abs_diff == 0.0, (
        f"family_identity_basis({family!r}, m={m}) is not the identity operator: "
        f"max|fwd(x)-x| = {max_abs_diff:.3e}"
    )


@pytest.mark.parametrize(
    "family, m",
    [
        ("entangled_qft", 2), ("entangled_qft", 3),
        ("tebd", 2), ("tebd", 3),
        ("mera", 2), ("mera", 4),  # mera: m must be a power of 2
    ],
)
def test_phase_family_identity_basis_is_identity_operator(family, m):
    b = family_identity_basis(family, m=m, n=m)
    x = _random_complex_image(2**m, seed=m)
    max_abs_diff = float(jnp.max(jnp.abs(b.forward_transform(x) - x)))
    assert max_abs_diff < 1e-9, (
        f"family_identity_basis({family!r}, m={m}) is not the identity operator: "
        f"max|fwd(x)-x| = {max_abs_diff:.3e}"
    )


@pytest.mark.parametrize("family", ["rich", "real_rich"])
def test_circuit_identity_gate_shapes(family):
    """Rich/RealRich identity bases use H (2,2) and U4 (2,2,2,2) gates only,
    Hadamard-first, each at its identity element."""
    b = family_identity_basis(family, m=3, n=3)
    shapes = [tuple(t.shape) for t in b.tensors]
    assert all(s in {(2, 2), (2, 2, 2, 2)} for s in shapes), \
        f"unexpected gate shapes present: {set(shapes)}"
    last_h = max(i for i, s in enumerate(shapes) if s == (2, 2))
    first_u4 = min(i for i, s in enumerate(shapes) if s == (2, 2, 2, 2))
    assert last_h < first_u4, f"gates not Hadamard-first: shapes={shapes}"
    eye2 = jnp.eye(2, dtype=jnp.complex128)
    eye4 = jnp.eye(4, dtype=jnp.complex128).reshape(2, 2, 2, 2)
    for t in b.tensors:
        ref = eye2 if t.shape == (2, 2) else eye4
        assert jnp.allclose(t, ref, atol=1e-12), "gate is not at its identity element"


# --- random init: unitary, reproducible, seed-distinct ---------------------
@pytest.mark.parametrize(
    "family, m",
    [
        ("qft", 2), ("qft", 3),
        ("rich", 2), ("real_rich", 2),
        ("tebd", 2), ("tebd", 3),
        ("entangled_qft", 3),
        ("mera", 2), ("mera", 4),
    ],
)
def test_random_basis_unitary_reproducible_distinct(family, m):
    a = family_random_basis(family, m=m, n=m, seed=11)
    b = family_random_basis(family, m=m, n=m, seed=11)
    c = family_random_basis(family, m=m, n=m, seed=12)
    x = _random_complex_image(2**m, seed=5)
    nx = float(jnp.linalg.norm(x))
    ny = float(jnp.linalg.norm(a.forward_transform(x)))
    assert abs(nx - ny) < 1e-9, f"family_random_basis({family!r}) not norm-preserving"
    assert all(jnp.allclose(jnp.asarray(p), jnp.asarray(q))
               for p, q in zip(a.tensors, b.tensors)), "same seed differs"
    assert not all(jnp.allclose(jnp.asarray(p), jnp.asarray(q))
                   for p, q in zip(a.tensors, c.tensors)), "seeds not distinct"
