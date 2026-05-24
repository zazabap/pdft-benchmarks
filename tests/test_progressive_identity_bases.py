"""Unit tests for the progressive-curriculum identity-init basis helpers.

Each family's per-stage init must be the literal identity operator (so all
stages start from the same neutral point), with gates laid out H-first in
the family's storage order.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from pdft_benchmarks.bases import (
    qft_identity_basis,
    real_rich_identity_basis,
    rich_identity_basis,
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
