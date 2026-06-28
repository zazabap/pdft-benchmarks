"""DCT-IV experiment-suite infra: identity/random init, unfreeze orders,
inner/outer partition, and the load-bearing warm-start bit-exactness.

These guard the helpers the dct4_* drivers stand on. The warm-start check is the
one that can silently regress: DCT-IV emits two identical mirror-Q CNOTs and an
R_y + branch-H on the same qubit per level, so the inner-level gates must be
paired POSITIONALLY (a (kind, qubits) dict would collide). If that pairing
breaks, the warm operator drifts toward identity instead of staying bit-exact.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pdft

from pdft_benchmarks.bases import (
    _dct4_random_basis,
    dct4_identity_basis,
    dct4_inner_outer_indices,
    dct4_warm_from_trained_blocked,
    family_random_basis,
)
from pdft_benchmarks.unfreeze import dct4_unfreeze_orders, train_progressive_unfreeze


def _rnd_real(shape, seed):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape),
                       dtype=jnp.complex128)


def test_dct4_identity_basis_is_identity_op():
    b = dct4_identity_basis(3, 3)
    x = _rnd_real((8, 8), 1)
    err = float(jnp.max(jnp.abs(b.forward_transform(x) - x)))
    assert err < 1e-12, f"identity basis not a no-op: max|fwd(x)-x|={err:.2e}"


def test_dct4_random_basis_is_real_and_orthogonal():
    b = family_random_basis("dct4", 3, 3, 5)
    imag = max(float(jnp.max(jnp.abs(jnp.imag(t)))) for t in b.tensors)
    assert imag < 1e-12, f"random init not real: max|imag|={imag:.2e}"
    x = _rnd_real((8, 8), 2)
    rt = float(jnp.max(jnp.abs(b.inverse_transform(b.forward_transform(x)) - x)))
    assert rt < 1e-10, f"random init not orthogonal: round-trip err={rt:.2e}"


def test_dct4_unfreeze_orders_are_permutations():
    for m in (2, 3, 4):
        G = len(pdft.DCT4Basis(m=m, n=m).tensors)
        orders = dct4_unfreeze_orders(m, m)
        assert set(orders.keys()) == {"bg", "lr", "rl"}
        for name, seq in orders.items():
            assert sorted(seq) == list(range(G)), f"{name} not a permutation at m={m}"
        assert orders["rl"] == list(reversed(orders["lr"]))


def test_dct4_bg_prefix_completes_inner_blocks():
    # Unfreezing block-growth stages 1..k must complete the blocked DCT-IV(k, k):
    # the bg prefix of length |inner(k,k)| is exactly the inner-block index set.
    orders = dct4_unfreeze_orders(4, 4)
    for k in (1, 2, 3):
        inner_k, _ = dct4_inner_outer_indices(4, 4, k, k)
        assert set(orders["bg"][:len(inner_k)]) == set(inner_k), \
            f"bg prefix != inner({k},{k})"


def test_dct4_inner_outer_partition():
    # Complete & disjoint, and the m=n=8 inner-3 partition is 34 inner / 180 outer.
    inner, outer = dct4_inner_outer_indices(8, 8, 3, 3)
    G = len(pdft.DCT4Basis(m=8, n=8).tensors)
    assert sorted(inner + outer) == list(range(G))
    assert not (set(inner) & set(outer))
    assert len(inner) == 34 and len(outer) == 180, \
        f"{len(inner)} inner + {len(outer)} outer (expected 34 + 180 = {G})"


def test_dct4_warm_from_trained_blocked_is_bit_exact():
    # THE critical check: a full DCT4Basis warm-started from a trained
    # BlockedBasis(DCT4(2,2), 2, 2) must reproduce the blocked operator exactly.
    inner = _dct4_random_basis(2, 2, 7)
    blocked = pdft.BlockedBasis(inner=inner, block_log_m=2, block_log_n=2)
    warm = dct4_warm_from_trained_blocked(blocked)
    assert warm.m == 4 and warm.n == 4
    x = _rnd_real((16, 16), 3)
    err = float(jnp.max(jnp.abs(blocked.forward_transform(x) - warm.forward_transform(x))))
    assert err < 1e-10, f"warm-start not bit-exact vs blocked: max|diff|={err:.2e}"


def test_dct4_progressive_unfreeze_runs():
    # Smoke: the unfreeze loop runs end-to-end on a DCT4Basis and returns one.
    ds = [np.asarray(_rnd_real((8, 8), s)) for s in range(4)]
    res = train_progressive_unfreeze(
        family_random_basis("dct4", 3, 3, 1), ds,
        unfreeze_order=dct4_unfreeze_orders(3, 3)["bg"][:2], lr=0.01,
        max_steps_per_stage=3, loss=pdft.MSELoss(k=10), min_steps_per_stage=1)
    assert isinstance(res.basis, pdft.DCT4Basis)
    assert len(res.stages) == 2
