"""Block-structure-aware identity-pull regularisation for QFT-family bases.

Implements:
- ``qft_identity_table(m, n)``: per-gate identity-element table in QFTBasis
  canonical order (Hadamards first, then controlled-phase gates), matching
  the tensor layout of ``pdft_benchmarks.bases.qft_identity_basis``.
- ``qft_inner_mask(m, n, inner_m, inner_n)``: per-gate boolean mask, True
  iff gate acts only on inner-block qubits (axis-1 q in [1..inner_m] or
  axis-2 q in [m+1..m+inner_n]).
- ``BlockMaskedIdentityRegQFTMSELoss``: MSELoss subclass that adds a
  block-structure-aware L2 pull toward identity via pdft's `_extra_loss`
  hook. Outer/block-index gates are penalised W times more strongly than
  inner-block gates, matching the structure of `blocked_8`'s optimum.

Used by the qft_identity_regularization experiment to test whether a
block-structure-matched prior closes the gap from qft_identity (~31.66 dB)
to blocked_8 (32.26 dB) on DIV2K-8q.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
from pdft.circuit.builder import controlled_phase_diag
from pdft.loss import MSELoss


def qft_identity_table(m: int, n: int) -> list[jax.Array]:
    """Per-gate identity-element table in QFTBasis canonical order.

    First ``m + n`` entries are 2x2 identity (Hadamard identity element).
    Remaining entries are ``controlled_phase_diag(0.0) = [[1,1],[1,1]]``
    (controlled-phase identity, the 2x2 diagonal-stack representation
    of the 4x4 identity).

    Order matches ``QFTBasis(m, n).tensors`` after Hadamard-first sort,
    which is what ``qft_identity_basis(m, n)`` produces.
    """
    eye2 = jnp.eye(2, dtype=jnp.complex128)
    cp_identity = controlled_phase_diag(0.0)
    n_hadamards = m + n
    n_cps = (m * (m - 1)) // 2 + (n * (n - 1)) // 2
    return [eye2] * n_hadamards + [cp_identity] * n_cps


def qft_inner_mask(m: int, n: int,
                   inner_m: int, inner_n: int) -> list[bool]:
    """Return a per-gate boolean mask in QFTBasis canonical order.

    Entry ``i`` is ``True`` iff gate ``i`` acts ONLY on inner-block
    qubits — i.e. every qubit the gate touches is in the inner range
    for its axis:

    - Axis-1 inner qubits: ``{1, 2, ..., inner_m}`` (1-indexed).
    - Axis-2 inner qubits: ``{m+1, m+2, ..., m+inner_n}`` (1-indexed).

    A gate is "outer" if any qubit it touches is in the block-index
    set. For QFT(m=8, n=8) with inner=(3,3), the mask has 12 True
    (= the inner-block gates of a hypothetical BlockedBasis(QFTBasis(3,3),
    block_log_m=5, block_log_n=5)) and 60 False.

    Order matches ``qft_identity_table(m, n)`` and
    ``QFTBasis(m, n).tensors`` (Hadamard-first canonical sort).

    Reuses the inner/outer classification logic from
    ``pdft_benchmarks.bases.qft_warm_from_trained_blocked``.
    """
    from pdft.bases.circuit.qft import _qft_gates_1d

    def _is_inner_q(q_1ix: int) -> bool:
        if 1 <= q_1ix <= m:
            return q_1ix <= inner_m
        return (q_1ix - m) <= inner_n

    gates_emit = _qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)
    perm = sorted(range(len(gates_emit)),
                  key=lambda i: gates_emit[i]["kind"] != "H")
    sorted_gates = [gates_emit[i] for i in perm]
    return [all(_is_inner_q(q) for q in g["qubits"]) for g in sorted_gates]


@dataclass(frozen=True)
class BlockMaskedIdentityRegQFTMSELoss(MSELoss):
    """MSELoss + block-structure-aware L2 pull toward QFT identity elements.

    Loss = sum_pixel |x - T_dagger(topk(T(x), k))|^2
         + lam * R_block(theta)

    where the block-structure-aware reg is

        R_block(theta) = sum_{g in outer} outer_weight * ||T_g - I_g||_F^2
                       + sum_{g in inner}                ||T_g - I_g||_F^2

    The inner/outer mask is determined by (inner_m, inner_n) via
    qft_inner_mask: a gate is inner iff every qubit it touches is an
    inner-block qubit (axis-1 q in [1..inner_m] or axis-2 q in
    [m+1..m+inner_n]).

    At blocked_8's optimum (m=n=8, inner=(3,3)): the 60 outer gates are
    bit-exactly at identity and contribute 0 to R_block; the 12 inner
    gates take their trained-QFT(3,3) values and contribute the inner
    L2 term. Designed so that lam can grow without pushing the
    inner-block gates back toward identity.
    """
    lam: float = 0.0
    m: int = 0
    n: int = 0
    inner_m: int = 3
    inner_n: int = 3
    outer_weight: float = 10.0

    def _extra_loss(self, tensors: Sequence[jax.Array]) -> jax.Array:
        if self.lam == 0.0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        identities = qft_identity_table(self.m, self.n)
        is_inner = qft_inner_mask(self.m, self.n,
                                   self.inner_m, self.inner_n)
        if len(identities) != len(tensors):
            raise ValueError(
                f"BlockMaskedIdentityRegQFTMSELoss: gate count mismatch — "
                f"m={self.m}, n={self.n} → {len(identities)} entries, "
                f"but tensors has {len(tensors)} elements. "
                f"Did you pass the wrong (m, n)?"
            )
        total = jnp.asarray(0.0, dtype=jnp.float64)
        W = jnp.asarray(self.outer_weight, dtype=jnp.float64)
        for t, i, inner in zip(tensors, identities, is_inner):
            sq = jnp.sum(jnp.abs(t - i) ** 2)
            total = total + (sq if inner else W * sq)
        return self.lam * total


__all__ = ["qft_identity_table", "qft_inner_mask",
           "BlockMaskedIdentityRegQFTMSELoss"]
