"""Basis registry: name -> factory(m, n, seed) -> AbstractSparseBasis.

Adding a new basis variant = one entry here.

Note: m+n is the total qubit count. MERA requires m+n to be a power of 2;
the pipeline checks this and emits a "skipped: incompatible_qubits" record
rather than raising.

For BlockedBasis variants, the convention here is that the inner basis
operates on a (m_inner, n_inner) sub-block where (m, n) refers to the
OUTER image dimensions and the block size is (2 ** block_log_m,
2 ** block_log_n). The default partition is (m // 2, n // 2) for both
the inner basis and block_log values, which yields square blocks of size
sqrt(image_size). Customize by importing the factories directly and
constructing BlockedBasis with explicit block_log_m / block_log_n.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pdft

BasisFactory = Callable[..., Any]   # (m, n, seed=0) -> AbstractSparseBasis


def _blocked(m: int, n: int, seed: int, inner_cls,
             inner_m: int | None = None, inner_n: int | None = None):
    # Asymmetric split: inner gets the larger half on odd dims so
    # inner_m + block_log_m == m exactly. At even m this matches the
    # previous symmetric split (m//2 + m//2 == m). At odd m (e.g.
    # QuickDraw m=5) it gives inner_m=3 + block_log_m=2 instead of
    # 2+2=4 (which would have lost a qubit).
    #
    # The optional `inner_m` / `inner_n` overrides pin the inner basis (and
    # therefore the block-pixel-size, since block_size = 2**inner_*) to a
    # fixed scale regardless of m. Used by the *_8 variants below to force
    # 8×8 blocks (inner_m=3) at any m≥3, matching block_dct_8 / block_fft_8.
    if inner_m is None:
        inner_m = (m + 1) // 2
    if inner_n is None:
        inner_n = (n + 1) // 2
    block_log_m = m - inner_m
    block_log_n = n - inner_n
    if block_log_m < 0 or block_log_n < 0:
        raise ValueError(
            f"_blocked: inner_m={inner_m}, inner_n={inner_n} larger than "
            f"outer m={m}, n={n}; block_log would be negative"
        )
    inner = inner_cls(m=inner_m, n=inner_n) if seed == 0 else inner_cls(m=inner_m, n=inner_n)
    return pdft.BlockedBasis(inner=inner, block_log_m=block_log_m, block_log_n=block_log_n)


BASIS_FACTORIES: dict[str, BasisFactory] = {
    # Circuit topologies (4): compared against DCT / FFT / blockDCT.
    "qft":           lambda m, n, seed=0: pdft.QFTBasis(m=m, n=n),
    # Identity-initialized QFT: same topology as `qft`, but every gate
    # starts at the QFT-family identity (H -> I_2, CP -> phase 0). Ablation
    # against the analytic-QFT init to test whether the analytic init
    # carries useful inductive bias or whether training reaches the same
    # optimum from a featureless start.
    "qft_identity": lambda m, n, seed=0: qft_identity_basis(m=m, n=n),
    "entangled_qft": lambda m, n, seed=0: pdft.EntangledQFTBasis(m=m, n=n, seed=seed),
    "tebd":          lambda m, n, seed=0: pdft.TEBDBasis(m=m, n=n, seed=seed),
    "mera":          lambda m, n, seed=0: pdft.MERABasis(m=m, n=n, seed=seed),
    # Block topologies (3): default split — inner basis at (m+1)//2.
    # At QuickDraw m=5: inner_m=3 → 8×8 blocks. At DIV2K m=8: inner_m=4 → 16×16 blocks.
    "blocked":       lambda m, n, seed=0: _blocked(m, n, seed, pdft.QFTBasis),
    "rich":          lambda m, n, seed=0: _blocked(m, n, seed, pdft.RichBasis),
    "real_rich":     lambda m, n, seed=0: _blocked(m, n, seed, pdft.RealRichBasis),
    # Block topologies, fixed 8×8 block size (inner_m=inner_n=3) at any m≥3.
    # Apples-to-apples with classical block_dct_8 / block_fft_8 (which also
    # operate on 8×8 patches regardless of image size). At DIV2K m=8: inner
    # basis at m=3 (3 qubits/axis) replicated across a 32×32 grid of 8×8 blocks.
    "blocked_8":     lambda m, n, seed=0: _blocked(m, n, seed, pdft.QFTBasis,       inner_m=3, inner_n=3),
    "rich_8":        lambda m, n, seed=0: _blocked(m, n, seed, pdft.RichBasis,      inner_m=3, inner_n=3),
    "real_rich_8":   lambda m, n, seed=0: _blocked(m, n, seed, pdft.RealRichBasis,  inner_m=3, inner_n=3),
    # Block topologies, fixed 4×4 block size (inner_m=inner_n=2). Sweep
    # extension; spec §6.1.
    "blocked_4":     lambda m, n, seed=0: _blocked(m, n, seed, pdft.QFTBasis,       inner_m=2, inner_n=2),
    "rich_4":        lambda m, n, seed=0: _blocked(m, n, seed, pdft.RichBasis,      inner_m=2, inner_n=2),
    "real_rich_4":   lambda m, n, seed=0: _blocked(m, n, seed, pdft.RealRichBasis,  inner_m=2, inner_n=2),
    # Block topologies, fixed 16×16 block size (inner_m=inner_n=4).
    "blocked_16":    lambda m, n, seed=0: _blocked(m, n, seed, pdft.QFTBasis,       inner_m=4, inner_n=4),
    "rich_16":       lambda m, n, seed=0: _blocked(m, n, seed, pdft.RichBasis,      inner_m=4, inner_n=4),
    "real_rich_16":  lambda m, n, seed=0: _blocked(m, n, seed, pdft.RealRichBasis,  inner_m=4, inner_n=4),
    # Block topologies, fixed 32×32 block size (inner_m=inner_n=5). Only
    # valid at m≥5; in scope for DIV2K m=8. QuickDraw m=5 also satisfies
    # m≥5 (block_log = 0, no wrapping) but the cell is degenerate with
    # the unblocked basis at m=n=5 and is not part of the QuickDraw grid.
    "blocked_32":    lambda m, n, seed=0: _blocked(m, n, seed, pdft.QFTBasis,       inner_m=5, inner_n=5),
    "rich_32":       lambda m, n, seed=0: _blocked(m, n, seed, pdft.RichBasis,      inner_m=5, inner_n=5),
    "real_rich_32":  lambda m, n, seed=0: _blocked(m, n, seed, pdft.RealRichBasis,  inner_m=5, inner_n=5),
}


def qft_identity_basis(m: int, n: int) -> pdft.QFTBasis:
    """Construct a `QFTBasis(m, n)` initialized to the QFT-family identity.

    Each gate is pinned to the identity element of its manifold:
      - Hadamard -> 2x2 identity matrix.
      - Controlled-phase -> `controlled_phase_diag(0.0)` = [[1,1],[1,1]],
        the 2x2-stack representation of the 4x4 identity (phase phi = 0).

    Topology matches the standard QFT decomposition (same gate sequence as
    `pdft.QFTBasis(m, n)`); only the initial tensor values differ. All
    m+n axis-1 + axis-2 gate parameters remain trainable on the U(2)
    Riemannian manifold during `train_basis_batched`.
    """
    import jax.numpy as jnp
    from pdft.bases.circuit.qft import _qft_gates_1d
    from pdft.circuit.builder import controlled_phase_diag

    eye2 = jnp.eye(2, dtype=jnp.complex128)
    cp_identity = controlled_phase_diag(0.0)

    gates_emit = _qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)
    temporal: list = []
    for g in gates_emit:
        if g["kind"] == "H":
            temporal.append(eye2)
        elif g["kind"] == "CP":
            temporal.append(cp_identity)
        else:
            raise AssertionError(f"unexpected QFT gate kind {g['kind']}")

    # QFTBasis stores tensors in Hadamard-first canonical order; stable-sort
    # so within-group emit order is preserved.
    emit_perm = sorted(
        range(len(gates_emit)),
        key=lambda i: gates_emit[i]["kind"] != "H",
    )
    sorted_tensors = [temporal[i] for i in emit_perm]
    return pdft.QFTBasis(m=m, n=n, tensors=sorted_tensors)


def qft_warm_from_trained_blocked(trained_blocked: pdft.BlockedBasis) -> pdft.QFTBasis:
    """Embed a trained `BlockedBasis(QFTBasis(m_i, n_i), block_log_m, block_log_n)`
    into a full `QFTBasis(m, n)` whose initial operator equals the trained blocked
    one bit-exactly.

    Construction:
      - Inner gates (acting only on the inner qubits of each axis) take the
        TRAINED inner tensor values.
      - Every other gate (touching a block-index qubit, or a cross-axis CP) is
        pinned to identity: H -> I_2, controlled-phase -> phase 0
        (= [[1,1],[1,1]]).

    The returned basis exposes all m+n axis-1 + axis-2 gate parameters as
    trainable, so subsequent `train_basis_batched` is free to drift away from
    the blocked configuration. Used by the warm-start experiment to
    demonstrate that the trained blocked optimum is reachable from the larger
    QFT family — and that training from this init does not degrade.
    """
    import jax.numpy as jnp
    import numpy as np
    from pdft.bases.circuit.qft import _qft_gates_1d
    from pdft.circuit.builder import controlled_phase_diag

    inner = trained_blocked.inner
    if not isinstance(inner, pdft.QFTBasis):
        raise TypeError(
            f"qft_warm_from_trained_blocked: inner must be QFTBasis, "
            f"got {type(inner).__name__}"
        )
    m_inner, n_inner = inner.m, inner.n
    m_outer = m_inner + trained_blocked.block_log_m
    n_outer = n_inner + trained_blocked.block_log_n

    # The trained inner.tensors is in QFTBasis canonical order (Hadamards
    # first, then CPs, in gate-emission order within each group). To map a
    # trained inner tensor to its outer-gate position, we recompute that
    # canonicalization on the inner gate sequence.
    inner_gates_emit = (
        _qft_gates_1d(m_inner, offset=0)
        + _qft_gates_1d(n_inner, offset=m_inner)
    )
    if len(inner_gates_emit) != len(inner.tensors):
        raise AssertionError(
            f"inner gate count mismatch: {len(inner_gates_emit)} gates emitted "
            f"vs {len(inner.tensors)} trained tensors in BlockedBasis.inner"
        )
    inner_emit_perm = sorted(
        range(len(inner_gates_emit)),
        key=lambda i: inner_gates_emit[i]["kind"] != "H",
    )
    # Inverse map: emit-order index -> sorted-order index, so we can index
    # inner.tensors (which is in sorted order) by the emit-order position.
    inner_emit_to_sorted = [0] * len(inner_emit_perm)
    for new_idx, orig_idx in enumerate(inner_emit_perm):
        inner_emit_to_sorted[orig_idx] = new_idx
    trained_in_emit_order = [
        inner.tensors[inner_emit_to_sorted[j]] for j in range(len(inner_gates_emit))
    ]

    # Inner-qubit-pattern -> trained tensor lookup (outer 1-indexed qubits).
    # Inner qubits sit at the LOW indices of each outer axis: axis 1 q in
    # [1..m_inner], axis 2 q in [m_outer+1..m_outer+n_inner]. Inner gates'
    # 1-indexed qubits translate via:
    #     q_inner in [1..m_inner]               -> q_outer = q_inner
    #     q_inner in [m_inner+1..m_inner+n_inner] -> q_outer = q_inner + (m_outer - m_inner)
    def _to_outer_q(q_inner_1ix: int) -> int:
        return q_inner_1ix if q_inner_1ix <= m_inner else q_inner_1ix + (m_outer - m_inner)

    inner_lookup: dict[tuple, Any] = {}
    for j, g in enumerate(inner_gates_emit):
        outer_q = tuple(_to_outer_q(q) for q in g["qubits"])
        inner_lookup[(g["kind"], outer_q)] = trained_in_emit_order[j]

    eye2 = jnp.eye(2, dtype=jnp.complex128)
    cp_identity = controlled_phase_diag(0.0)

    def _is_inner_outer_q(q_outer_1ix: int) -> bool:
        if 1 <= q_outer_1ix <= m_outer:
            return q_outer_1ix <= m_inner
        return (q_outer_1ix - m_outer) <= n_inner

    outer_gates_emit = (
        _qft_gates_1d(m_outer, offset=0)
        + _qft_gates_1d(n_outer, offset=m_outer)
    )
    new_temporal: list[Any] = []
    for g in outer_gates_emit:
        if g["kind"] == "H":
            (q,) = g["qubits"]
            if _is_inner_outer_q(q):
                new_temporal.append(inner_lookup[("H", (q,))])
            else:
                new_temporal.append(eye2)
        elif g["kind"] == "CP":
            q_ctrl, q_tgt = g["qubits"]
            if _is_inner_outer_q(q_ctrl) and _is_inner_outer_q(q_tgt):
                new_temporal.append(inner_lookup[("CP", (q_ctrl, q_tgt))])
            else:
                new_temporal.append(cp_identity)
        else:
            raise AssertionError(f"unexpected QFT gate kind {g['kind']}")

    # QFTBasis stores tensors in Hadamard-first canonical order; sort.
    outer_emit_perm = sorted(
        range(len(outer_gates_emit)),
        key=lambda i: outer_gates_emit[i]["kind"] != "H",
    )
    sorted_tensors = [new_temporal[i] for i in outer_emit_perm]
    # Ensure dtypes are consistent (jnp arrays); some trained tensors may
    # have been loaded as numpy arrays.
    sorted_tensors = [jnp.asarray(t, dtype=jnp.complex128) for t in sorted_tensors]
    return pdft.QFTBasis(m=m_outer, n=n_outer, tensors=sorted_tensors)

__all__ = ["BASIS_FACTORIES", "BasisFactory", "qft_identity_basis"]
