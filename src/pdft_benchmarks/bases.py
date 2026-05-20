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


def _identity_tensor_for(t) -> "jax.Array":
    """Return the identity element of the slot a canonical-init tensor occupies.

    Classification rule (verified for QFT, EntangledQFT, TEBD, MERA, RichBasis,
    RealRichBasis at canonical init):
      - shape (2, 2) and approximately equal to the constant Hadamard
        (1/sqrt(2)) * [[1, 1], [1, -1]]: this is an H slot -> identity = I_2.
      - shape (2, 2) and NOT the Hadamard: this is a controlled-phase slot in
        the 2x2 diag-stack form -> identity = [[1, 1], [1, 1]] (= phase 0).
      - shape (2, 2, 2, 2): this is a U(4) slot (Rich/RealRich) -> identity =
        the 4x4 identity reshaped to (2, 2, 2, 2).

    Used by ``identity_basis_for`` to construct T = identity init for any
    registered basis topology.
    """
    import jax.numpy as jnp
    import numpy as np

    HADAMARD = (1.0 / np.sqrt(2.0)) * np.array(
        [[1, 1], [1, -1]], dtype=np.complex128
    )
    ta = np.asarray(t)
    if ta.shape == (2, 2):
        if np.allclose(ta, HADAMARD, atol=1e-9):
            return jnp.eye(2, dtype=jnp.complex128)
        return jnp.array([[1, 1], [1, 1]], dtype=jnp.complex128)
    if ta.shape == (2, 2, 2, 2):
        return jnp.eye(4, dtype=jnp.complex128).reshape(2, 2, 2, 2)
    raise ValueError(
        f"_identity_tensor_for: unexpected tensor shape {ta.shape}; "
        f"expected (2,2) or (2,2,2,2)"
    )


def identity_basis_for(name: str, m: int, n: int):
    """Construct the named basis with every gate pinned to its identity element.

    Generalises ``qft_identity_basis`` to all 7 registered topologies. Each
    slot's identity element is detected from the canonical-init tensor's
    shape and value via ``_identity_tensor_for``:

      - H gate -> ``I_2``
      - CP gate -> ``[[1, 1], [1, 1]]`` (phase 0, the 2x2 diag-stack form of
        the 4x4 identity)
      - U(4) gate (Rich/RealRich) -> ``eye(4).reshape(2, 2, 2, 2)``

    Result: ``basis.forward_transform(x) == x`` for any x, regardless of
    topology. Used by the L1-init-anchor sweep to isolate the topology
    effect from the analytic-init effect: all 7 bases start from the same
    T=identity operator, and L1 anchors at identity for all.

    For BlockedBasis variants (blocked_8, rich_8, real_rich_8), only the
    inner basis is identity-initialised; the block-tiling is implicit and
    not parameterised.
    """
    factory = BASIS_FACTORIES[name]
    canonical = factory(m=m, n=n, seed=0)

    if isinstance(canonical, pdft.BlockedBasis):
        inner = canonical.inner
        inner_cls = type(inner)
        new_inner_tensors = [_identity_tensor_for(t) for t in inner.tensors]
        new_inner = inner_cls(m=inner.m, n=inner.n, tensors=new_inner_tensors)
        return pdft.BlockedBasis(
            inner=new_inner,
            block_log_m=canonical.block_log_m,
            block_log_n=canonical.block_log_n,
        )

    cls = type(canonical)
    new_tensors = [_identity_tensor_for(t) for t in canonical.tensors]
    return cls(m=m, n=n, tensors=new_tensors)


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


def qft_warm_from_smaller_qft(
    trained_smaller: "pdft.QFTBasis",
) -> "pdft.QFTBasis":
    """Embed a trained `QFTBasis(k, k)` into `QFTBasis(k+1, k+1)` with the
    newly introduced gates at their identity element.

    Used by the qft_progressive curriculum: stage k+1's init carries forward
    the trained gates of stage k, with the gates that touch the new
    (k+1)-th qubit per axis pinned at identity. The induced operator on
    the inner (k+1)-qubit space is QFT(k) ⊗ I_2; when wrapped in
    BlockedBasis(..., 8-k-1, 8-k-1) at stage k+1, the global image operator
    is bit-exactly identical to stage k's operator. The operator-preservation
    property (QFT(k) ⊗ I_2) is verified by
    test_operator_preservation_at_stage_boundary_k2_to_k3 (added in Task 2).

    Construction:
      - For each gate in QFT(k+1, k+1)'s emission order:
        * If the gate's qubits are a subset of QFT(k, k)'s inner qubit set
          (axis-1: {1..k}; axis-2: {k+2..2k+1} after the axis-2 offset shifts
          from k to k+1), copy the trained tensor from `trained_smaller`.
        * Else (the gate touches the newly introduced qubit per axis: axis-1
          qubit k+1, axis-2 qubit 2k+2), set to H -> I_2; CP -> phase 0.
      - Hadamard-first canonical sort to match QFTBasis storage convention.
    """
    import jax.numpy as jnp
    from pdft.bases.circuit.qft import _qft_gates_1d
    from pdft.circuit.builder import controlled_phase_diag

    if trained_smaller.m != trained_smaller.n:
        raise ValueError(
            f"qft_warm_from_smaller_qft: requires m == n, "
            f"got m={trained_smaller.m}, n={trained_smaller.n}"
        )
    k = trained_smaller.m
    new_k = k + 1

    smaller_gates_emit = (
        _qft_gates_1d(k, offset=0) + _qft_gates_1d(k, offset=k)
    )
    if len(smaller_gates_emit) != len(trained_smaller.tensors):
        raise AssertionError(
            f"smaller gate count mismatch: {len(smaller_gates_emit)} emitted "
            f"vs {len(trained_smaller.tensors)} stored tensors"
        )
    smaller_emit_perm = sorted(
        range(len(smaller_gates_emit)),
        key=lambda i: smaller_gates_emit[i]["kind"] != "H",
    )
    smaller_emit_to_sorted = [0] * len(smaller_emit_perm)
    for sorted_idx, emit_idx in enumerate(smaller_emit_perm):
        smaller_emit_to_sorted[emit_idx] = sorted_idx
    smaller_in_emit_order = [
        trained_smaller.tensors[smaller_emit_to_sorted[j]]
        for j in range(len(smaller_gates_emit))
    ]

    def _smaller_q_to_larger_q(q_smaller_1ix: int) -> int:
        return q_smaller_1ix if q_smaller_1ix <= k else q_smaller_1ix + 1

    smaller_lookup: dict = {}
    for j, g in enumerate(smaller_gates_emit):
        larger_qs = tuple(_smaller_q_to_larger_q(q) for q in g["qubits"])
        smaller_lookup[(g["kind"], larger_qs)] = smaller_in_emit_order[j]

    larger_gates_emit = (
        _qft_gates_1d(new_k, offset=0) + _qft_gates_1d(new_k, offset=new_k)
    )

    eye2 = jnp.eye(2, dtype=jnp.complex128)
    cp_identity = controlled_phase_diag(0.0)

    new_temporal: list = []
    for g in larger_gates_emit:
        key = (g["kind"], g["qubits"])
        if key in smaller_lookup:
            new_temporal.append(smaller_lookup[key])
        elif g["kind"] == "H":
            new_temporal.append(eye2)
        elif g["kind"] == "CP":
            new_temporal.append(cp_identity)
        else:
            raise AssertionError(f"unexpected QFT gate kind {g['kind']}")

    larger_emit_perm = sorted(
        range(len(larger_gates_emit)),
        key=lambda i: larger_gates_emit[i]["kind"] != "H",
    )
    sorted_tensors = [new_temporal[i] for i in larger_emit_perm]
    sorted_tensors = [jnp.asarray(t, dtype=jnp.complex128) for t in sorted_tensors]
    return pdft.QFTBasis(m=new_k, n=new_k, tensors=sorted_tensors)


def qft_inner_outer_indices(
    m: int, n: int, inner_m: int, inner_n: int,
) -> tuple[list[int], list[int]]:
    """Return (inner_indices, outer_indices) for QFTBasis(m, n) given an
    (inner_m, inner_n) inner-block boundary.

    A gate is "inner" iff every qubit it touches is in the inner qubit set:
    axis-1 qubits {1..inner_m}, axis-2 qubits {m+1..m+inner_n}. Indices are
    into the H-first canonical order that QFTBasis(m, n).tensors stores —
    same order used by qft_identity_basis() and qft_warm_from_trained_blocked().

    For m=n=8, inner_m=inner_n=3: returns 12 inner indices (6 H + 6 CP)
    and 60 outer indices.
    """
    from pdft.bases.circuit.qft import _qft_gates_1d

    if inner_m > m or inner_n > n or inner_m < 0 or inner_n < 0:
        raise ValueError(
            f"qft_inner_outer_indices: invalid inner (inner_m={inner_m}, "
            f"inner_n={inner_n}) for m={m}, n={n}"
        )
    if inner_m == 0 and inner_n == 0:
        # No inner qubits — every gate is outer.
        gates_emit = _qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)
        return [], list(range(len(gates_emit)))

    def _is_inner_q(q_1ix: int) -> bool:
        if 1 <= q_1ix <= m:
            return q_1ix <= inner_m
        # axis-2: q in [m+1..m+n]
        return (q_1ix - m) <= inner_n

    gates_emit = _qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)
    # H-first canonical sort, matching QFTBasis storage.
    emit_perm = sorted(
        range(len(gates_emit)),
        key=lambda i: gates_emit[i]["kind"] != "H",
    )
    inner: list[int] = []
    outer: list[int] = []
    for sorted_idx, emit_idx in enumerate(emit_perm):
        g = gates_emit[emit_idx]
        if all(_is_inner_q(q) for q in g["qubits"]):
            inner.append(sorted_idx)
        else:
            outer.append(sorted_idx)
    return inner, outer


__all__ = ["BASIS_FACTORIES", "BasisFactory", "qft_identity_basis",
           "identity_basis_for", "qft_warm_from_smaller_qft",
           "qft_inner_outer_indices"]
