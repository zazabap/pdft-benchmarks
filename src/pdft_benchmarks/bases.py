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
    # Learnable cosine basis: exact DCT-IV at init, then relaxed (ancilla-free).
    "dct4":          lambda m, n, seed=0: pdft.DCT4Basis(m=m, n=n),
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
    # Learnable ancilla-free DCT-IV inner on b×b blocks — apples-to-apples with
    # block_dct_b / rich_b / real_rich_b over the HEVC transform-unit sweep
    # {4,8,16,32}. (dct4_32 is degenerate with full-image dct4 at m=5, so it is
    # excluded from the QuickDraw grid.)
    "dct4_4":        lambda m, n, seed=0: _blocked(m, n, seed, pdft.DCT4Basis,      inner_m=2, inner_n=2),
    "dct4_8":        lambda m, n, seed=0: _blocked(m, n, seed, pdft.DCT4Basis,      inner_m=3, inner_n=3),
    "dct4_16":       lambda m, n, seed=0: _blocked(m, n, seed, pdft.DCT4Basis,      inner_m=4, inner_n=4),
    "dct4_32":       lambda m, n, seed=0: _blocked(m, n, seed, pdft.DCT4Basis,      inner_m=5, inner_n=5),
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


# ---------------------------------------------------------------------------
# Generic per-family identity / random init (consumed by qft_progressive).
#
# Two generic dispatchers reproduce, bit-for-bit, the per-family builders the
# family×init progressive sweep needs — identity-operator init and Haar/native
# random init for the six families (qft, rich, real_rich, tebd, entangled_qft,
# mera) — so the committed random cells stay reproducible.
# ---------------------------------------------------------------------------


def _circuit_identity_basis(inner_cls, m: int, n: int):
    """Shape-based identity-operator init for U(4) families (Rich/RealRich):
    every (2,2) gate -> I_2, every (2,2,2,2) gate -> I_4."""
    import jax.numpy as jnp

    base = inner_cls(m=m, n=n)
    eye2 = jnp.eye(2, dtype=jnp.complex128)
    eye4 = jnp.eye(4, dtype=jnp.complex128).reshape(2, 2, 2, 2)
    new_tensors = []
    for t in base.tensors:
        if t.shape == (2, 2):
            new_tensors.append(eye2)
        elif t.shape == (2, 2, 2, 2):
            new_tensors.append(eye4)
        else:
            raise AssertionError(
                f"{inner_cls.__name__}: unexpected gate tensor shape {t.shape}; "
                "_circuit_identity_basis only handles H (2,2) and U4 (2,2,2,2)"
            )
    return inner_cls(m=m, n=n, tensors=new_tensors,
                     code=base.code, inv_code=base.inv_code)


def _circuit_identity_by_value(inner_cls, m: int, n: int):
    """Value-based identity-operator init for the QFT-derived phase families
    (QFT/EntangledQFT/TEBD/MERA): Hadamard slot -> I_2, controlled-phase slot
    -> controlled_phase_diag(0) = [[1,1],[1,1]], U4 slot -> I_4."""
    import jax.numpy as jnp
    import numpy as np
    from pdft.circuit.builder import controlled_phase_diag

    hadamard = jnp.asarray(np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0),
                           dtype=jnp.complex128)
    eye4 = jnp.eye(4, dtype=jnp.complex128).reshape(2, 2, 2, 2)
    cp0 = controlled_phase_diag(0.0)

    base = inner_cls(m=m, n=n)
    new_tensors = []
    for t in base.tensors:
        t = jnp.asarray(t)
        if t.shape == (2, 2, 2, 2):
            new_tensors.append(eye4)
        elif t.shape == (2, 2):
            if bool(jnp.allclose(t, hadamard, atol=1e-9)):
                new_tensors.append(jnp.eye(2, dtype=jnp.complex128))
            else:
                new_tensors.append(cp0)
        else:
            raise AssertionError(
                f"{inner_cls.__name__}: unexpected gate tensor shape {t.shape}"
            )
    return inner_cls(m=m, n=n, tensors=new_tensors,
                     code=base.code, inv_code=base.inv_code)


def _haar_unitary(dim: int, rng):
    """Haar-distributed U(dim) via QR of a complex Ginibre matrix (Mezzadri)."""
    import numpy as np

    z = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    ph = np.diagonal(r)
    ph = ph / np.abs(ph)
    return q * ph


def _haar_special_orthogonal(dim: int, rng):
    """Haar-distributed SO(dim) via QR of a real Ginibre matrix (det = +1)."""
    import numpy as np

    z = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(z)
    q = q * np.sign(np.diagonal(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _circuit_random_basis(inner_cls, m: int, n: int, seed: int):
    """Haar-random init for U(4) families: complex U(2)/U(4) for RichBasis,
    real SO(2)/SO(4) for RealRichBasis. Seeded, reproducible."""
    import jax.numpy as jnp
    import numpy as np

    base = inner_cls(m=m, n=n)
    is_real = inner_cls is pdft.RealRichBasis
    rng = np.random.default_rng(seed)
    new_tensors = []
    for t in base.tensors:
        if t.shape == (2, 2):
            u = _haar_special_orthogonal(2, rng) if is_real else _haar_unitary(2, rng)
            new_tensors.append(jnp.asarray(u, dtype=jnp.complex128))
        elif t.shape == (2, 2, 2, 2):
            u = _haar_special_orthogonal(4, rng) if is_real else _haar_unitary(4, rng)
            new_tensors.append(jnp.asarray(u.reshape(2, 2, 2, 2), dtype=jnp.complex128))
        else:
            raise AssertionError(
                f"{inner_cls.__name__}: unexpected gate tensor shape {t.shape}; "
                "_circuit_random_basis only handles H (2,2) and U4 (2,2,2,2)"
            )
    return inner_cls(m=m, n=n, tensors=new_tensors,
                     code=base.code, inv_code=base.inv_code)


def _qft_random_basis(m: int, n: int, seed: int):
    """QFT topology with Haar-random gate init: H slot -> Haar U(2), CP slot ->
    controlled_phase_diag(uniform[0, 2pi)). Seeded, reproducible."""
    import jax.numpy as jnp
    import numpy as np
    from pdft.circuit.builder import controlled_phase_diag

    hadamard = jnp.asarray(np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0),
                           dtype=jnp.complex128)
    base = pdft.QFTBasis(m=m, n=n)
    rng = np.random.default_rng(seed)
    new_tensors = []
    for t in base.tensors:
        t = jnp.asarray(t)
        if t.shape == (2, 2) and bool(jnp.allclose(t, hadamard, atol=1e-9)):
            new_tensors.append(jnp.asarray(_haar_unitary(2, rng), dtype=jnp.complex128))
        elif t.shape == (2, 2):
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            new_tensors.append(jnp.asarray(controlled_phase_diag(phi), dtype=jnp.complex128))
        else:
            raise AssertionError(f"QFTBasis: unexpected gate tensor shape {t.shape}")
    return pdft.QFTBasis(m=m, n=n, tensors=new_tensors,
                         code=base.code, inv_code=base.inv_code)


def dct4_random_basis(m: int, n: int, seed: int):
    """Haar real-orthogonal init of the learnable DCT-IV circuit (`DCT4Basis`).

    The QFT-study init policy (`_qft_random_basis`: Haar U(2) on Hadamard slots,
    uniform phase on controlled-phase slots) translated to the **real-orthogonal**
    DCT-IV setting, so the random init stays a real-orthogonal operator (DCT-IV's
    defining property) rather than drifting complex like the QFT one:

      - U(4) gate (mirror-Q CNOT, controlled-R_y; tensor shape (2,2,2,2))
        -> Haar SO(4), reshaped.
      - controlled-phase "Delta" sign gate (tensor shape (2,2) in the
        `controlled_phase_diag` diag-stack form, row 0 == [1, 1])
        -> `controlled_phase_diag(random {0, pi})` (a random real sign).
      - any other (2,2) gate (affine R_y rotation, branch Hadamard)
        -> Haar SO(2).

    Built on the canonical `DCT4Basis(m, n)` so the gate topology +
    `code`/`inv_code` (and therefore each gate's auto-selected Riemannian
    manifold) are unchanged; only the gate *values* are reseeded. Seeded and
    reproducible via `np.random.default_rng(seed)`. Used by the DCT-IV
    seed-variance study (the normal-training analog of the QFT seed sweep).
    """
    import jax.numpy as jnp
    import numpy as np
    from pdft.circuit.builder import controlled_phase_diag

    base = pdft.DCT4Basis(m=m, n=n)
    rng = np.random.default_rng(seed)
    new_tensors = []
    for t in base.tensors:
        t = jnp.asarray(t)
        if t.shape == (2, 2, 2, 2):
            u = _haar_special_orthogonal(4, rng)
            new_tensors.append(jnp.asarray(u.reshape(2, 2, 2, 2), dtype=jnp.complex128))
        elif t.shape == (2, 2):
            # controlled_phase_diag(phi) == [[1, 1], [1, e^{i phi}]]; the CP
            # "Delta" sign gate is the only (2,2) gate whose row 0 is [1, 1].
            if bool(jnp.allclose(t[0], jnp.ones(2, dtype=t.dtype), atol=1e-9)):
                phi = float(rng.choice([0.0, np.pi]))
                new_tensors.append(jnp.asarray(controlled_phase_diag(phi), dtype=jnp.complex128))
            else:
                u = _haar_special_orthogonal(2, rng)
                new_tensors.append(jnp.asarray(u, dtype=jnp.complex128))
        else:
            raise AssertionError(
                f"DCT4Basis: unexpected gate tensor shape {t.shape}; "
                "dct4_random_basis only handles (2,2) and (2,2,2,2)"
            )
    return pdft.DCT4Basis(m=m, n=n, tensors=new_tensors,
                          code=base.code, inv_code=base.inv_code)


def dct4_random_controlled_basis(m: int, n: int, seed: int):
    """Haar real-orthogonal init of the *controlled* (O(2)-twiddle) DCT-IV.

    The controlled analogue of `dct4_random_basis`: same init policy on the
    trainable gates, but built on `DCT4Basis(parametrization="controlled")` so
    the mirror Q/R CNOTs are the structural "CX" flips (fixed routing) and the
    twiddle is a single-angle O(2) `CRY` leaf rather than dense O(4).

      - (2,2,2,2) gate (the CX mirror CNOT): kept AS-IS — fixed routing, applied
        as an index flip, contributes no trainable parameter.
      - controlled-phase "Delta" sign gate ((2,2), row 0 == [1, 1]):
        `controlled_phase_diag(random {0, pi})`.
      - any other (2,2) gate (the O(2) `CRY` twiddle or branch Hadamard):
        Haar SO(2).

    Reproducible via `np.random.default_rng(seed)`. Returns the basis only
    (the CX mirror has zero gradient by construction, so no `frozen_indices`
    is needed — it cannot move).
    """
    import jax.numpy as jnp
    import numpy as np
    from pdft.circuit.builder import controlled_phase_diag

    base = pdft.DCT4Basis(m=m, n=n, parametrization="controlled")
    rng = np.random.default_rng(seed)
    new_tensors = []
    for t in base.tensors:
        t = jnp.asarray(t)
        if t.shape == (2, 2, 2, 2):
            new_tensors.append(t)  # CX mirror CNOT: fixed routing, keep exact
        elif t.shape == (2, 2):
            if bool(jnp.allclose(t[0], jnp.ones(2, dtype=t.dtype), atol=1e-9)):
                phi = float(rng.choice([0.0, np.pi]))
                new_tensors.append(jnp.asarray(controlled_phase_diag(phi), dtype=jnp.complex128))
            else:
                u = _haar_special_orthogonal(2, rng)
                new_tensors.append(jnp.asarray(u, dtype=jnp.complex128))
        else:
            raise AssertionError(
                f"DCT4Basis(controlled): unexpected gate tensor shape {t.shape}; "
                "dct4_random_controlled_basis only handles (2,2) and (2,2,2,2)"
            )
    return pdft.DCT4Basis(m=m, n=n, tensors=new_tensors,
                          parametrization="controlled",
                          code=base.code, inv_code=base.inv_code)


_FAMILY_CLASS = {
    "qft": pdft.QFTBasis,
    "rich": pdft.RichBasis,
    "real_rich": pdft.RealRichBasis,
    "tebd": pdft.TEBDBasis,
    "entangled_qft": pdft.EntangledQFTBasis,
    "mera": pdft.MERABasis,
}


def family_identity_basis(family: str, m: int, n: int):
    """Identity-operator init for a circuit family (bare, unblocked).

    Rich/RealRich use shape-based identity (U(2)/U(4)); QFT/TEBD/EntangledQFT/
    MERA use value-based identity (Hadamard -> I_2, controlled-phase -> phase 0).
    """
    cls = _FAMILY_CLASS[family]
    if family in ("rich", "real_rich"):
        return _circuit_identity_basis(cls, m, n)
    return _circuit_identity_by_value(cls, m, n)


def family_random_basis(family: str, m: int, n: int, seed: int):
    """Random (Haar / native-seed) init for a circuit family (bare, unblocked).

    QFT -> Haar U(2) on H slots + uniform phase on CP slots; Rich/RealRich ->
    Haar U/SO on every gate; TEBD/EntangledQFT/MERA -> the constructor's own
    seeded random init. Seeded; reproduces the family×init sweep's random cells.
    """
    if family == "dct4":
        return dct4_random_basis(m, n, seed)
    cls = _FAMILY_CLASS[family]
    if family in ("tebd", "entangled_qft", "mera"):
        return cls(m=m, n=n, seed=seed)
    if family == "qft":
        return _qft_random_basis(m, n, seed)
    return _circuit_random_basis(cls, m, n, seed)


def dct4_controlled_basis(m: int, n: int):
    """Structured (O(2)) DCT-IV basis + the mirror-CNOT indices to freeze.

    Returns ``(basis, frozen_indices)`` where ``basis`` is
    ``pdft.DCT4Basis(m, n, parametrization="controlled")`` (twiddle gates train
    on O(2), no dense O(4)) and ``frozen_indices`` are the mirror ``Q``/``R``
    CNOT gates — the only remaining ``(2,2,2,2)`` gates — to pass as
    ``train_basis_batched(frozen_indices=...)`` so they stay fixed routing.
    """
    import pdft

    basis = pdft.DCT4Basis(m=m, n=n, parametrization="controlled")
    frozen_indices = [i for i, t in enumerate(basis.tensors)
                      if tuple(t.shape) == (2, 2, 2, 2)]
    return basis, frozen_indices


__all__ = ["BASIS_FACTORIES", "BasisFactory", "qft_identity_basis",
           "identity_basis_for", "qft_warm_from_smaller_qft",
           "qft_inner_outer_indices",
           "family_identity_basis", "family_random_basis", "dct4_random_basis",
           "dct4_controlled_basis", "dct4_random_controlled_basis"]
