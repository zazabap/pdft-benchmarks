"""Tests for the qft_progressive curriculum: qft_warm_from_smaller_qft helper
and stage-boundary operator preservation."""
import jax.numpy as jnp
import numpy as np

import pdft
from pdft.circuit.builder import controlled_phase_diag
from pdft_benchmarks.bases import qft_warm_from_smaller_qft


def _almost_unitary_2x2(seed: int) -> jnp.ndarray:
    """Build a random 2x2 unitary (close to but not equal to identity)
    via the polar decomposition of (I + 0.5 * Gaussian)."""
    rng = np.random.default_rng(seed)
    A = np.eye(2, dtype=np.complex128) + 0.5 * (
        rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    )
    U, _, Vh = np.linalg.svd(A)
    return jnp.asarray(U @ Vh, dtype=jnp.complex128)


def test_qft_warm_from_smaller_qft_k1_to_k2():
    """QFTBasis(1, 1) -> QFTBasis(2, 2): 2 trained H slots lift into their
    correct positions in the k=2 tensor list; the 2 new H slots and 2 new
    CP slots take their identity element."""
    h_axis1 = _almost_unitary_2x2(seed=1)
    h_axis2 = _almost_unitary_2x2(seed=2)
    smaller = pdft.QFTBasis(m=1, n=1, tensors=[h_axis1, h_axis2])
    assert len(smaller.tensors) == 2

    larger = qft_warm_from_smaller_qft(smaller)

    assert larger.m == 2
    assert larger.n == 2
    assert len(larger.tensors) == 6, f"expected 6 tensors (4 H + 2 CP), got {len(larger.tensors)}"

    eye2 = jnp.eye(2, dtype=jnp.complex128)
    cp_identity = controlled_phase_diag(0.0)

    # Canonical (H-first) order. In larger (QFT(2,2)):
    #   H@q1 = smaller's H@q1 (= h_axis1)
    #   H@q2 = identity (new)
    #   H@q3 = smaller's axis-2 H (= h_axis2; smaller's q2 maps to larger's q3)
    #   H@q4 = identity (new)
    #   CP(q1,q2) = identity (new)
    #   CP(q3,q4) = identity (new)
    assert jnp.allclose(larger.tensors[0], h_axis1, atol=1e-12), \
        "tensor[0] (H@q1) should match smaller's H@q1"
    assert jnp.allclose(larger.tensors[1], eye2, atol=1e-12), \
        "tensor[1] (H@q2) should be I_2 (new gate)"
    assert jnp.allclose(larger.tensors[2], h_axis2, atol=1e-12), \
        "tensor[2] (H@q3) should match smaller's axis-2 H (smaller's q2)"
    assert jnp.allclose(larger.tensors[3], eye2, atol=1e-12), \
        "tensor[3] (H@q4) should be I_2 (new gate)"
    assert jnp.allclose(larger.tensors[4], cp_identity, atol=1e-12), \
        "tensor[4] (CP(q1,q2)) should be cp_identity (new gate)"
    assert jnp.allclose(larger.tensors[5], cp_identity, atol=1e-12), \
        "tensor[5] (CP(q3,q4)) should be cp_identity (new gate)"


def test_qft_warm_from_smaller_qft_k3_to_k4_gate_counts_and_identity_init():
    """QFTBasis(3, 3) -> QFTBasis(4, 4): exactly 12 trained tensors are
    copied; exactly 8 new gates (2 H + 6 CP) take their identity element."""
    rng_tensors = [_almost_unitary_2x2(seed=10 + i) for i in range(6)]
    def _phase_2x2(seed: int) -> jnp.ndarray:
        rng = np.random.default_rng(seed)
        phases = np.exp(1j * rng.uniform(-np.pi, np.pi, size=4))
        return jnp.asarray(phases.reshape(2, 2), dtype=jnp.complex128)
    rng_cp = [_phase_2x2(seed=100 + i) for i in range(6)]
    smaller_tensors = rng_tensors + rng_cp  # 6 H + 6 CP, H-first

    smaller = pdft.QFTBasis(m=3, n=3, tensors=smaller_tensors)
    assert len(smaller.tensors) == 12

    larger = qft_warm_from_smaller_qft(smaller)

    assert larger.m == 4
    assert larger.n == 4
    assert len(larger.tensors) == 20, f"expected 20 tensors, got {len(larger.tensors)}"

    eye2 = jnp.eye(2, dtype=jnp.complex128)
    cp_identity = controlled_phase_diag(0.0)

    n_inherited = 0
    n_new_identity = 0
    for i, t in enumerate(larger.tensors):
        is_h_slot = i < 8
        ident_for_slot = eye2 if is_h_slot else cp_identity
        if jnp.allclose(t, ident_for_slot, atol=1e-12):
            n_new_identity += 1
        else:
            n_inherited += 1
    assert n_inherited >= 12, f"expected ≥12 inherited (non-identity) tensors, got {n_inherited}"
    assert n_new_identity >= 8, f"expected ≥8 new identity tensors, got {n_new_identity}"
    assert n_inherited + n_new_identity == 20


def test_qft_inner_outer_indices_m8_n8_inner3_3():
    """For QFT(8, 8) with (inner_m=3, inner_n=3): exactly 12 inner indices
    (6 H + 6 CP) and 60 outer indices, partitioning {0..71}."""
    from pdft_benchmarks.bases import qft_inner_outer_indices

    inner, outer = qft_inner_outer_indices(m=8, n=8, inner_m=3, inner_n=3)

    assert len(inner) == 12, f"expected 12 inner indices, got {len(inner)}"
    assert len(outer) == 60, f"expected 60 outer indices, got {len(outer)}"
    # Partition: union covers {0..71}, no overlap.
    assert sorted(inner + outer) == list(range(72)), \
        "inner + outer must partition the 72-tensor index space"
    assert set(inner).isdisjoint(set(outer)), "inner/outer must be disjoint"
    # 6 H + 6 CP: first 16 indices are H slots; inner H indices must be
    # in that range and have exactly 6 entries.
    inner_h = [i for i in inner if i < 16]
    inner_cp = [i for i in inner if i >= 16]
    assert len(inner_h) == 6, f"expected 6 inner H indices, got {len(inner_h)}"
    assert len(inner_cp) == 6, f"expected 6 inner CP indices, got {len(inner_cp)}"


def test_qft_inner_outer_indices_degenerate_inner_zero():
    """inner_m=inner_n=0: every gate is outer."""
    from pdft_benchmarks.bases import qft_inner_outer_indices

    inner, outer = qft_inner_outer_indices(m=4, n=4, inner_m=0, inner_n=0)
    assert inner == []
    # QFT(4, 4) has 4+4 H + 6+6 CP = 20 gates.
    assert outer == list(range(20))


def test_operator_preservation_at_stage_boundary_k2_to_k3():
    """BlockedBasis(QFT(2, 2), 6, 6).forward_transform(x) ==
    BlockedBasis(QFT(3, 3) lifted via qft_warm_from_smaller_qft, 5, 5).forward_transform(x)

    Verifies the spec's central operator-preservation claim: putting newly
    introduced gates at identity in the larger QFT keeps the global image
    operator bit-exactly identical at the stage boundary, so the
    training-dynamics curve is continuous across boundaries.
    """
    # Build a small "trained" QFTBasis(2, 2) by perturbing identity init.
    # We need ALL gates to be realistic, including CPs (PhaseManifold —
    # each entry on U(1)).
    h_tensors = [_almost_unitary_2x2(seed=20 + i) for i in range(4)]
    cp_tensors = []
    rng = np.random.default_rng(42)
    for i in range(2):
        phases = np.exp(1j * rng.uniform(-np.pi, np.pi, size=4))
        cp_tensors.append(jnp.asarray(phases.reshape(2, 2), dtype=jnp.complex128))
    smaller_inner = pdft.QFTBasis(m=2, n=2, tensors=h_tensors + cp_tensors)
    smaller_block = pdft.BlockedBasis(
        inner=smaller_inner, block_log_m=6, block_log_n=6,
    )

    # Lift to k=3.
    larger_inner = qft_warm_from_smaller_qft(smaller_inner)
    larger_block = pdft.BlockedBasis(
        inner=larger_inner, block_log_m=5, block_log_n=5,
    )

    # Apply both to the same random 256x256 image.
    rng = np.random.default_rng(123)
    x = jnp.asarray(
        rng.standard_normal((256, 256)) + 1j * rng.standard_normal((256, 256)),
        dtype=jnp.complex128,
    )
    y_small = smaller_block.forward_transform(x)
    y_large = larger_block.forward_transform(x)

    max_abs_diff = float(jnp.max(jnp.abs(y_large - y_small)))
    # Numerical noise floor: 256x256 complex with many gate contractions
    # — 1e-9 is a comfortable tolerance well below any meaningful drift.
    assert max_abs_diff < 1e-9, (
        f"operator preservation FAILED at k=2 -> k=3 stage boundary: "
        f"max |y_large - y_small| = {max_abs_diff:.3e}"
    )
