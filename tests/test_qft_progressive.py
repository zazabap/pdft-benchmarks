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
