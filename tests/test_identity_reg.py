"""Tests for identity_reg: identity-element table, inner/outer mask,
and block-masked reg-loss class."""
import jax.numpy as jnp
import pdft
from pdft.loss import MSELoss
from pdft_benchmarks.identity_reg import (
    BlockMaskedIdentityRegQFTMSELoss,
    qft_identity_table,
    qft_inner_mask,
)


# --- identity table -----------------------------------------------------

def test_identity_table_length_qft_5_5():
    """QFT(5,5) has 5+5 Hadamards + 5*4/2 + 5*4/2 = 10 + 10 + 10 = 30 gates."""
    table = qft_identity_table(5, 5)
    assert len(table) == 30


def test_identity_table_length_qft_8_8():
    """QFT(8,8) has 16 Hadamards + 28 + 28 = 72 gates."""
    table = qft_identity_table(8, 8)
    assert len(table) == 72


def test_identity_table_hadamards_first():
    """First m+n entries are 2x2 identity (Hadamard identity element)."""
    m, n = 5, 5
    table = qft_identity_table(m, n)
    eye2 = jnp.eye(2, dtype=jnp.complex128)
    for i in range(m + n):
        assert jnp.allclose(table[i], eye2), f"entry {i} is not I_2"


def test_identity_table_cps_second():
    """Remaining entries are controlled-phase identity = [[1,1],[1,1]]."""
    from pdft.circuit.builder import controlled_phase_diag
    m, n = 5, 5
    table = qft_identity_table(m, n)
    cp_id = controlled_phase_diag(0.0)
    for i in range(m + n, len(table)):
        assert jnp.allclose(table[i], cp_id), f"entry {i} is not CP-identity"


def test_identity_table_matches_qft_identity_basis():
    """Table aligns with qft_identity_basis tensor order, gate-for-gate."""
    from pdft_benchmarks.bases import qft_identity_basis
    m, n = 5, 5
    table = qft_identity_table(m, n)
    basis = qft_identity_basis(m, n)
    assert len(table) == len(basis.tensors)
    for i, (t, ref) in enumerate(zip(table, basis.tensors)):
        assert jnp.allclose(t, ref), f"entry {i} mismatch with qft_identity_basis"


# --- inner mask ---------------------------------------------------------

def test_inner_mask_length_matches_table():
    """Mask length matches identity-table length: 72 for QFT(8,8)."""
    mask = qft_inner_mask(8, 8, 3, 3)
    assert len(mask) == 72


def test_inner_mask_counts_match_blocked_8():
    """Inner = 12 gates (6 H + 6 CP); outer = 60 gates.

    Matches the 12-vs-60 structural split of blocked_8's optimum.
    """
    mask = qft_inner_mask(8, 8, 3, 3)
    n_inner = sum(mask)
    n_outer = len(mask) - n_inner
    assert n_inner == 12, f"expected 12 inner gates, got {n_inner}"
    assert n_outer == 60, f"expected 60 outer gates, got {n_outer}"


def test_inner_mask_hadamard_counts():
    """Of the first m+n entries (Hadamards), exactly inner_m + inner_n are inner."""
    m, n, inner_m, inner_n = 8, 8, 3, 3
    mask = qft_inner_mask(m, n, inner_m, inner_n)
    hadamard_mask = mask[:m + n]
    assert sum(hadamard_mask) == inner_m + inner_n, \
        f"expected {inner_m + inner_n} inner Hadamards, got {sum(hadamard_mask)}"


def test_inner_mask_consistent_with_warm_start_structure():
    """Inner gates here correspond to the gates that take trained-inner
    values in qft_warm_from_trained_blocked. Verify by independent
    construction.

    qft_warm_from_trained_blocked pins gates as identity unless they
    act ONLY on inner qubits (axis-1 q ∈ [1..inner_m], axis-2 q ∈
    [m+1..m+inner_n]). That's exactly the inner-mask definition here.
    """
    from pdft.bases.circuit.qft import _qft_gates_1d
    m, n, inner_m, inner_n = 8, 8, 3, 3
    mask = qft_inner_mask(m, n, inner_m, inner_n)

    def _is_inner_q(q_1ix: int) -> bool:
        if 1 <= q_1ix <= m:
            return q_1ix <= inner_m
        return (q_1ix - m) <= inner_n

    gates_emit = _qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)
    perm = sorted(range(len(gates_emit)),
                  key=lambda i: gates_emit[i]["kind"] != "H")
    sorted_gates = [gates_emit[i] for i in perm]

    for i, g in enumerate(sorted_gates):
        is_inner = all(_is_inner_q(q) for q in g["qubits"])
        assert mask[i] == is_inner, \
            f"gate {i} (kind={g['kind']}, qubits={g['qubits']}): mask says " \
            f"{mask[i]}, manual says {is_inner}"


# --- BlockMaskedIdentityRegQFTMSELoss -----------------------------------

def test_reg_loss_lam_zero_equals_base_mse():
    """At lambda=0, the reg loss equals plain MSELoss on the same tensors."""
    from pdft.bases.circuit.qft import qft_code
    from pdft.loss import loss_function

    m, n = 2, 2
    code, tensors = qft_code(m, n)
    inv_code, _ = qft_code(m, n, inverse=True)
    pic = jnp.ones((4, 4), dtype=jnp.complex128) / 4.0

    base = float(loss_function(tensors, m, n, code, pic, MSELoss(k=1),
                                inverse_code=inv_code))
    reg = float(loss_function(
        tensors, m, n, code, pic,
        BlockMaskedIdentityRegQFTMSELoss(k=1, lam=0.0, m=m, n=n,
                                          inner_m=1, inner_n=1,
                                          outer_weight=10.0),
        inverse_code=inv_code,
    ))
    assert abs(reg - base) < 1e-10


def test_reg_loss_at_identity_tensors_is_pure_mse():
    """When tensors == identity table, reg term is zero; loss == base MSE.

    Holds for any (lam, outer_weight) because every gate's distance from
    its identity element is exactly zero at qft_identity init.
    """
    from pdft.bases.circuit.qft import qft_code
    from pdft.loss import loss_function
    from pdft_benchmarks.bases import qft_identity_basis

    m, n = 2, 2
    code, _ = qft_code(m, n)
    inv_code, _ = qft_code(m, n, inverse=True)
    basis = qft_identity_basis(m, n)
    tensors = list(basis.tensors)
    pic = jnp.ones((4, 4), dtype=jnp.complex128) / 4.0

    lam = 0.5
    base = float(loss_function(tensors, m, n, code, pic, MSELoss(k=1),
                                inverse_code=inv_code))
    reg = float(loss_function(
        tensors, m, n, code, pic,
        BlockMaskedIdentityRegQFTMSELoss(k=1, lam=lam, m=m, n=n,
                                          inner_m=1, inner_n=1,
                                          outer_weight=10.0),
        inverse_code=inv_code,
    ))
    assert abs(reg - base) < 1e-10


def test_reg_loss_outer_weight_amplifies_outer_gates_only():
    """Outer gates contribute W * ||T_g - I_g||_F^2; inner contribute 1x.

    Construct: take identity tensors and perturb exactly ONE outer gate
    and ONE inner gate by the same amount. Reg term should be
    lam * (1 + W) * ||perturbation||_F^2.
    """
    from pdft.bases.circuit.qft import qft_code
    from pdft.loss import loss_function
    from pdft_benchmarks.bases import qft_identity_basis

    m, n, inner_m, inner_n = 5, 5, 3, 3
    code, _ = qft_code(m, n)
    inv_code, _ = qft_code(m, n, inverse=True)
    basis = qft_identity_basis(m, n)
    base_tensors = [jnp.asarray(t) for t in basis.tensors]
    pic = jnp.zeros((2**m, 2**n), dtype=jnp.complex128)  # base MSE = 0

    mask = qft_inner_mask(m, n, inner_m, inner_n)
    # Find one inner and one outer Hadamard index.
    inner_h_idx = next(i for i in range(m + n) if mask[i])
    outer_h_idx = next(i for i in range(m + n) if not mask[i])

    pert = jnp.zeros((2, 2), dtype=jnp.complex128).at[0, 0].set(0.1)
    pert_norm_sq = float(jnp.sum(jnp.abs(pert) ** 2))   # 0.01

    t_both = [t for t in base_tensors]
    t_both[inner_h_idx] = base_tensors[inner_h_idx] + pert
    t_both[outer_h_idx] = base_tensors[outer_h_idx] + pert

    lam, W = 1.0, 10.0
    loss = float(loss_function(
        t_both, m, n, code, pic,
        BlockMaskedIdentityRegQFTMSELoss(k=1, lam=lam, m=m, n=n,
                                          inner_m=inner_m, inner_n=inner_n,
                                          outer_weight=W),
        inverse_code=inv_code,
    ))
    base = float(loss_function(t_both, m, n, code, pic, MSELoss(k=1),
                                inverse_code=inv_code))
    expected_reg = lam * (1.0 + W) * pert_norm_sq
    assert abs(loss - base - expected_reg) < 1e-6, \
        f"got reg = {loss - base}, expected {expected_reg}"


# --- end-to-end lambda=0 equivalence -----------------------------------

# --- L1IdentityRegQFTMSELoss ---------------------------------------------

def test_l1_reg_loss_lam_zero_equals_base_mse():
    """At lambda=0, the L1 reg loss equals plain MSELoss."""
    from pdft.bases.circuit.qft import qft_code
    from pdft.loss import loss_function
    from pdft_benchmarks.identity_reg import L1IdentityRegQFTMSELoss

    m, n = 2, 2
    code, tensors = qft_code(m, n)
    inv_code, _ = qft_code(m, n, inverse=True)
    pic = jnp.ones((4, 4), dtype=jnp.complex128) / 4.0

    base = float(loss_function(tensors, m, n, code, pic, MSELoss(k=1),
                                inverse_code=inv_code))
    reg = float(loss_function(
        tensors, m, n, code, pic,
        L1IdentityRegQFTMSELoss(k=1, lam=0.0, m=m, n=n),
        inverse_code=inv_code,
    ))
    assert abs(reg - base) < 1e-10


def test_l1_reg_loss_at_identity_tensors_is_pure_mse():
    """At qft_identity init (T_g == I_g for all g), the L1 reg term equals
    only the smoothing constant lam * n_gates * sqrt(eps) — a small
    deterministic offset (≈ 3e-6 at the default eps=1e-12), not zero.
    """
    import math
    from pdft.bases.circuit.qft import qft_code
    from pdft.loss import loss_function
    from pdft_benchmarks.bases import qft_identity_basis
    from pdft_benchmarks.identity_reg import L1IdentityRegQFTMSELoss

    m, n = 2, 2
    code, _ = qft_code(m, n)
    inv_code, _ = qft_code(m, n, inverse=True)
    basis = qft_identity_basis(m, n)
    tensors = list(basis.tensors)
    pic = jnp.ones((4, 4), dtype=jnp.complex128) / 4.0

    lam = 0.5
    eps = 1e-12
    n_gates = len(tensors)
    expected_smoothing_offset = lam * n_gates * math.sqrt(eps)

    base = float(loss_function(tensors, m, n, code, pic, MSELoss(k=1),
                                inverse_code=inv_code))
    reg = float(loss_function(
        tensors, m, n, code, pic,
        L1IdentityRegQFTMSELoss(k=1, lam=lam, m=m, n=n, eps=eps),
        inverse_code=inv_code,
    ))
    assert abs(reg - base - expected_smoothing_offset) < 1e-9


def test_l1_reg_loss_gradient_at_identity_is_finite():
    """Regression test: at qft_identity init, the L1 reg gradient must be
    finite (not NaN). Naive jnp.linalg.norm at the kink returns NaN and
    NaN-poisons Adam's first step, freezing every gate at identity.
    """
    import jax
    from pdft.bases.circuit.qft import qft_code
    from pdft.loss import loss_function
    from pdft_benchmarks.bases import qft_identity_basis
    from pdft_benchmarks.identity_reg import L1IdentityRegQFTMSELoss

    m, n = 3, 3
    code, _ = qft_code(m, n)
    inv_code, _ = qft_code(m, n, inverse=True)
    basis = qft_identity_basis(m, n)
    tensors = list(basis.tensors)
    pic = jnp.ones((2**m, 2**n), dtype=jnp.complex128) / (2**(m+n))

    loss_obj = L1IdentityRegQFTMSELoss(k=1, lam=1.0, m=m, n=n)

    def total_loss(tensors_list):
        return loss_function(tensors_list, m, n, code, pic, loss_obj,
                             inverse_code=inv_code)

    grads = jax.grad(total_loss)(tensors)
    for i, g in enumerate(grads):
        assert jnp.all(jnp.isfinite(g)), \
            f"gate {i} gradient has NaN/Inf at qft_identity init"


def test_l1_reg_loss_adds_lam_times_sum_of_fro_distances():
    """Reg term = lam * sum_g ||T_g - I_g||_F (unsquared)."""
    from pdft.bases.circuit.qft import qft_code
    from pdft.loss import loss_function
    from pdft_benchmarks.identity_reg import L1IdentityRegQFTMSELoss

    m, n = 2, 2
    code, tensors = qft_code(m, n)
    inv_code, _ = qft_code(m, n, inverse=True)
    pic = jnp.zeros((4, 4), dtype=jnp.complex128)

    table = qft_identity_table(m, n)
    # Use the same smoothed form the class implements: sqrt(||T - I||_F^2 + eps).
    eps = 1e-12
    expected_reg = float(sum(jnp.sqrt(jnp.sum(jnp.abs(t - i) ** 2) + eps)
                              for t, i in zip(tensors, table)))

    lam = 0.3
    base = float(loss_function(tensors, m, n, code, pic, MSELoss(k=1),
                                inverse_code=inv_code))
    reg = float(loss_function(
        tensors, m, n, code, pic,
        L1IdentityRegQFTMSELoss(k=1, lam=lam, m=m, n=n, eps=eps),
        inverse_code=inv_code,
    ))
    assert abs(reg - base - lam * expected_reg) < 1e-6


def test_reg_loss_lam_zero_e2e_matches_base_mse_training():
    """End-to-end: training with BlockMaskedIdentityRegQFTMSELoss(lam=0)
    produces the same loss trajectory as training with vanilla MSELoss.

    Tiny config (m=n=3, 4 images, 2 epochs) to keep the test fast.
    """
    import numpy as np
    from pdft_benchmarks.bases import qft_identity_basis

    m, n = 3, 3
    rng = np.random.default_rng(0)
    imgs = [
        jnp.asarray(rng.standard_normal((2**m, 2**n)).astype(np.float64),
                    dtype=jnp.complex128)
        for _ in range(4)
    ]
    k = max(1, round(2 ** (m + n) * 0.1))

    basis_a = qft_identity_basis(m, n)
    res_a = pdft.train_basis_batched(
        basis_a, dataset=imgs, loss=pdft.MSELoss(k=k),
        epochs=2, batch_size=2, optimizer="adam",
        validation_split=0.0, early_stopping_patience=10**9,
        warmup_frac=0.1, lr_peak=1e-3, lr_final=1e-5,
        max_grad_norm=None, shuffle=False, seed=42,
    )

    basis_b = qft_identity_basis(m, n)
    res_b = pdft.train_basis_batched(
        basis_b, dataset=imgs,
        loss=BlockMaskedIdentityRegQFTMSELoss(
            k=k, lam=0.0, m=m, n=n,
            inner_m=2, inner_n=2, outer_weight=10.0),
        epochs=2, batch_size=2, optimizer="adam",
        validation_split=0.0, early_stopping_patience=10**9,
        warmup_frac=0.1, lr_peak=1e-3, lr_final=1e-5,
        max_grad_norm=None, shuffle=False, seed=42,
    )

    assert len(res_a.loss_history) == len(res_b.loss_history)
    for la, lb in zip(res_a.loss_history, res_b.loss_history):
        assert abs(la - lb) < 1e-9, f"loss divergence: {la} vs {lb}"
    for ta, tb in zip(res_a.basis.tensors, res_b.basis.tensors):
        assert jnp.allclose(ta, tb, atol=1e-9), "final tensors differ"
