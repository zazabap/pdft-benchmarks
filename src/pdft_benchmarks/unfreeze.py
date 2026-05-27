"""Progressive gate-unfreezing training loop, plateau detector,
Riemannian grad-norm probe, and QFT unfreeze-order helper."""
from __future__ import annotations


def qft_unfreeze_orders(m: int, n: int) -> dict[str, list[int]]:
    """Three unfreeze orderings as lists of indices into a QFTBasis(m, n).tensors.

    `basis.tensors` is stored Hadamard-first (stable within group), matching
    `qft_identity_basis`. We rebuild the emission gate list, map each emitted
    gate to its storage index, then order the storage indices three ways:

      - lr: emission order
      - rl: reverse emission order
      - bg: block-growth — group by block-stage k (highest within-axis qubit a
        gate touches); per stage do row-axis then col-axis, H_k before its CPs,
        CPs by ascending lower qubit. Stage k completes QFT(k, k).
    """
    from pdft.bases.circuit.qft import _qft_gates_1d

    gates = _qft_gates_1d(m, offset=0) + _qft_gates_1d(n, offset=m)
    G = len(gates)

    # emission index -> storage index (Hadamard-first stable sort, == qft_identity_basis)
    emit_perm = sorted(range(G), key=lambda i: gates[i]["kind"] != "H")
    emission_to_storage = [0] * G
    for storage_pos, emission_idx in enumerate(emit_perm):
        emission_to_storage[emission_idx] = storage_pos

    def axis_of(q: int) -> int:
        return 0 if q <= m else 1

    def within(q: int) -> int:
        return q if q <= m else q - m

    keys = []  # (stage, axis, kind_rank, lower_qubit, emission_idx)
    for e, g in enumerate(gates):
        if g["kind"] == "H":
            (q,) = g["qubits"]
            keys.append((within(q), axis_of(q), 0, within(q), e))
        else:  # CP, qubits = (control, target), same axis
            c, t = g["qubits"]
            stage = max(within(c), within(t))
            keys.append((stage, axis_of(c), 1, min(within(c), within(t)), e))

    lr = [emission_to_storage[e] for e in range(G)]
    rl = list(reversed(lr))
    bg = [emission_to_storage[k[4]] for k in sorted(keys)]
    return {"bg": bg, "lr": lr, "rl": rl}


def _plateau_reason(grad_norm, loss, loss_prev, *, step, min_steps, grad_tol, loss_tol):
    """Return the trigger reason ("grad_norm" | "loss_delta") or None.

    Not evaluated until `step >= min_steps`. Grad-norm stationarity takes
    precedence; the loss-flatness check needs a previous loss to compare.
    """
    if step < min_steps:
        return None
    if grad_norm < grad_tol:
        return "grad_norm"
    if loss_prev is not None and abs(loss - loss_prev) < loss_tol:
        return "loss_delta"
    return None


def _make_gradnorm_probe(basis, loss):
    """Build `probe(tensors, batch, frozen_set) -> (loss: float, grad_norm: float)`.

    Mirrors `_build_jit_adam_step`'s forward/backward (same loss_function, same
    Wirtinger conjugation) then projects the Euclidean gradient onto the manifold
    tangent space via pdft's `_batched_project`, zeroing frozen indices — so the
    norm reflects stationarity of the *trainable* gates only.
    """
    import jax
    import jax.numpy as jnp
    from pdft.loss import loss_function
    from pdft.optimizers.core import _batched_project, _common_setup

    m, n = basis.m, basis.n
    code, inv_code = basis.code, basis.inv_code

    def _per_image(tensors, img):
        return loss_function(tensors, m, n, code, img, loss, inverse_code=inv_code)

    _batched = jax.vmap(_per_image, in_axes=(None, 0))

    def _stacked_loss(tensors, batch):
        return jnp.mean(_batched(tensors, batch))

    _val_grad = jax.jit(jax.value_and_grad(_stacked_loss))

    def probe(tensors, batch, frozen_set):
        loss_val, raw_grads = _val_grad(tensors, batch)
        grads = [jnp.conj(g) for g in raw_grads]  # Wirtinger, matches adam_step
        state = _common_setup(tensors)
        _, grad_norm = _batched_project(state, grads,
                                        frozen_indices=frozen_set or None)
        return float(loss_val), float(grad_norm)

    return probe
