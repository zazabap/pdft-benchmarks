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


def dct4_unfreeze_orders(m: int, n: int) -> dict[str, list[int]]:
    """Three unfreeze orderings as lists of indices into ``DCT4Basis(m, n).tensors``.

    DCT4Basis stores gates Hadamard-first BY VALUE (only the branch-H merges
    equal the Hadamard; the R_y rotations are kind "H" but are not), so we read
    the storage permutation from the builder's value-based ``_hadamard_first_perm``
    instead of the QFT kind-based key.

      - lr: emission order
      - rl: reverse emission order
      - bg: block growth — group by stage (the largest within-axis builder qubit
        a gate touches; = sub-block size, since the DCT-IV on qubits 1..k IS the
        size-2^k block), row axis before col, emission order within. Unfreezing
        stages 1..k completes the blocked DCT-IV(k, k).
    """
    from pdft.bases.circuit.dct4 import _dct4_gates_1d
    from pdft.circuit.builder import _hadamard_first_perm

    gates = _dct4_gates_1d(m, offset=0) + _dct4_gates_1d(n, offset=m)
    G = len(gates)
    perm = _hadamard_first_perm([g["tensor"] for g in gates])
    emission_to_storage = [0] * G
    for storage_pos, emission_idx in enumerate(perm):
        emission_to_storage[emission_idx] = storage_pos

    def axis_of(q: int) -> int:
        return 0 if q <= m else 1

    def within(q: int) -> int:
        return q if q <= m else q - m

    keys = []  # (stage, axis, emission_idx)
    for e, g in enumerate(gates):
        within_qs = [within(q) for q in g["qubits"]]
        keys.append((max(within_qs), axis_of(g["qubits"][0]), e))

    lr = [emission_to_storage[e] for e in range(G)]
    rl = list(reversed(lr))
    bg = [emission_to_storage[k[2]] for k in sorted(keys)]
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


from dataclasses import dataclass, field


@dataclass
class StageSummary:
    stage: int
    n_trainable: int
    gate_index: int
    start_step: int
    end_step: int
    n_steps: int
    final_loss: float
    final_grad_norm: float
    trigger: str
    extra: dict = field(default_factory=dict)  # e.g. per-stage PSNR from a callback


@dataclass
class UnfreezeResult:
    basis: object
    trace: list  # list[dict]: step, stage, n_trainable, loss, grad_norm
    stages: list  # list[StageSummary]


def train_progressive_unfreeze(
    basis, dataset, *,
    unfreeze_order, lr, max_steps_per_stage, loss,
    grad_tol=1e-5, loss_tol=1e-5, min_steps_per_stage=5,
    beta1=0.9, beta2=0.999, eps=1e-8, seed=0,
    stage_callback=None, grad_check_every=1,
):
    """Cumulatively unfreeze gates in `unfreeze_order`, training each stage to a
    plateau on a fixed batch (`dataset`). Returns an `UnfreezeResult`.

    `stage_callback(stage:int, tensors:list) -> dict | None` runs at each stage
    end; its return is stored on the stage summary's `extra` (used for PSNR).
    """
    import jax.numpy as jnp
    from pdft.manifolds import group_by_manifold, stack_tensors
    from pdft.training.adam_step import _build_jit_adam_step

    batch = jnp.stack([jnp.asarray(x, dtype=jnp.complex128) for x in dataset], axis=0)
    all_idx = set(range(len(basis.tensors)))
    probe = _make_gradnorm_probe(basis, loss)
    groups = group_by_manifold(list(basis.tensors))  # fixed grouping (by shape)

    current = [jnp.asarray(t) for t in basis.tensors]

    def _zero_adam():
        m_state, v_state = [], []
        for _manifold, idxs in groups.items():
            pb = stack_tensors(current, list(idxs))
            m_state.append(jnp.zeros_like(pb))
            v_state.append(jnp.zeros(pb.shape, dtype=jnp.float64))
        return m_state, v_state

    trace, stages = [], []
    global_step = 0

    for s in range(1, len(unfreeze_order) + 1):
        trainable = set(unfreeze_order[:s])
        frozen = frozenset(all_idx - trainable)
        step_fn = _build_jit_adam_step(
            basis, loss, beta1=beta1, beta2=beta2, eps=eps,
            max_grad_norm=None, frozen_set=frozen if frozen else None)
        m_state, v_state = _zero_adam()

        start_step = global_step + 1
        loss_prev = None
        stage_step = 0
        trigger = "max_steps"
        L = float("nan")
        gnorm = float("inf")  # carried between grad probes

        while stage_step < max_steps_per_stage:
            stage_step += 1
            global_step += 1
            # The Adam step returns the loss for free (forward+backward already
            # done inside it). The grad-norm probe is a *second* full
            # value_and_grad and is the per-step bottleneck (~2x the step). The
            # loss-delta plateau check only needs L, so probe the Riemannian
            # grad norm only every `grad_check_every` steps (always on stage
            # step 1 so the trace/staircase has a grad value from the start).
            current, m_state, v_state, L = step_fn(
                current, m_state, v_state, batch,
                jnp.asarray(lr), jnp.asarray(global_step, dtype=jnp.int32))
            L = float(L)
            if stage_step == 1 or stage_step % grad_check_every == 0:
                _, gnorm = probe(current, batch, frozen)
            trace.append({"step": global_step, "stage": s, "n_trainable": s,
                          "loss": L, "grad_norm": gnorm})
            reason = _plateau_reason(gnorm, L, loss_prev, step=stage_step,
                                     min_steps=min_steps_per_stage,
                                     grad_tol=grad_tol, loss_tol=loss_tol)
            loss_prev = L
            if reason is not None:
                trigger = reason
                break

        # Accurate final grad norm for the summary (the breaking step may not
        # have been a probe step when grad_check_every > 1).
        _, gnorm = probe(current, batch, frozen)
        extra = stage_callback(s, current) if stage_callback is not None else {}
        stages.append(StageSummary(
            stage=s, n_trainable=s, gate_index=unfreeze_order[s - 1],
            start_step=start_step, end_step=global_step, n_steps=stage_step,
            final_loss=L, final_grad_norm=gnorm, trigger=trigger, extra=extra or {}))

    final_basis = type(basis)(m=basis.m, n=basis.n, tensors=current)
    return UnfreezeResult(basis=final_basis, trace=trace, stages=stages)
