"""DMRG-style environment-sweep training of the controlled O(2)-twiddle DCT-IV.

Used by the sweep-training study (results/training/5_sweep_training_dct):
visit gates one at a time (Gauss-Seidel); the derivative of the top-k
reconstruction loss w.r.t. a gate tensor is its *environment* (the network
contracted with everything except that gate); the manifold point minimizing the
linearized loss Re<E, G> is the SVD polar factor `-U V^T` (closed-form angle
for the Delta-sign phase gate). A backtracking acceptance check makes the
fixed-batch loss monotone non-increasing — no learning rate, no momentum, no
schedule.

Gate classification mirrors ``disturbance.disturb_controlled_dct4``:
  - (2,2,2,2) mirror-Q/R gate            -> "o4"   (polar on O(4))
  - (2,2) Delta-sign (row 0 == [1,1])    -> "phase" (closed-form angle)
  - other (2,2) (branch-H / R_y / CRY)   -> "o2"   (polar on O(2))

The engine (`sweep_train`) is I/O-free: loss closures are injected, per-sweep
side effects (PSNR eval, checkpointing) happen in an ``on_sweep_end`` callback.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def sweep_order(n_gates: int, order: str) -> list[int]:
    """Gate-visit order: ``fwd`` = emission order of dct4_code, ``rev`` = reversed."""
    if order == "fwd":
        return list(range(n_gates))
    if order == "rev":
        return list(range(n_gates - 1, -1, -1))
    raise ValueError(f"order must be 'fwd' or 'rev', got {order!r}")


def _is_delta_sign(a: np.ndarray) -> bool:
    return a.shape == (2, 2) and bool(np.allclose(a[0], np.ones(2), atol=1e-9))


def classify_gate(t) -> str:
    a = np.asarray(t)
    if a.shape == (2, 2, 2, 2):
        return "o4"
    if a.shape == (2, 2):
        return "phase" if _is_delta_sign(a) else "o2"
    raise ValueError(f"unexpected gate tensor shape {a.shape}")


def _polar(m: np.ndarray) -> np.ndarray:
    """Nearest orthogonal matrix (polar factor U V^T via SVD)."""
    u, _, vt = np.linalg.svd(m)
    return u @ vt


def polar_candidate(env) -> np.ndarray:
    """Minimizer of the linearized loss <Re(env), G>_F over O(d).

    Taking Re() IS the restriction of the complex environment to the
    real-orthogonal manifold (canonical real inner product), not a shortcut.
    """
    e = np.real(np.asarray(env)).astype(np.float64)
    d = 4 if e.shape == (2, 2, 2, 2) else 2
    return (-_polar(e.reshape(d, d))).reshape(e.shape)


def phase_candidate(env) -> float:
    """Minimizer of Re[conj(E11) e^{i phi}]: phi* = pi + arg(E11) = angle(-E11)."""
    e = np.asarray(env)
    return float(np.angle(-e[1, 1]))


def _wrap_angle(a: float) -> float:
    return float(np.angle(np.exp(1j * a)))


def interpolated(gate, candidate, kind: str, t: float) -> np.ndarray:
    """Backtracking point at fraction ``t`` between the gate and its candidate.

    O-gates: polar((1-t) G + t G*) — retracts the chord back onto O(d).
    Phase: angle interpolation phi(t) = phi_0 + t wrap(phi* - phi_0), where
    ``candidate`` is the target angle (a float).
    """
    if kind == "phase":
        from pdft.circuit.builder import controlled_phase_diag

        phi0 = float(np.angle(np.asarray(gate)[1, 1]))
        return np.asarray(controlled_phase_diag(phi0 + t * _wrap_angle(candidate - phi0)))
    g0 = np.real(np.asarray(gate)).astype(np.float64)
    g1 = np.real(np.asarray(candidate)).astype(np.float64)
    d = 4 if g0.shape == (2, 2, 2, 2) else 2
    return _polar(((1.0 - t) * g0 + t * g1).reshape(d, d)).reshape(g0.shape)


@dataclass
class Visit:
    sweep: int
    pos: int
    gate: int
    kind: str
    loss_before: float
    loss_after: float
    t: float | None
    accepted: bool
    skip_reason: str | None = None


@dataclass
class SweepStats:
    sweep: int
    loss_end: float
    n_accepted: int
    n_skipped: int
    wall_s: float


@dataclass
class SweepResult:
    tensors: list
    visits: list[Visit] = field(default_factory=list)
    sweeps: list[SweepStats] = field(default_factory=list)
    converged: bool = False
    final_loss: float = float("nan")

    @property
    def n_accepted_total(self) -> int:
        return sum(s.n_accepted for s in self.sweeps)


def sweep_train(
    tensors,
    value_and_grad_fn,
    loss_fn,
    *,
    order: str = "fwd",
    max_sweeps: int = 20,
    rel_tol: float = 1e-5,
    backtrack_ts: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125),
    env_tol: float = 1e-12,
    max_visits: int | None = None,
    on_sweep_end=None,
    start_sweep: int = 1,
    visits=None,
    sweeps=None,
) -> SweepResult:
    """Gauss-Seidel environment sweeps until plateau or ``max_sweeps``.

    ``value_and_grad_fn(tensors) -> (loss, grads)`` and
    ``loss_fn(tensors) -> loss`` are injected jitted closures over a fixed
    batch. ``on_sweep_end(sweep_idx, tensors, stats)`` runs after each sweep
    (PSNR eval / checkpointing). ``start_sweep``/``visits``/``sweeps`` support
    resuming from a checkpoint; when provided, the lists are extended IN PLACE
    (callers may mirror appends). ``max_visits`` truncates each sweep (debug /
    smoke only). The fixed-batch loss is monotone non-increasing by
    construction (strict-decrease acceptance).

    Caveats:
      - Resuming with ``start_sweep > max_sweeps`` runs zero sweeps and
        returns the checkpoint tensors unchanged, but reports
        ``converged=False`` — the caller must preserve the checkpointed
        converged state itself.
      - ``l0`` comes from ``value_and_grad_fn`` and ``l1`` from ``loss_fn``
        — two separately-compiled XLA programs that can differ at ~1e-16 on
        identical inputs — so with ``rel_tol=0`` noise-level accepts can
        defer termination all the way to ``max_sweeps``. Irrelevant at the
        default tolerances.
    """
    import jax.numpy as jnp

    tensors = [jnp.asarray(t, dtype=jnp.complex128) for t in tensors]
    kinds = [classify_gate(np.asarray(t)) for t in tensors]
    idx = sweep_order(len(tensors), order)
    if max_visits is not None:
        idx = idx[:max_visits]
    all_visits = visits if visits is not None else []
    all_sweeps = sweeps if sweeps is not None else []
    converged = False

    for s in range(start_sweep, max_sweeps + 1):
        t_sweep = time.perf_counter()
        n_acc = n_skip = 0
        loss_start = None
        cached = None  # (l0_raw, grads) from the last skipped visit
        for pos, gi in enumerate(idx):
            # A skipped visit leaves the tensors unchanged, so the previous
            # value_and_grad result is still valid (the closures are
            # deterministic); reuse it instead of recomputing.
            l0_raw, grads = (cached if cached is not None
                             else value_and_grad_fn(tensors))
            l0 = float(l0_raw)
            if loss_start is None:
                loss_start = l0
            env = np.conj(np.asarray(grads[gi]))
            kind = kinds[gi]
            env_mag = (abs(env[1, 1]) if kind == "phase"
                       else float(np.abs(np.real(env)).max()))
            if env_mag < env_tol:
                all_visits.append(Visit(s, pos, gi, kind, l0, l0, None, False,
                                        "zero_env"))
                n_skip += 1
                cached = (l0_raw, grads)
                continue
            cand = phase_candidate(env) if kind == "phase" else polar_candidate(env)
            accepted = False
            for t in backtrack_ts:
                gnew = interpolated(np.asarray(tensors[gi]), cand, kind, t)
                if np.allclose(gnew, np.asarray(tensors[gi]), atol=1e-14):
                    break  # no-op candidate: smaller t only gets closer; skip
                trial = list(tensors)
                trial[gi] = jnp.asarray(gnew, dtype=jnp.complex128)
                l1 = float(loss_fn(trial))
                if l1 < l0:
                    tensors = trial
                    all_visits.append(Visit(s, pos, gi, kind, l0, l1, float(t),
                                            True))
                    n_acc += 1
                    accepted = True
                    break
            if accepted:
                cached = None
            else:
                all_visits.append(Visit(s, pos, gi, kind, l0, l0, None, False,
                                        "no_decrease"))
                n_skip += 1
                cached = (l0_raw, grads)
        loss_end = float(loss_fn(tensors))
        stats = SweepStats(s, loss_end, n_acc, n_skip,
                           time.perf_counter() - t_sweep)
        all_sweeps.append(stats)
        if on_sweep_end is not None:
            on_sweep_end(s, tensors, stats)
        if loss_start is None:  # zero visits (max_visits=0 / no gates)
            loss_start = loss_end
        rel = (loss_start - loss_end) / max(abs(loss_start), 1e-30)
        if n_acc == 0 or rel < rel_tol:
            converged = True
            break

    return SweepResult(
        tensors=tensors, visits=all_visits, sweeps=all_sweeps,
        converged=converged,
        final_loss=(all_sweeps[-1].loss_end if all_sweeps
                    else float(loss_fn(tensors))))
