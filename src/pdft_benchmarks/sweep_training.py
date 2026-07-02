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
