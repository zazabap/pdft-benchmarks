"""Block-structure metrics for trained QFTBasis operators.

The trained full-image QFT factors itself into a block code: ~half of the
Hadamard-role gates collapse to non-mixing Pauli-Z/X (classical block indices)
while the controlled-phase gates stay fully parameterized, so the transform
becomes block-diagonal with a rich intra-block transform (paper sec5.3).

This module provides pure-numpy measures of "how block" a trained operator is:
gate-collapse classification, dense-operator block-leakage, and a block-size
sweep. No I/O, no plotting — the CLI (tools/render_qft_block_structure.py)
loads operators and drives these.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def to_complex(t) -> np.ndarray:
    """A trained_seed tensor (dict {'real','imag'} or array) -> 2x2 complex."""
    if isinstance(t, np.ndarray):
        return t.astype(complex)
    return np.asarray(t["real"], float) + 1j * np.asarray(t["imag"], float)


def h_mixing(U: np.ndarray) -> float:
    """Mixing score 2|U00||U01| in [0,1]: 1 = Hadamard, 0 = frozen (Pauli)."""
    return float(2.0 * abs(U[0, 0]) * abs(U[0, 1]))


def classify_h(U: np.ndarray, thresh: float = 0.5) -> str:
    """Classify an H-role gate as 'H' (mixing), 'Z' or 'X' (frozen)."""
    if h_mixing(U) > thresh:
        return "H"
    return "Z" if abs(U[0, 0]) >= abs(U[0, 1]) else "X"


def gate_summary(tensors, m: int = 8, n: int = 8) -> dict:
    """Per-operator gate-collapse summary from the 72-gate tensor list."""
    G = [to_complex(t) for t in tensors]
    nH = m + n
    H, CP = G[:nH], G[nH:]
    row_tags = [classify_h(H[i]) for i in range(m)]
    col_tags = [classify_h(H[m + i]) for i in range(n)]
    mixing = [h_mixing(H[i]) for i in range(nH)]
    phis = np.array([np.angle(U[1, 1] / U[0, 0]) for U in CP])
    n_mix_row = sum(t == "H" for t in row_tags)
    n_mix_col = sum(t == "H" for t in col_tags)
    return {
        "mixing": mixing,
        "row_tags": row_tags,
        "col_tags": col_tags,
        "n_mix_row": n_mix_row,
        "n_mix_col": n_mix_col,
        "block_row": 2 ** n_mix_row,
        "block_col": 2 ** n_mix_col,
        "cp_abs_phase_median": float(np.median(np.abs(phis))),
        "cp_active_frac": float(np.mean(np.abs(1 - np.exp(1j * phis)) > 0.05)),
        "n_cp": len(CP),
    }


def block_leakage(W: np.ndarray, b: int) -> float:
    """Fraction of operator energy that crosses block boundaries at block size b.

    Group rows/cols of |W|^2 into K=N/b contiguous index-blocks, form the KxK
    block-coupling matrix, and subtract the best one-to-one (Hungarian) match.
    0 == perfectly block-structured (each input block maps to one output block);
    permutation- and scale-invariant.
    """
    N = W.shape[0]
    K = N // b
    C = (np.abs(W) ** 2).reshape(K, b, K, b).sum(axis=(1, 3))
    tot = C.sum()
    if tot <= 0:
        return 0.0
    r, c = linear_sum_assignment(-C)
    return float(1.0 - C[r, c].sum() / tot)


def leakage_sweep(W: np.ndarray, sizes=(2, 4, 8, 16, 32, 64, 128)) -> dict:
    """Block-leakage at each candidate block size (the 'knee' locates the block)."""
    return {int(b): block_leakage(W, b) for b in sizes}


def effective_block_size(W: np.ndarray,
                         sizes=(2, 4, 8, 16, 32, 64, 128, 256),
                         tol: float = 1e-6) -> int:
    """Smallest block size at which leakage vanishes (the emergent block)."""
    for b in sizes:
        if block_leakage(W, b) <= tol:
            return int(b)
    return int(W.shape[0])


def materialize_factor(forward_transform, N: int = 256, axis: int = 0) -> np.ndarray:
    """Materialize the 1-D factor of a separable 2-D transform, up to a global
    scale, via delta-image extraction (block-leakage is scale-invariant).

    forward_transform: a JAX-native callable mapping a (N,N) array to (N,N)
    (e.g. `basis.forward_transform` — do NOT numpy-wrap it, that breaks vmap).
    axis=0 -> row factor (delta at column 0, sweep the row), axis=1 -> col factor.
    """
    import jax
    import jax.numpy as jnp

    deltas = np.zeros((N, N, N), dtype=np.complex128)
    for i in range(N):
        if axis == 0:
            deltas[i, i, 0] = 1.0
        else:
            deltas[i, 0, i] = 1.0
    d = jnp.asarray(deltas)
    try:
        Ys = np.asarray(jax.vmap(forward_transform)(d))          # fast path
    except Exception:
        Ys = np.stack([np.asarray(forward_transform(d[i])) for i in range(N)])
    Ys = Ys.reshape(N, N, N)
    if axis == 0:
        # Ys[i][r,c] = Wrow[r,i]*Wcol[c,0]; pick strongest c0 of Wcol[:,0].
        c0 = int(np.argmax((np.abs(Ys) ** 2).sum(axis=(0, 1))))
        return Ys[:, :, c0].T
    r0 = int(np.argmax((np.abs(Ys) ** 2).sum(axis=(0, 2))))
    return Ys[:, r0, :].T


def _stats(xs) -> dict:
    a = np.asarray(xs, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if a.size > 1 else 0.0),
            "min": float(a.min()), "max": float(a.max())}


def _summarize(ops: list, block_sizes) -> dict:
    n = len(ops)
    tags = np.array([[t != "H" for t in o["row_tags"]]
                     + [t != "H" for t in o["col_tags"]] for o in ops], float)
    freeze_prob = tags.mean(axis=0).tolist()           # 16-vector
    hist: dict[str, int] = {}
    for o in ops:
        for key in ("block_row", "block_col"):
            hist[str(o[key])] = hist.get(str(o[key]), 0) + 1
    sweep = {str(int(b)): _stats([o["leakage_sweep"][b] for o in ops])
             for b in block_sizes}
    # Representative seed = closest to modal endpoint (n_mix_row == 4, lowest leak16).
    modal = sorted(ops, key=lambda o: (abs(o["n_mix_row"] - 4), o["eff_leakage"]))
    return {
        "n": n,
        "freeze_prob": freeze_prob,
        "n_mix_row": _stats([o["n_mix_row"] for o in ops]),
        "n_mix_col": _stats([o["n_mix_col"] for o in ops]),
        "block_size_hist": hist,
        "sweep": sweep,
        "eff_leakage": _stats([o["eff_leakage"] for o in ops]),
        "cp_active_frac": _stats([o["cp_active_frac"] for o in ops]),
        "representative_seed": int(modal[0]["seed"]),
        "representative_ordering": modal[0]["ordering"],
    }


def aggregate(ops: list, block_sizes=(2, 4, 8, 16, 32, 64, 128)) -> dict:
    """Summarize per-operator records per ordering and pooled."""
    orderings = sorted({o["ordering"] for o in ops})
    return {
        "block_sizes": [int(b) for b in block_sizes],
        "orderings": {ordr: _summarize([o for o in ops if o["ordering"] == ordr],
                                       block_sizes) for ordr in orderings},
        "pooled": _summarize(ops, block_sizes),
    }
