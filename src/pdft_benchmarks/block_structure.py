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
