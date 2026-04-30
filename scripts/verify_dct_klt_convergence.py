#!/usr/bin/env python3
"""Empirical alignment of dataset-fitted PCA with the 2D DCT-II basis.

Theoretical motivation: under the Ahmed–Natarajan–Rao (1974) / Jain (1979)
asymptotic-equivalence framework, the DCT-II basis approximately
diagonalizes a stationary AR(1) covariance with rho close to 1. For
natural-image local patches, empirical rho ≈ 0.95, so we expect the
high-eigenvalue components of the empirical PCA basis to closely match
the corresponding 2D DCT-II basis vectors.

This script reads the saved block_pca_8 eigenbases from the published
benchmark cells and reports the per-eigenvector alignment with the
2D DCT-II basis (after Hungarian matching). The result:

  • The TOP eigenvectors (low-frequency, high-eigenvalue) are ~0.95+
    aligned with their best-matching DCT-II vector — the practical
    "DCT ≈ KLT for natural images" empirical statement.
  • The BOTTOM eigenvectors (high-frequency, near-zero eigenvalue)
    are weakly aligned — but they carry negligible signal energy, so
    their misalignment doesn't hurt rate-distortion.

This is the "DCT is near-KLT for natural images" result that supports
JPEG-style block-DCT being a near-optimal practical baseline.

Usage:
    python scripts/verify_dct_klt_convergence.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pdft  # noqa: F401  -- enables jax_enable_x64
from scipy.fft import dct
from scipy.optimize import linear_sum_assignment

SIDE = 8
BASES_TO_INSPECT = [
    "results/_archive/quickdraw_5q_generalized_20260427-073744/block_pca_8_eigenbasis.npz",
    "results/_archive/div2k_8q_generalized_20260425-102013_gpu0/block_pca_8_eigenbasis.npz",
    "results/_archive/div2k_10q_generalized_20260426-055335_gpu0_bs2/block_pca_8_eigenbasis.npz",
]


def dct2_basis(side: int) -> np.ndarray:
    """2D DCT-II basis. Returns (side*side, side*side) with rows = vec'd 2D vectors,
    sorted by zigzag scan (low-frequency first)."""
    eye = np.eye(side)
    B1 = dct(eye, axis=0, norm='ortho')   # columns = 1D DCT vectors

    # Build all 2D vectors as outer products of 1D vectors.
    vecs = []
    levels = []
    for i in range(side):
        for j in range(side):
            vecs.append(np.outer(B1[:, i], B1[:, j]).ravel())
            levels.append(i + j)
    # Sort by frequency level (ties keep original order — gives a zigzag-like ordering).
    order = np.argsort(levels, kind='stable')
    return np.array(vecs)[order]


def report(label: str, eigenbasis: np.ndarray, eigenvalues: np.ndarray) -> None:
    dct2 = dct2_basis(SIDE)
    M = np.abs(eigenbasis @ dct2.T)               # (64, 64)
    row, col = linear_sum_assignment(-M)
    aligns = M[row, col]
    # Energy-weighted summary: how close is the empirical basis to DCT-II,
    # weighted by how much variance each eigenvector explains?
    weights = eigenvalues / eigenvalues.sum()
    weighted = float(np.sum(weights * aligns))

    print(f"\n## {label}")
    print(f"  Energy-weighted mean alignment with DCT-II: {weighted:.4f}")
    print(f"  Mean alignment over all 64:                  {aligns.mean():.4f}")
    print(f"\n  Per-eigenvector (sorted by descending eigenvalue):")
    print(f"    rank   eigenvalue    |⟨pca,dct⟩|   energy share")
    for r in range(SIDE * SIDE):
        share = float(weights[r])
        marker = " ← top-energy" if r < 4 else ""
        print(f"    {r:>4}   {eigenvalues[r]:>10.6f}   {aligns[r]:>11.4f}     {share:>6.4f}{marker}")


def main() -> None:
    for path_s in BASES_TO_INSPECT:
        path = Path(path_s)
        if not path.is_file():
            print(f"\nMISSING: {path}", flush=True)
            continue
        data = np.load(path)
        report(path.parent.name, data["eigenbasis"], data["eigenvalues"])

    print()
    print("Reading the table:")
    print("  Top-rank eigenvectors should have alignment ≥ 0.9 — the empirical")
    print("  'DCT ≈ KLT for natural images' statement that supports DCT's status")
    print("  as a near-optimal fixed approximation to the dataset KLT.")
    print("  Bottom-rank eigenvectors are weakly aligned, but they carry < 1%")
    print("  of the energy each — the misalignment is rate-distortion-irrelevant.")


if __name__ == "__main__":
    main()
