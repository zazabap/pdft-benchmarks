"""Tests for the QFT block-structure metrics (src/pdft_benchmarks/block_structure.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pdft_benchmarks import block_structure as bs

H2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
X2 = np.array([[0, 1], [1, 0]], dtype=complex)


def test_h_mixing_hadamard_is_one():
    assert bs.h_mixing(H2) == pytest.approx(1.0, abs=1e-9)


def test_h_mixing_pauli_is_zero():
    assert bs.h_mixing(Z2) == pytest.approx(0.0, abs=1e-9)
    assert bs.h_mixing(X2) == pytest.approx(0.0, abs=1e-9)


def test_classify_h():
    assert bs.classify_h(H2) == "H"
    assert bs.classify_h(Z2) == "Z"
    assert bs.classify_h(X2) == "X"


def test_to_complex_roundtrip():
    t = {"real": [[0.0, 1.0], [1.0, 0.0]], "imag": [[0.0, 0.0], [0.0, 0.0]]}
    np.testing.assert_allclose(bs.to_complex(t), X2)


def test_gate_summary_on_real_seed():
    p = Path("results/training/2_direct_training/random_seed/div2k_8q/_runs/bg/trained_seed_050.json")
    if not p.exists():
        pytest.skip("trained_seed_050 not present")
    tensors = json.loads(p.read_text())["tensors"]
    g = bs.gate_summary(tensors, m=8, n=8)
    assert len(g["mixing"]) == 16
    assert g["n_cp"] == 56
    assert 2 <= g["n_mix_row"] <= 6 and 2 <= g["n_mix_col"] <= 6
    assert g["block_row"] == 2 ** g["n_mix_row"]
    assert g["cp_active_frac"] > 0.9  # CP gates stay live


import scipy.linalg


def _block_diag_matrix(n_blocks, b, seed=0):
    rng = np.random.default_rng(seed)
    blocks = [rng.standard_normal((b, b)) + 1j * rng.standard_normal((b, b))
              for _ in range(n_blocks)]
    return scipy.linalg.block_diag(*blocks)


def test_block_leakage_perfect_block():
    W = _block_diag_matrix(16, 16)            # 256x256, 16 diagonal 16-blocks
    assert bs.block_leakage(W, 16) < 1e-12
    assert bs.block_leakage(W, 8) > 0.1       # finer probe sees cross-coupling


def test_block_leakage_permutation_invariant():
    # Permuting which output-block each input-block maps to (X-like relabeling)
    # must not raise leakage: Hungarian matching absorbs it.
    W = _block_diag_matrix(16, 16)
    perm = np.concatenate([np.roll(np.arange(16), 1).reshape(16, 1) * 16 + j
                           for j in range(16)], axis=1).ravel()  # block-permute rows
    Wp = W[perm, :]
    assert bs.block_leakage(Wp, 16) < 1e-12


def test_leakage_sweep_and_eff_block():
    W = _block_diag_matrix(16, 16)
    sweep = bs.leakage_sweep(W)
    assert sweep[16] < 1e-12 and sweep[8] > 0.1
    assert bs.effective_block_size(W) == 16


def test_materialize_factor_untrained_qft_is_not_block():
    pdft = pytest.importorskip("pdft")
    q = pdft.QFTBasis(m=8, n=8)
    W = bs.materialize_factor(q.forward_transform)   # pass the JAX-native callable
    assert W.shape == (256, 256)
    assert bs.block_leakage(W, 16) > 0.5      # closed-form QFT: no block structure
