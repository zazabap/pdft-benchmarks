#!/usr/bin/env python3
"""Quantify how 'block' a single trained QFT operator is, and render the panels.

For one trained_*.json operator (default: the canonical published trained QFT,
results/final/div2k_8q/by_basis/qft) compute:
  A  gate-collapse: which Hadamard-role gates froze to non-mixing Pauli-Z/X;
  B  dense-operator block-leakage at the emergent block (the 1-D row factor);
  C  block-leakage vs block-size sweep (the 'knee' locating the block scale);
  D  the frequency-space view (mean test-set power spectrum, untrained vs trained).
Write block_structure.json (the operator's metrics) and render four figures
(PDF + SVG) into <out-base>/figures/.

Runs on CPU (analysis is cheap; avoids GPU contention). The frequency panel needs
the DIV2K dataset and is skipped (with a warning) if it is unavailable, or with
--no-freq.

Usage:
    # canonical trained QFT -> figures in the random_seed writeup's figures dir
    python tools/render_qft_block_structure.py
    # a different operator / output location
    python tools/render_qft_block_structure.py \
        --basis results/final/div2k_8q/by_basis/qft/trained_qft.json \
        --out-base results/training/2_direct_training/random_seed/div2k_8q
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")          # analysis is CPU-cheap
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

DEFAULT_BASIS = Path("results/final/div2k_8q/by_basis/qft/trained_qft.json")
DEFAULT_OUTBASE = Path("results/training/2_direct_training/random_seed/div2k_8q")
BLOCK_SIZES = (2, 4, 8, 16, 32, 64, 128)


def _metrics(d):
    """Single-operator block-structure metrics from a trained_*.json dict."""
    import jax.numpy as jnp
    import pdft
    from pdft_benchmarks import block_structure as bs

    m, n = int(d["m"]), int(d["n"])
    g = bs.gate_summary(d["tensors"], m=m, n=n)
    T = [jnp.asarray(np.asarray(t["real"]) + 1j * np.asarray(t["imag"]),
                     dtype=jnp.complex128) for t in d["tensors"]]
    W = bs.materialize_factor(pdft.QFTBasis(m=m, n=n, tensors=T).forward_transform,
                              N=2 ** m, axis=0)
    sweep = bs.leakage_sweep(W, BLOCK_SIZES)
    return {
        "basis": d.get("key", "qft"),
        "seed": d.get("seed"),
        "m": m, "n": n,
        **g,
        "leakage_sweep": {str(int(b)): float(v) for b, v in sweep.items()},
        "eff_block": bs.effective_block_size(W),
        "eff_leakage": float(sweep[16]),
    }, W, sweep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basis", default=str(DEFAULT_BASIS),
                    help="trained_*.json operator to analyze.")
    ap.add_argument("--out-base", default=str(DEFAULT_OUTBASE),
                    help="dir whose figures/ subdir receives the panels; also "
                         "where block_structure.json is written.")
    ap.add_argument("--out", default=None,
                    help="metrics JSON path (default: <out-base>/block_structure.json).")
    ap.add_argument("--compute-only", action="store_true", default=False)
    ap.add_argument("--no-freq", action="store_true", default=False,
                    help="skip the DIV2K mean-power-spectrum panel (needs the dataset).")
    args = ap.parse_args()

    base = Path(args.out_base)
    out = Path(args.out) if args.out else base / "block_structure.json"

    d = json.loads(Path(args.basis).read_text())
    metrics, W, sweep = _metrics(d)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[block] {metrics['basis']} seed {metrics['seed']}: "
          f"n_mix={metrics['n_mix_row']}/{metrics['n_mix_col']} "
          f"block={metrics['block_row']}x{metrics['block_col']} "
          f"leak16={metrics['eff_leakage'] * 100:.3f}% "
          f"cp_active={metrics['cp_active_frac']:.3f}")
    print(f"[block] wrote {out}")

    if args.compute_only:
        return 0

    from render_qft_block_structure_figs import render_core, render_freq_spectrum
    render_core(d, base, W=W, sweep=sweep)
    if not args.no_freq:
        try:
            render_freq_spectrum(d, base)
        except Exception as e:  # noqa: BLE001 -- dataset optional; core figs already done
            print(f"[block] skipped freq-spectrum panel "
                  f"({type(e).__name__}: {e}); core figures rendered.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
