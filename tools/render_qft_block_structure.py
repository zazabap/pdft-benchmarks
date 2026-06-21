#!/usr/bin/env python3
"""Quantify how 'block' the trained random-seed QFTBasis operators are.

For every trained_seed_*.json operator (100 seeds x bg/lr/rl), compute:
  A  gate-collapse: which Hadamard-role gates froze to Pauli-Z/X, CP activity;
  B  dense-operator block-leakage at the emergent block (the row 1-D factor);
  C  block-leakage vs block-size sweep (the 'knee' locating the block scale).
Aggregate across seeds, write block_structure.json, and render three figures
(PDF + SVG): block_gate_collapse, block_operator_heatmap, block_leakage_sweep.

Runs on CPU (analysis is cheap; avoids GPU contention). Re-render from the
saved JSON with --from-json (no recompute), mirroring render_init_distribution.py.

Usage:
    # compute + render (writes JSON + figures)
    python tools/render_qft_block_structure.py \
        --base results/training/2_direct_training/random_seed/div2k_8q
    # just recompute the JSON (no figures)
    python tools/render_qft_block_structure.py --compute-only \
        --base ... --orderings bg --seeds 50,51 --out /tmp/bs.json
    # re-render figures from the committed JSON (no recompute)
    python tools/render_qft_block_structure.py --from-json --base ...
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

DEFAULT_BASE = Path("results/training/2_direct_training/random_seed/div2k_8q")
ALL_ORDERINGS = ("bg", "lr", "rl")
BLOCK_SIZES = (2, 4, 8, 16, 32, 64, 128)


def _parse_seeds(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _operator_records(base, orderings, seeds):
    """Load each trained_seed operator and compute its per-op metric record."""
    import jax.numpy as jnp
    import pdft
    from pdft_benchmarks import block_structure as bs

    records = []
    for ordr in orderings:
        rundir = base / "_runs" / ordr
        files = (sorted(rundir.glob("trained_seed_*.json")) if seeds is None
                 else [rundir / f"trained_seed_{s:03d}.json" for s in seeds])
        for p in files:
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            tensors = d["tensors"]
            g = bs.gate_summary(tensors, m=int(d["m"]), n=int(d["n"]))
            T = [jnp.asarray(np.asarray(t["real"]) + 1j * np.asarray(t["imag"]),
                             dtype=jnp.complex128) for t in tensors]
            basis = pdft.QFTBasis(m=int(d["m"]), n=int(d["n"]), tensors=T)
            W = bs.materialize_factor(basis.forward_transform,
                                      N=2 ** int(d["m"]), axis=0)
            sweep = bs.leakage_sweep(W, BLOCK_SIZES)
            rec = {"ordering": ordr, "seed": int(d["seed"]), **g,
                   "leakage_sweep": sweep,
                   "eff_block": bs.effective_block_size(W),
                   "eff_leakage": float(sweep[16])}
            records.append(rec)
            print(f"[block] {ordr} seed {d['seed']:>3}: "
                  f"n_mix={g['n_mix_row']}/{g['n_mix_col']} "
                  f"eff_block={rec['eff_block']} leak16={rec['eff_leakage']:.4f}")
    return records


def _compute(args):
    from pdft_benchmarks import block_structure as bs
    base = Path(args.base)
    orderings = args.orderings.split(",") if args.orderings else list(ALL_ORDERINGS)
    seeds = _parse_seeds(args.seeds)
    recs = _operator_records(base, orderings, seeds)
    if not recs:
        raise SystemExit("[block] no operators found")
    agg = bs.aggregate(recs, BLOCK_SIZES)
    agg["dataset"] = base.name
    agg["per_op"] = recs
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--orderings", default=None, help="comma list, default bg,lr,rl")
    ap.add_argument("--seeds", default=None, help="e.g. 1-100 or 50,51 (default: all)")
    ap.add_argument("--out", default=None, help="JSON path (default: <base>/block_structure.json)")
    ap.add_argument("--compute-only", action="store_true", default=False)
    ap.add_argument("--from-json", action="store_true", default=False)
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out) if args.out else base / "block_structure.json"

    if args.from_json:
        agg = json.loads(out.read_text())
    else:
        agg = _compute(args)
        out.write_text(json.dumps(agg, indent=2))
        print(f"[block] wrote {out}")

    if args.compute_only:
        return 0

    from render_qft_block_structure_figs import render_all  # Task 5
    render_all(agg, base)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
