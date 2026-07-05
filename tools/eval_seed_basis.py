#!/usr/bin/env python3
"""Load a saved per-seed trained operator and compute MSE / PSNR on the fixed
test set — the "use" half of qft_seed_sweep.py's trained_seed_*.json.

Reads trained_seed_NNN.json ({m, n, tensors}), rebuilds the QFTBasis, evaluates
on the canonical seed-42 test split (identical to what the sweep scores on), and
prints MSE + PSNR at the requested keep ratios. Multiple --seed values evaluate
a batch; --json emits machine-readable output.

Usage:
    python tools/eval_seed_basis.py --ordering bg --seed 3
    python tools/eval_seed_basis.py --ordering rl --seed 7,12,20 --ratios 0.1,0.2
    python tools/eval_seed_basis.py --basis path/to/trained_seed_003.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_BASE = Path("results/training/2_direct_training/random_seed/div2k_8q")


def _parse_list(spec, cast):
    return [cast(x) for x in spec.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--dataset", default="div2k_8q",
                    choices=["div2k_8q", "quickdraw_5q", "tuberlin_8q"])
    ap.add_argument("--base", default=str(DEFAULT_BASE),
                    help="random_seed/<dataset> dir holding _runs/<ordering>/.")
    ap.add_argument("--ordering", default="bg", choices=["bg", "lr", "rl"])
    ap.add_argument("--seed", default=None, help="seed or comma list, e.g. 3,7,12.")
    ap.add_argument("--basis", default=None,
                    help="explicit path to a trained_seed_*.json (overrides "
                         "--ordering/--seed).")
    ap.add_argument("--ratios", default="0.05,0.1,0.15,0.2")
    ap.add_argument("--json", action="store_true", default=False)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax.numpy as jnp
    import numpy as np
    import pdft
    import pdft.io  # noqa: F401
    from pdft_benchmarks import datasets as ds_mod
    from pdft_benchmarks.evaluation import evaluate_basis_shared

    DATASET_CFG = {"quickdraw_5q": (5, "load_quickdraw", "img_size"),
                   "div2k_8q": (8, "load_div2k", "size"),
                   "tuberlin_8q": (8, "load_tuberlin", "size")}
    m_q, loader_name, size_kw = DATASET_CFG[args.dataset]
    ratios = tuple(_parse_list(args.ratios, float))
    base = Path(args.base)

    # Locate the trained_seed files to evaluate.
    paths: list[Path] = []
    if args.basis:
        paths = [Path(args.basis)]
    else:
        for s in _parse_list(args.seed, int):
            paths.append(base / "_runs" / args.ordering / f"trained_seed_{s:03d}.json")
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"[eval] missing operator file(s): {missing}\n"
              f"  (only seeds finished AFTER basis-saving was enabled have one; "
              f"regenerate with: qft_seed_sweep.py --seeds <s> --force)", file=sys.stderr)
        return 2

    # Fixed test set: canonical seed-42 split, identical to the sweep.
    _, test_imgs = getattr(ds_mod, loader_name)(
        n_train=500, n_test=50, seed=42, **{size_kw: 2 ** m_q})

    out = []
    for p in paths:
        d = json.loads(p.read_text())
        tensors = [jnp.asarray(np.array(t["real"], dtype=np.float64)
                               + 1j * np.array(t["imag"], dtype=np.float64),
                               dtype=jnp.complex128) for t in d["tensors"]]
        basis = pdft.QFTBasis(m=int(d["m"]), n=int(d["n"]), tensors=tensors)
        metrics, _ = evaluate_basis_shared(basis, test_imgs, keep_ratios=ratios)
        row = {"file": str(p), "ordering": d.get("ordering"), "seed": d.get("seed"),
               "metrics": {str(r): {"mse": metrics[str(r)]["mean_mse"],
                                    "psnr": metrics[str(r)]["mean_psnr"]}
                           for r in ratios}}
        out.append(row)
        if not args.json:
            print(f"{d.get('ordering')} seed {d.get('seed')}  ({p.name}):")
            for r in ratios:
                mm = metrics[str(r)]
                print(f"   rho={r:<5}  MSE={mm['mean_mse']:.6g}  PSNR={mm['mean_psnr']:.3f} dB")
    if args.json:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
