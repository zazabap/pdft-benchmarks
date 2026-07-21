#!/usr/bin/env python3
"""Re-score the saved DCT-IV seed-study operators at a different keep-ratio set.

The seed sweep (`experiments/dct4_seed_sweep.py`) saves every trained operator
to `_runs/trained_seed_<NNN>.json`. Changing the *evaluation* keep ratios rho
does NOT require retraining — the training objective (top-20% MSE) is unchanged,
so we reconstruct each saved `DCT4Basis` and re-evaluate it on the fixed
seed-42 test set at the new ratios, overwriting the `psnr` block of each cell
`_runs/seed_<NNN>.json` in place. This is the whole reason the operators are
saved.

Reconstruction mirrors `dct4_random_basis`: a canonical `DCT4Basis(m,n)` gives
the gate topology + `code`/`inv_code`; the stored complex tensors replace the
gate values (DCT-IV gates are real-orthogonal, stored complex with ~0 imag).

Usage:
    PYTHONPATH=<latest pdft src> python tools/reeval_dct4_seed_ratios.py \
        --base results/training/2_direct_training/random_seed/dct_div2k_8q \
        --seeds 1-100 --ratios 0.01,0.05,0.10,0.20 --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _atomic_write_json(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="random_seed/dct_<dataset> dir holding _runs/.")
    ap.add_argument("--seeds", default="1-100")
    ap.add_argument("--ratios", default="0.01,0.05,0.10,0.20",
                    help="Comma-separated keep ratios to score at.")
    ap.add_argument("--dataset", default="div2k_8q")
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU index; omit for CPU (eval is light).")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pdft
    import pdft.io  # noqa: F401 — needed by evaluate_basis_shared
    from pdft_benchmarks.datasets.div2k import load_div2k
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.presets import get_preset

    if not hasattr(pdft, "DCT4Basis"):
        print(f"[reeval] FATAL: pdft at {pdft.__file__} lacks DCT4Basis; put "
              f"latest pdft src first on PYTHONPATH.", file=sys.stderr)
        return 3

    base_dir = Path(args.base)
    cell_dir = base_dir / "_runs"
    seeds = _parse_seeds(args.seeds)
    ratios = tuple(float(x) for x in args.ratios.split(","))
    m = n = 8

    preset = get_preset(args.dataset, "generalized")
    # Same fixed seed-42 test set used for training-time scoring.
    _, test_imgs = load_div2k(n_train=preset.n_train, n_test=preset.n_test,
                              seed=42, size=2 ** m)

    chosen = jax.devices()[0]
    print(f"[reeval] device={chosen} platform={chosen.platform!r} "
          f"pdft={pdft.__version__} ratios={ratios}")

    # Canonical basis: reused for code/inv_code so the circuit compiles once.
    canon = pdft.DCT4Basis(m=m, n=n)

    def reconstruct(tensors_json):
        tensors = [jnp.asarray(np.asarray(t["real"]) + 1j * np.asarray(t["imag"]),
                               dtype=jnp.complex128) for t in tensors_json]
        return pdft.DCT4Basis(m=m, n=n, tensors=tensors,
                              code=canon.code, inv_code=canon.inv_code)

    done, missing, max_drift = 0, [], 0.0
    for s in seeds:
        op_path = cell_dir / f"trained_seed_{s:03d}.json"
        cell_path = cell_dir / f"seed_{s:03d}.json"
        if not op_path.exists() or not cell_path.exists():
            missing.append(s)
            continue
        op = json.loads(op_path.read_text())
        basis = reconstruct(op["tensors"])
        metrics, _ = evaluate_basis_shared(basis, test_imgs, keep_ratios=ratios)
        new_psnr = {f"{r}": float(metrics[str(r)]["mean_psnr"]) for r in ratios}

        cell = json.loads(cell_path.read_text())
        old_psnr = cell.get("psnr", {})
        # Sanity: a ratio present in both old and new must match (same operator,
        # same test set, same eval) — drift flags a reconstruction error.
        for k in set(new_psnr) & set(old_psnr):
            max_drift = max(max_drift, abs(new_psnr[k] - old_psnr[k]))
        cell["psnr"] = new_psnr
        cell["keep_ratios"] = list(ratios)
        _atomic_write_json(cell_path, cell)
        done += 1
        print(f"[reeval] seed_{s:03d}: PSNR@.20={new_psnr.get('0.2', float('nan')):.3f} "
              f"@.01={new_psnr.get('0.01', float('nan')):.3f}")

    print(f"[reeval] re-scored {done} cells; missing={missing}; "
          f"max overlap drift={max_drift:.2e} dB")
    if max_drift > 1e-3:
        print(f"[reeval] WARN: overlap drift {max_drift:.2e} dB exceeds 1e-3 — "
              f"check reconstruction.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
