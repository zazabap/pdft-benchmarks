#!/usr/bin/env python3
"""Full-parameter training of a chosen seed-init for a set of circuit families.

A counterpart to experiments/qft_seed_sweep.py (which trains the *progressive
gate-unfreeze* QFT), this driver trains the STANDARD full-parameter objective
(all gate parameters at once, top-10% MSE — the paper's headline protocol) for
several families from a SINGLE random-initialisation seed, and writes one
`by_basis`-style cell per basis so the result drops straight into the paper's
Table 2 / Figure 4 / Figure 5 pipeline.

Init is uniform across families via `family_random_basis(family, m, n, seed)`
(Haar U(2)/phase for QFT; constructor-seeded for EntangledQFT/TEBD/MERA; Haar
U/SO for Rich). The held-out TEST set is the canonical seed-42 split, identical
to the paper, so the only thing the seed moves is the trained operator.

Basis keys (this driver's vocabulary):
    qft, entangled_qft, tebd, mera   — bare full-image families
    rich_full                        — bare full-image RichBasis (U(4) gates)
    rich_8                           — RichBasis wrapped into fixed 8×8 blocks
                                       (inner 3-qubit RichBasis, seed-98 init)

MERA is unavailable on QuickDraw (m+n=10 is not a power of two); pass a basis
list that omits it there.

Usage:
    # DIV2K-8q, all six, ~1008 steps, no early stop, one GPU:
    python experiments/seed98_full_train.py --gpu 0 --dataset div2k_8q --seed 98 \
        --bases qft,entangled_qft,tebd,mera,rich_full,rich_8 \
        --epochs 112 --no-early-stop

    # QuickDraw-5q, five (no MERA), early stopping on:
    python experiments/seed98_full_train.py --gpu 1 --dataset quickdraw_5q --seed 98 \
        --bases qft,entangled_qft,tebd,rich_full,rich_8 --epochs 112
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# random_seed/<dataset> -> results/structure/<experiment> mapping.
DATASET_CFG = {
    "div2k_8q": dict(m=8, loader="load_div2k", size_kw="size",
                     exp="div2k_8q_pca_vs_block_dct"),
    "quickdraw_5q": dict(m=5, loader="load_quickdraw", size_kw="img_size",
                         exp="quickdraw_pca_vs_block_dct"),
}
DEFAULT_RATIOS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)


def _parse_ratios(spec):
    return tuple(float(x) for x in spec.split(",") if x.strip())


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def build_basis(key: str, m: int, n: int, seed: int):
    """Seed-`seed` initialisation for a driver basis key (fresh each call)."""
    import pdft
    from pdft_benchmarks.bases import family_random_basis
    if key in ("qft", "entangled_qft", "tebd", "mera"):
        return family_random_basis(key, m, n, seed)
    if key == "rich_full":
        return family_random_basis("rich", m, n, seed)
    if key == "rich_8":
        # Fixed 8x8 blocks: inner 3-qubit RichBasis (seed-init), block-wrapped.
        inner = family_random_basis("rich", 3, 3, seed)
        return pdft.BlockedBasis(inner=inner, block_log_m=m - 3, block_log_n=n - 3)
    raise ValueError(f"unknown basis key {key!r}")


VALID_KEYS = ("qft", "entangled_qft", "tebd", "mera", "rich_full", "rich_8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU index; sets CUDA_VISIBLE_DEVICES before JAX import.")
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CFG))
    ap.add_argument("--bases", required=True,
                    help=f"Comma-separated subset of {VALID_KEYS}.")
    ap.add_argument("--seed", type=int, default=98,
                    help="Init + training-RNG seed (test set stays seed-42).")
    ap.add_argument("--epochs", type=int, default=112,
                    help="Epoch budget. DIV2K: 112 ≈ 1008 steps (9 steps/epoch).")
    ap.add_argument("--no-early-stop", action="store_true", default=False,
                    help="Disable early stopping (train the full epoch budget).")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override preset.batch_size (e.g. lower for rich_full).")
    ap.add_argument("--keep-ratios", default=",".join(str(r) for r in DEFAULT_RATIOS))
    ap.add_argument("--out", default=None,
                    help="Output by_basis dir (default: results/structure/<exp>/"
                         "by_basis_seed<seed>).")
    ap.add_argument("--force", action="store_true", default=False,
                    help="Re-run bases whose cell already exists.")
    args = ap.parse_args()

    bad = [b for b in args.bases.split(",") if b.strip() and b.strip() not in VALID_KEYS]
    if bad:
        print(f"unknown basis key(s): {bad}; choices: {VALID_KEYS}", file=sys.stderr)
        return 2

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    cfg = DATASET_CFG[args.dataset]
    m = n = cfg["m"]
    bases = [b.strip() for b in args.bases.split(",") if b.strip()]
    ratios = _parse_ratios(args.keep_ratios)
    out_base = Path(args.out) if args.out else \
        REPO / "results/structure" / cfg["exp"] / f"by_basis_seed{args.seed}"
    out_base.mkdir(parents=True, exist_ok=True)

    import dataclasses
    import jax
    import numpy as np  # noqa: F401
    import pdft  # noqa: F401
    import pdft.io  # noqa: F401  (operator (de)serialisation registry)
    from pdft_benchmarks import datasets as ds_mod
    from pdft_benchmarks.presets import get_preset
    from pdft_benchmarks._training import train_one_basis_batched
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.experiment_utils import git_sha, serialize_tensors

    device = jax.devices()[0]
    preset = get_preset(args.dataset, "generalized")
    overrides = dict(seed=args.seed, epochs=args.epochs)
    if args.no_early_stop:
        overrides["early_stopping_patience"] = 10 ** 9
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    preset = dataclasses.replace(preset, **overrides)

    # Fixed canonical test set (seed 42), identical to the paper / headline cells.
    train_pool, test_imgs = getattr(ds_mod, cfg["loader"])(
        n_train=preset.n_train, n_test=preset.n_test, seed=42,
        **{cfg["size_kw"]: 2 ** m})

    print(f"[seed98] dataset={args.dataset} m=n={m} seed={args.seed} "
          f"epochs={preset.epochs} early_stop="
          f"{'off' if args.no_early_stop else preset.early_stopping_patience} "
          f"batch={preset.batch_size} device={device} bases={bases}")
    print(f"[seed98] out={out_base}")

    env = {
        "dataset": args.dataset, "m": m, "n": n, "seed": args.seed,
        "init": "family_random_basis", "objective": "top10pct_mse",
        "epochs": preset.epochs, "no_early_stop": args.no_early_stop,
        "batch_size": preset.batch_size, "lr_peak": preset.lr_peak,
        "keep_ratios": list(ratios), "test_seed": 42,
        "n_train": preset.n_train, "n_test": preset.n_test,
        "git_sha": git_sha(), "device": str(device),
    }
    _atomic_write_json(out_base / "env.json", env)

    for key in bases:
        cell_dir = out_base / key
        cell_path = cell_dir / "metrics.json"
        if cell_path.exists() and not args.force:
            print(f"[seed98] skip {key} (cell exists; --force to redo)")
            continue
        t0 = time.perf_counter()
        factory = (lambda k=key: build_basis(k, m, n, args.seed))
        res = train_one_basis_batched(factory, train_pool, preset, device=device)
        metrics, _ = evaluate_basis_shared(res.basis, test_imgs, keep_ratios=ratios)
        elapsed = time.perf_counter() - t0

        cell = {key: {
            "metrics": metrics,
            "seed": args.seed, "init": "family_random_basis",
            "objective": "top10pct_mse",
            "steps": res.steps, "epochs_completed": res.epochs_completed,
            "elapsed_seconds": round(elapsed, 1),
        }}
        _atomic_write_json(cell_path, cell)
        _atomic_write_json(cell_dir / "env.json", env)
        _atomic_write_json(cell_dir / f"trained_{key}.json", {
            "dataset": args.dataset, "key": key, "m": int(res.basis.m),
            "n": int(res.basis.n), "seed": args.seed,
            "tensors": serialize_tensors(res.basis.tensors),
        })
        _atomic_write_json(cell_dir / "loss_history" / f"{key}_loss.json", {
            "loss_history": res.loss_history, "val_history": res.val_history,
            "steps": res.steps, "epochs_completed": res.epochs_completed,
        })
        psnr20 = metrics.get("0.2", {}).get("mean_psnr")
        print(f"[seed98] {key}: {res.steps} steps, {elapsed:.0f}s, "
              f"PSNR@.20={psnr20:.3f} dB" if psnr20 is not None
              else f"[seed98] {key}: {res.steps} steps, {elapsed:.0f}s")

    print(f"[seed98] done. cells in {out_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
