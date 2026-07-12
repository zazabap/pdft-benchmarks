#!/usr/bin/env python3
"""Drive the qft_freeze_sweep experiment on DIV2K-8q.

Two cells using the upstream pdft.train_basis_batched(..., frozen_indices=...)
parameter (added in pdft commit 31decf7):

  freeze_outer:  QFTBasis(8, 8) identity init, freeze 60 outer gates,
                 train 12 inner gates. Should match blocked_8 PSNR
                 (32.26 dB @ rho=0.20) by operator equivalence.

  freeze_inner:  QFTBasis(8, 8) identity init, freeze 12 inner gates,
                 train 60 outer gates. Vice-versa configuration probing
                 the standalone compression signal of inter-block /
                 cross-block structure.

Inner-block boundary fixed at (inner_m=3, inner_n=3) to match the
blocked_8 anchor's structural form.

Cells land at results/training/_archive/qft_freeze_sweep/div2k_8q/_runs/{freeze_outer,freeze_inner}/.
Manifest at results/training/_archive/qft_freeze_sweep/div2k_8q/manifest.json.

Usage:
    python experiments/qft/qft_freeze_sweep.py --gpu 0 [--epochs 112]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=112,
                        help="Epoch budget per cell (default 112 = 1008 steps).")
    parser.add_argument("--out-base", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="div2k_8q",
                        choices=["div2k_8q"])
    parser.add_argument("--preset", type=str, default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--inner-m", type=int, default=3)
    parser.add_argument("--inner-n", type=int, default=3)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["both", "freeze_outer", "freeze_inner"],
                        help="Which cell(s) to train. Default 'both'.")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import jax
    import pdft
    import pdft.io  # noqa: F401
    from pdft_benchmarks.bases import (
        qft_identity_basis,
        qft_inner_outer_indices,
    )
    from pdft_benchmarks.datasets import load_div2k
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.experiment_utils import git_sha, serialize_tensors
    from pdft_benchmarks.presets import get_preset

    # GPU fail-fast — same pattern as qft_progressive driver.
    devices = jax.devices()
    chosen = devices[0]
    print(f"[qft_freeze_sweep] JAX devices: {devices}")
    print(f"[qft_freeze_sweep] chosen device: {chosen} (platform={chosen.platform!r})")
    if args.gpu is not None and chosen.platform not in ("gpu", "cuda"):
        print(
            f"[qft_freeze_sweep] FATAL: --gpu {args.gpu} requested but JAX "
            f"sees platform={chosen.platform!r}. Aborting.",
            file=sys.stderr,
        )
        return 2

    preset = get_preset(args.dataset, args.preset)
    preset = replace(preset, epochs=args.epochs, early_stopping_patience=10**9)
    print(f"[qft_freeze_sweep] dataset={args.dataset}, epochs={preset.epochs}, "
          f"seed={preset.seed}, inner=({args.inner_m},{args.inner_n})")

    m = n = 8
    train_imgs_np, test_imgs_np = load_div2k(
        n_train=preset.n_train, n_test=preset.n_test,
        seed=preset.seed, size=2**m,
    )
    k_train = max(1, round(2 ** (m + n) * 0.1))
    print(f"[qft_freeze_sweep] m=n={m}, k_train={k_train}, "
          f"{len(train_imgs_np)} train, {len(test_imgs_np)} test images")

    inner_idx, outer_idx = qft_inner_outer_indices(
        m=m, n=n, inner_m=args.inner_m, inner_n=args.inner_n,
    )
    assert len(inner_idx) == 12 and len(outer_idx) == 60, \
        f"unexpected partition: {len(inner_idx)} inner + {len(outer_idx)} outer"
    print(f"[qft_freeze_sweep] inner indices ({len(inner_idx)}): {inner_idx}")
    print(f"[qft_freeze_sweep] outer indices ({len(outer_idx)}): "
          f"first 10 = {outer_idx[:10]} ... last 3 = {outer_idx[-3:]}")

    out_base = Path(args.out_base) if args.out_base else \
        Path(f"results/training/_archive/qft_freeze_sweep/{args.dataset}/_runs")
    out_base.mkdir(parents=True, exist_ok=True)

    cell_specs = []
    if args.mode in ("both", "freeze_outer"):
        cell_specs.append({
            "name": "freeze_outer",
            "basis_name": "qft_freeze_outer",
            "frozen_indices": outer_idx,
            "trainable_count": len(inner_idx),
            "description": (
                "QFT(8,8) identity init, freeze 60 outer gates, train 12 "
                "inner. Operator-equivalent to BlockedBasis(QFT(3,3), 5, 5)."
            ),
        })
    if args.mode in ("both", "freeze_inner"):
        cell_specs.append({
            "name": "freeze_inner",
            "basis_name": "qft_freeze_inner",
            "frozen_indices": inner_idx,
            "trainable_count": len(outer_idx),
            "description": (
                "QFT(8,8) identity init, freeze 12 inner gates, train 60 "
                "outer. Vice-versa configuration."
            ),
        })

    summaries: list[dict] = []
    for spec in cell_specs:
        out_dir = out_base / spec["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "loss_history").mkdir(parents=True, exist_ok=True)

        print(f"\n[qft_freeze_sweep] === cell {spec['name']!r} "
              f"({spec['trainable_count']} trainable / "
              f"{len(spec['frozen_indices'])} frozen) -> {out_dir} ===")
        print(f"[qft_freeze_sweep]   {spec['description']}")

        basis = qft_identity_basis(m=m, n=n)

        t0 = time.perf_counter()
        result = pdft.train_basis_batched(
            basis,
            dataset=train_imgs_np,
            loss=pdft.MSELoss(k=k_train),
            epochs=preset.epochs,
            batch_size=preset.batch_size,
            optimizer=preset.optimizer,
            validation_split=preset.validation_split,
            early_stopping_patience=preset.early_stopping_patience,
            warmup_frac=preset.warmup_frac,
            lr_peak=preset.lr_peak,
            lr_final=preset.lr_final,
            max_grad_norm=preset.max_grad_norm,
            shuffle=True, seed=preset.seed,
            val_every_k_epochs=preset.val_every_k_epochs,
            frozen_indices=spec["frozen_indices"],
        )
        elapsed = time.perf_counter() - t0
        steps = int(result.steps)
        epochs_done = int(result.epochs_completed)
        print(f"[qft_freeze_sweep]   trained in {elapsed:.1f}s, "
              f"steps={steps}, epochs={epochs_done}")

        eval_metrics, _ = evaluate_basis_shared(
            result.basis, test_imgs_np,
            keep_ratios=(0.05, 0.10, 0.15, 0.20),
        )
        psnr20 = float(eval_metrics["0.2"]["mean_psnr"])
        Lf = float(result.val_history[-1]) if len(result.val_history) > 0 else float("nan")
        print(f"[qft_freeze_sweep]   PSNR @ rho=0.20: {psnr20:.3f} dB, "
              f"val MSE final: {Lf:.3f}")

        # Sanity check: frozen tensors haven't moved from identity.
        from pdft_benchmarks.bases import qft_identity_basis as _ref
        ref_tensors = _ref(m=m, n=n).tensors
        import jax.numpy as jnp
        for i in spec["frozen_indices"]:
            diff = float(jnp.max(jnp.abs(result.basis.tensors[i] - ref_tensors[i])))
            assert diff == 0.0, (
                f"frozen index {i} drifted by {diff} — "
                f"frozen_indices semantics broken!"
            )
        print(f"[qft_freeze_sweep]   verified: all {len(spec['frozen_indices'])} "
              f"frozen tensors are bit-exactly at identity.")

        basis_name = spec["basis_name"]
        (out_dir / "metrics.json").write_text(json.dumps({
            basis_name: {
                "metrics": eval_metrics,
                "time": elapsed,
                "_pdft_py": {
                    "mode": spec["name"],
                    "trainable_count": int(spec["trainable_count"]),
                    "frozen_count": int(len(spec["frozen_indices"])),
                    "steps": steps,
                    "epochs_completed": epochs_done,
                    "device": str(jax.devices()[0]),
                    "n_test": int(len(test_imgs_np)),
                }
            }
        }, indent=2))
        (out_dir / "loss_history" / f"{basis_name}_loss.json").write_text(json.dumps({
            "step_losses": [float(x) for x in result.loss_history],
            "val_losses": [float(x) for x in result.val_history],
            "epochs_completed": epochs_done,
            "steps": steps,
        }, indent=2))
        (out_dir / f"trained_{basis_name}.json").write_text(json.dumps({
            "mode": spec["name"],
            "m": m, "n": n,
            "inner_m": int(args.inner_m), "inner_n": int(args.inner_n),
            "frozen_indices": list(spec["frozen_indices"]),
            "tensors": serialize_tensors(result.basis.tensors),
        }, indent=2))
        (out_dir / "env.json").write_text(json.dumps({
            "experiment": "qft_freeze_sweep",
            "cell": spec["name"],
            "epochs_used": epochs_done,
            "steps_used": steps,
            "trainable_count": int(spec["trainable_count"]),
            "frozen_count": int(len(spec["frozen_indices"])),
            "inner_m": int(args.inner_m), "inner_n": int(args.inner_n),
            "init_policy": "identity",
            "preset_name": args.preset,
            "preset_epochs": int(args.epochs),
            "device": str(jax.devices()[0]),
            "git_sha": git_sha(short=False),
        }, indent=2))

        summaries.append({
            "cell": spec["name"], "basis_name": basis_name,
            "trainable_count": int(spec["trainable_count"]),
            "frozen_count": int(len(spec["frozen_indices"])),
            "psnr_rho_020": psnr20,
            "val_mse_final": Lf,
            "steps": steps,
            "elapsed_seconds": float(elapsed),
        })

    manifest_path = out_base.parent / "manifest.json"
    manifest_path.write_text(json.dumps({
        "experiment": "qft_freeze_sweep",
        "dataset": args.dataset,
        "epochs": int(args.epochs),
        "inner_m": int(args.inner_m), "inner_n": int(args.inner_n),
        "cells": summaries,
        "anchors": {"qft": 31.29, "qft_identity": 31.66, "blocked_8": 32.26},
        "git_sha": git_sha(short=False),
    }, indent=2))

    print(f"\n[qft_freeze_sweep] complete. Manifest: {manifest_path}")
    print("[qft_freeze_sweep] PSNR @ rho=0.20:")
    for s in summaries:
        print(f"  {s['cell']:<14s} ({s['trainable_count']:>2d} trainable, "
              f"{s['frozen_count']:>2d} frozen): {s['psnr_rho_020']:.3f} dB "
              f"(val MSE {s['val_mse_final']:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
