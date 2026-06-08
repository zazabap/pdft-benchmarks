#!/usr/bin/env python3
"""Sweep BlockMaskedIdentityRegQFTMSELoss lambda on DIV2K-8q from
qft_identity init.

Trains QFT(8,8) with identity init under the headline preset for
several values of lambda at fixed outer_weight W=10 and inner block
(3,3). Saves per-run metrics, loss history, and trained tensors under
results/qft_identity_init/div2k_8q/_runs/reg_lambda_<lam>_W<W>/.

Standalone driver: does NOT modify the canonical pipeline. Uses
pdft_benchmarks.evaluation.evaluate_basis_shared for the eval step so
metrics-table semantics match the headline cells exactly.

Usage:
    python experiments/qft_identity_regularization.py --gpu 0 \\
        --lambdas 0,1e-3,1e-2,1e-1,1,10 --outer-weight 10
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. Sets CUDA_VISIBLE_DEVICES before JAX import.")
    parser.add_argument("--lambdas", type=str,
                        default="0,1e-3,1e-2,1e-1,1,10",
                        help="Comma-separated lambda values for the sweep.")
    parser.add_argument("--dataset", type=str, default="div2k_8q",
                        choices=["div2k_8q", "quickdraw", "tuberlin"],
                        help="Dataset + qubit-config. div2k_8q -> m=n=8, "
                             "256x256 DIV2K; quickdraw -> m=n=5, 32x32 "
                             "QuickDraw; tuberlin -> m=n=8, 256x256 "
                             "TU-Berlin sketches (sparse line drawings "
                             "at DIV2K scale).")
    parser.add_argument("--reg", type=str, default="block", choices=["block", "L1"],
                        help="Regulariser family. 'block': block-masked L2 "
                             "(BlockMaskedIdentityRegQFTMSELoss, default); "
                             "'L1': sum of unsquared Frobenius distances "
                             "(L1IdentityRegQFTMSELoss). The L1 variant ignores "
                             "--outer-weight, --inner-m, --inner-n.")
    parser.add_argument("--outer-weight", type=float, default=10.0,
                        help="W: weight on outer-gate contributions to R_block. "
                             "Default 10. Ignored when --reg=L1.")
    parser.add_argument("--inner-m", type=int, default=3,
                        help="Axis-1 inner-block size. Default 3 (matches blocked_8).")
    parser.add_argument("--inner-n", type=int, default=3,
                        help="Axis-2 inner-block size. Default 3 (matches blocked_8).")
    parser.add_argument("--out-base", type=str,
                        default=None,
                        help="Parent directory for per-lambda run folders. "
                             "Default depends on --dataset.")
    parser.add_argument("--preset", type=str, default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--epochs", type=int, default=112,
                        help="Override preset.epochs.")
    parser.add_argument("--no-early-stop", action="store_true", default=True,
                        help="Disable early-stopping (default true: matches headline).")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Imports below trigger JAX device discovery; must come AFTER env var.
    import jax
    import pdft
    import pdft.io  # noqa: F401 — register pdft.io submodule for evaluate_basis_shared
    from pdft_benchmarks.bases import qft_identity_basis
    from pdft_benchmarks.identity_reg import (
        BlockMaskedIdentityRegQFTMSELoss,
        L1IdentityRegQFTMSELoss,
    )
    from pdft_benchmarks.datasets.div2k import load_div2k
    from pdft_benchmarks.datasets.quickdraw import load_quickdraw
    from pdft_benchmarks.datasets.tuberlin import load_tuberlin
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.experiment_utils import serialize_tensors
    from pdft_benchmarks.presets import get_preset

    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    W = args.outer_weight
    print(f"[qft_id_reg] reg family: {args.reg}")
    print(f"[qft_id_reg] sweep lambdas: {lambdas}")
    if args.reg == "block":
        print(f"[qft_id_reg] outer_weight W={W}, inner=({args.inner_m},{args.inner_n})")
    print(f"[qft_id_reg] preset={args.preset}, epochs={args.epochs}")

    preset = get_preset(args.dataset, args.preset)
    if args.no_early_stop:
        preset = replace(preset, epochs=args.epochs, early_stopping_patience=10**9)
    else:
        preset = replace(preset, epochs=args.epochs)
    print(f"[qft_id_reg] dataset={args.dataset}, preset.epochs={preset.epochs}, "
          f"patience={preset.early_stopping_patience}")

    if args.dataset == "div2k_8q":
        m = n = 8
        train_imgs_np, test_imgs_np = load_div2k(
            n_train=preset.n_train, n_test=preset.n_test, seed=preset.seed, size=2**m,
        )
    elif args.dataset == "quickdraw":
        m = n = 5
        train_imgs_np, test_imgs_np = load_quickdraw(
            n_train=preset.n_train, n_test=preset.n_test, seed=preset.seed, img_size=2**m,
        )
    else:  # tuberlin
        m = n = 8
        train_imgs_np, test_imgs_np = load_tuberlin(
            n_train=preset.n_train, n_test=preset.n_test, seed=preset.seed, size=2**m,
        )
    k = max(1, round(2 ** (m + n) * 0.1))
    print(f"[qft_id_reg] m=n={m}, loaded {len(train_imgs_np)} train, "
          f"{len(test_imgs_np)} test images")

    if args.out_base is None:
        out_base = Path(f"results/qft_identity_init/{args.dataset}/_runs")
    else:
        out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    for lam in lambdas:
        lam_str = f"{lam:.0e}" if lam > 0 else "0"
        if args.reg == "block":
            lam_tag = f"reg_lambda_{lam_str}_W{int(W) if W == int(W) else W}"
        else:  # L1
            lam_tag = f"regL1_lambda_{lam_str}"
        out_dir = out_base / lam_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "loss_history").mkdir(parents=True, exist_ok=True)
        print(f"\n[qft_id_reg] === reg={args.reg}, lambda={lam:.2e} (out: {out_dir}) ===")

        basis = qft_identity_basis(m, n)
        if lam == 0:
            loss = pdft.MSELoss(k=k)
        elif args.reg == "block":
            loss = BlockMaskedIdentityRegQFTMSELoss(
                k=k, lam=lam, m=m, n=n,
                inner_m=args.inner_m, inner_n=args.inner_n,
                outer_weight=W)
        else:  # L1
            loss = L1IdentityRegQFTMSELoss(k=k, lam=lam, m=m, n=n)

        t0 = time.perf_counter()
        result = pdft.train_basis_batched(
            basis, dataset=train_imgs_np, loss=loss,
            epochs=preset.epochs, batch_size=preset.batch_size,
            optimizer=preset.optimizer,
            validation_split=preset.validation_split,
            early_stopping_patience=preset.early_stopping_patience,
            warmup_frac=preset.warmup_frac,
            lr_peak=preset.lr_peak,
            lr_final=preset.lr_final,
            max_grad_norm=preset.max_grad_norm,
            shuffle=True, seed=preset.seed,
            val_every_k_epochs=preset.val_every_k_epochs,
        )
        elapsed = time.perf_counter() - t0
        print(f"[qft_id_reg]   trained in {elapsed:.1f}s, steps={result.steps}, "
              f"epochs={result.epochs_completed}")

        eval_metrics, _ = evaluate_basis_shared(
            result.basis, test_imgs_np, keep_ratios=(0.05, 0.10, 0.15, 0.20),
        )

        (out_dir / "metrics.json").write_text(json.dumps(
            {"qft_identity_reg": {
                "metrics": eval_metrics,
                "time": elapsed,
                "_pdft_py": {
                    "reg": args.reg,
                    "lam": lam,
                    "outer_weight": W if args.reg == "block" else None,
                    "inner_m": args.inner_m if args.reg == "block" else None,
                    "inner_n": args.inner_n if args.reg == "block" else None,
                    "steps": int(result.steps),
                    "epochs_completed": int(result.epochs_completed),
                    "device": str(jax.devices()[0]),
                    "n_test": int(len(test_imgs_np)),
                }
            }},
            indent=2,
        ))
        (out_dir / "loss_history" / "qft_identity_reg_loss.json").write_text(json.dumps({
            "step_losses": [float(x) for x in result.loss_history],
            "val_losses": [float(x) for x in result.val_history],
            "epochs_completed": int(result.epochs_completed),
            "steps": int(result.steps),
        }, indent=2))
        (out_dir / "trained_qft_identity_reg.json").write_text(json.dumps({
            "lam": lam, "outer_weight": W,
            "inner_m": args.inner_m, "inner_n": args.inner_n,
            "m": m, "n": n,
            "tensors": serialize_tensors(result.basis.tensors),
        }, indent=2))
        psnr20 = eval_metrics["0.2"]["mean_psnr"]
        print(f"[qft_id_reg]   PSNR @ rho=0.20: {psnr20:.2f} dB")

    print("\n[qft_id_reg] sweep complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
