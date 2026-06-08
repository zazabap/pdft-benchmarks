#!/usr/bin/env python3
"""L1-to-init-anchor sweep across parametric bases on DIV2K-8q.

For each (basis, lambda) in the requested grid, trains the basis from
its canonical analytic init under L1InitAnchorMSELoss with target tensors
frozen at the init values, then evaluates on the DIV2K-8q test split at
rho in {0.05, 0.10, 0.15, 0.20}.

Tests whether the qft_identity L1 finding (gates collapse to 6 active
SO(2) rotations under L1 at lam=10) transfers to other topologies
(entangled_qft, tebd, mera, blocked_8, rich_8, real_rich_8). Mirrors the
structure of qft_identity_regularization.py — same dataset, same preset,
same eval grid.

Usage (split across two 3090s for parallelism):

    # GPU 0 — unblocked bases
    python experiments/div2k_8q_l1_init_anchor.py --gpu 0 \\
        --bases qft,entangled_qft,tebd,mera --lambdas 0,1,10

    # GPU 1 — blocked bases
    python experiments/div2k_8q_l1_init_anchor.py --gpu 1 \\
        --bases blocked_8,rich_8,real_rich_8 --lambdas 0,1,10

Output: results/qft_identity_init/div2k_8q_l1_init_anchor/_runs/<basis>_lambda_<lam>/
    metrics.json
    loss_history/<basis>_loss.json
    trained_<basis>.json
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
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. Sets CUDA_VISIBLE_DEVICES before JAX import.")
    parser.add_argument("--bases", type=str, required=True,
                        help="Comma-separated basis names. Choose from: qft, "
                             "entangled_qft, tebd, mera, blocked_8, rich_8, "
                             "real_rich_8.")
    parser.add_argument("--lambdas", type=str, default="0,1,10",
                        help="Comma-separated lambda values for the L1 anchor. "
                             "Default 0,1,10 (matches QFT identity-init headline lam=10).")
    parser.add_argument("--out-base", type=str,
                        default="results/qft_identity_init/div2k_8q_l1_init_anchor/_runs",
                        help="Parent directory for per-(basis,lambda) run folders.")
    parser.add_argument("--init-mode", type=str, default="canonical",
                        choices=["canonical", "identity"],
                        help="Initial tensor values + L1 anchor target. "
                             "'canonical' (default): each basis starts at its "
                             "analytic init (qft -> approximate FFT, etc.); "
                             "L1 anchors there. 'identity': each basis starts at "
                             "T = identity (H -> I_2, CP -> [[1,1],[1,1]], "
                             "U(4) -> 4x4 I); L1 anchors at identity for all.")
    parser.add_argument("--preset", type=str, default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--epochs", type=int, default=112,
                        help="Override preset.epochs. Default 112 (= 1008 steps).")
    parser.add_argument("--no-early-stop", action="store_true", default=True,
                        help="Disable early stopping (default true: matches headline).")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Imports below trigger JAX device discovery; must come AFTER env var.
    import jax
    import pdft
    import pdft.io  # noqa: F401
    from pdft_benchmarks.bases import BASIS_FACTORIES, identity_basis_for
    from pdft_benchmarks.identity_reg import L1InitAnchorMSELoss
    from pdft_benchmarks.datasets.div2k import load_div2k
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.experiment_utils import serialize_tensors
    from pdft_benchmarks.presets import get_preset

    bases = [b.strip() for b in args.bases.split(",") if b.strip()]
    unknown = [b for b in bases if b not in BASIS_FACTORIES]
    if unknown:
        print(f"unknown basis name(s): {unknown}; choices: "
              f"{sorted(BASIS_FACTORIES)}", file=sys.stderr)
        return 2

    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    print(f"[l1_anchor] bases={bases}, lambdas={lambdas}")

    m = n = 8
    preset = get_preset("div2k_8q", args.preset)
    if args.no_early_stop:
        preset = replace(preset, epochs=args.epochs,
                          early_stopping_patience=10**9)
    else:
        preset = replace(preset, epochs=args.epochs)
    print(f"[l1_anchor] preset.epochs={preset.epochs}, "
          f"patience={preset.early_stopping_patience}")

    train_imgs_np, test_imgs_np = load_div2k(
        n_train=preset.n_train, n_test=preset.n_test,
        seed=preset.seed, size=2**m,
    )
    k = max(1, round(2 ** (m + n) * 0.1))
    print(f"[l1_anchor] m=n={m}, loaded {len(train_imgs_np)} train, "
          f"{len(test_imgs_np)} test, k={k}")

    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    for basis_name in bases:
        for lam in lambdas:
            lam_str = f"{lam:.0e}" if lam > 0 else "0"
            tag = f"{basis_name}_lambda_{lam_str}"
            out_dir = out_base / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "loss_history").mkdir(parents=True, exist_ok=True)
            print(f"\n[l1_anchor] === basis={basis_name}, lambda={lam:.2e} "
                  f"(out: {out_dir}) ===")

            if args.init_mode == "identity":
                basis = identity_basis_for(basis_name, m=m, n=n)
            else:
                basis = BASIS_FACTORIES[basis_name](m=m, n=n, seed=preset.seed)
            target_tensors = tuple(basis.tensors)

            if lam == 0:
                loss = pdft.MSELoss(k=k)
            else:
                loss = L1InitAnchorMSELoss(
                    k=k, lam=lam, target_tensors=target_tensors,
                )

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
            print(f"[l1_anchor]   trained in {elapsed:.1f}s, "
                  f"steps={result.steps}")

            eval_metrics, _ = evaluate_basis_shared(
                result.basis, test_imgs_np,
                keep_ratios=(0.05, 0.10, 0.15, 0.20),
            )

            (out_dir / "metrics.json").write_text(json.dumps(
                {basis_name: {
                    "metrics": eval_metrics,
                    "time": elapsed,
                    "_pdft_py": {
                        "basis": basis_name,
                        "lam": lam,
                        "init_mode": args.init_mode,
                        "regulariser":
                            "L1InitAnchorMSELoss" if lam > 0 else "MSELoss",
                        "steps": int(result.steps),
                        "epochs_completed": int(result.epochs_completed),
                        "device": str(jax.devices()[0]),
                        "n_test": int(len(test_imgs_np)),
                    },
                }},
                indent=2,
            ))
            (out_dir / "loss_history" / f"{basis_name}_loss.json").write_text(
                json.dumps({
                    "step_losses": [float(x) for x in result.loss_history],
                    "val_losses": [float(x) for x in result.val_history],
                    "epochs_completed": int(result.epochs_completed),
                    "steps": int(result.steps),
                }, indent=2)
            )
            (out_dir / f"trained_{basis_name}.json").write_text(json.dumps({
                "basis": basis_name,
                "lam": lam,
                "m": m, "n": n,
                "tensors": serialize_tensors(result.basis.tensors),
            }, indent=2))
            psnr20 = eval_metrics["0.2"]["mean_psnr"]
            print(f"[l1_anchor]   PSNR @ rho=0.20: {psnr20:.2f} dB")

    print("\n[l1_anchor] sweep complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
