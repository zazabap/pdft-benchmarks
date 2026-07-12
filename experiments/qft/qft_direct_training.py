#!/usr/bin/env python3
"""QFT direct-training experiments (results/training/2_direct_training/).

Part 2 of the QFT training study: can good QFT operators be reached by training
*directly* (rather than by the warm-start of part 1)? Three sub-experiments,
dispatched by subcommand:

  l1-reg     Identity-init QFT under a block-masked / L1 identity-regularised
             loss, swept over lambda. Cells ->
             results/training/2_direct_training/identity_l1/<dataset>/_runs/.
  l1-anchor  L1-to-init-anchor across parametric bases — does the QFT L1
             finding transfer to other topologies? Cells ->
             results/training/2_direct_training/identity_l1/div2k_8q_l1_init_anchor/_runs/.
  unfreeze   Progressive gate-unfreezing of identity/random-init QFT, one gate
             thawed per stage. Cells ->
             results/training/2_direct_training/unfreeze/<dataset>/.

Standalone driver: does NOT use run_experiment; each mode writes its own cells
via evaluate_basis_shared so metrics-table semantics match the headline cells.

GPU isolation: --gpu sets CUDA_VISIBLE_DEVICES (+ CUDA_DEVICE_ORDER=PCI_BUS_ID)
BEFORE importing pdft_benchmarks (which transitively imports JAX). On this
mixed-GPU host the PCI-order pin makes --gpu N select nvidia-smi's GPU N.

Usage:
    python experiments/qft/qft_direct_training.py l1-reg    --gpu 0 --lambdas 0,1,10
    python experiments/qft/qft_direct_training.py l1-anchor --gpu 0 --bases qft,tebd
    python experiments/qft/qft_direct_training.py unfreeze  --gpu 0 --dataset div2k_8q
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path


# ===========================================================================
# l1-reg  — identity-reg lambda sweep  (was qft_identity_regularization.py)
# ===========================================================================
def _add_l1_reg(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--gpu", type=int, default=None,
                    help="GPU index. Sets CUDA_VISIBLE_DEVICES before JAX import.")
    sp.add_argument("--lambdas", type=str,
                    default="0,1e-3,1e-2,1e-1,1,10",
                    help="Comma-separated lambda values for the sweep.")
    sp.add_argument("--dataset", type=str, default="div2k_8q",
                    choices=["div2k_8q", "quickdraw", "tuberlin"],
                    help="Dataset + qubit-config. div2k_8q -> m=n=8, "
                         "256x256 DIV2K; quickdraw -> m=n=5, 32x32 "
                         "QuickDraw; tuberlin -> m=n=8, 256x256 "
                         "TU-Berlin sketches (sparse line drawings "
                         "at DIV2K scale).")
    sp.add_argument("--reg", type=str, default="block", choices=["block", "L1"],
                    help="Regulariser family. 'block': block-masked L2 "
                         "(BlockMaskedIdentityRegQFTMSELoss, default); "
                         "'L1': sum of unsquared Frobenius distances "
                         "(L1IdentityRegQFTMSELoss). The L1 variant ignores "
                         "--outer-weight, --inner-m, --inner-n.")
    sp.add_argument("--outer-weight", type=float, default=10.0,
                    help="W: weight on outer-gate contributions to R_block. "
                         "Default 10. Ignored when --reg=L1.")
    sp.add_argument("--inner-m", type=int, default=3,
                    help="Axis-1 inner-block size. Default 3 (matches blocked_8).")
    sp.add_argument("--inner-n", type=int, default=3,
                    help="Axis-2 inner-block size. Default 3 (matches blocked_8).")
    sp.add_argument("--out-base", type=str, default=None,
                    help="Parent directory for per-lambda run folders. "
                         "Default depends on --dataset.")
    sp.add_argument("--preset", type=str, default="generalized",
                    choices=["smoke", "moderate", "generalized"])
    sp.add_argument("--epochs", type=int, default=112,
                    help="Override preset.epochs.")
    sp.add_argument("--no-early-stop", action="store_true", default=True,
                    help="Disable early-stopping (default true: matches headline).")


def _run_l1_reg(args) -> int:
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
        out_base = Path(f"results/training/2_direct_training/identity_l1/{args.dataset}/_runs")
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


# ===========================================================================
# l1-anchor — L1-to-init-anchor across bases  (was div2k_8q_l1_init_anchor.py)
# ===========================================================================
def _add_l1_anchor(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--gpu", type=int, default=None,
                    help="GPU index. Sets CUDA_VISIBLE_DEVICES before JAX import.")
    sp.add_argument("--bases", type=str, required=True,
                    help="Comma-separated basis names. Choose from: qft, "
                         "entangled_qft, tebd, mera, blocked_8, rich_8, "
                         "real_rich_8.")
    sp.add_argument("--lambdas", type=str, default="0,1,10",
                    help="Comma-separated lambda values for the L1 anchor. "
                         "Default 0,1,10 (matches QFT identity-init headline lam=10).")
    sp.add_argument("--out-base", type=str,
                    default="results/training/2_direct_training/identity_l1/div2k_8q_l1_init_anchor/_runs",
                    help="Parent directory for per-(basis,lambda) run folders.")
    sp.add_argument("--init-mode", type=str, default="canonical",
                    choices=["canonical", "identity"],
                    help="Initial tensor values + L1 anchor target. "
                         "'canonical' (default): each basis starts at its "
                         "analytic init (qft -> approximate FFT, etc.); "
                         "L1 anchors there. 'identity': each basis starts at "
                         "T = identity (H -> I_2, CP -> [[1,1],[1,1]], "
                         "U(4) -> 4x4 I); L1 anchors at identity for all.")
    sp.add_argument("--preset", type=str, default="generalized",
                    choices=["smoke", "moderate", "generalized"])
    sp.add_argument("--epochs", type=int, default=112,
                    help="Override preset.epochs. Default 112 (= 1008 steps).")
    sp.add_argument("--no-early-stop", action="store_true", default=True,
                    help="Disable early stopping (default true: matches headline).")


def _run_l1_anchor(args) -> int:
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


# ===========================================================================
# unfreeze — progressive gate-unfreezing  (was qft_unfreeze.py)
# ===========================================================================
def _add_unfreeze(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--gpu", type=int, default=None)
    sp.add_argument("--dataset", default="quickdraw_5q",
                    choices=["quickdraw_5q", "div2k_8q", "tuberlin_8q"])
    sp.add_argument("--orderings", default="bg,lr,rl")
    sp.add_argument("--batch", type=int, default=None,
                    help="Fixed batch size. Default: full train set at m=5, else 50.")
    sp.add_argument("--lr", type=float, default=None, help="Default: preset.lr_peak.")
    sp.add_argument("--grad-tol", type=float, default=1e-5)
    sp.add_argument("--loss-tol", type=float, default=1e-5)
    sp.add_argument("--min-steps", type=int, default=5)
    sp.add_argument("--max-steps", type=int, default=2000)
    sp.add_argument("--psnr-every", type=int, default=1,
                    help="Evaluate per-stage test PSNR only every K stages (the "
                         "final stage is always evaluated). K>1 cuts the dominant "
                         "m=8 eval cost; the loss/grad-norm staircase is unaffected.")
    sp.add_argument("--grad-check-every", type=int, default=1,
                    help="Run the Riemannian grad-norm probe only every K steps "
                         "(loss-delta plateau uses the Adam step's free loss every "
                         "step). K>1 roughly halves the per-step cost at m=8.")
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--init", default="identity", choices=["identity", "random"],
                    help="Gate initialisation: QFT-family identity (default) or "
                         "Haar-random (H slots -> Haar U(2), CP slots -> uniform phase).")
    sp.add_argument("--init-seed", type=int, default=None,
                    help="Seed for --init random (default: preset.seed). Shared "
                         "across orderings so they start from the same basis.")
    sp.add_argument("--preset", default="generalized",
                    choices=["smoke", "moderate", "generalized"])
    sp.add_argument("--out", default=None)


def _run_unfreeze(args) -> int:
    # imports AFTER setting CUDA_VISIBLE_DEVICES (mirrors qft_progressive.py)
    import jax
    import numpy as np
    import pdft
    import pdft.io  # noqa: F401  (needed by evaluate_basis_shared)
    from pdft_benchmarks.bases import family_random_basis, qft_identity_basis
    from pdft_benchmarks.datasets import load_div2k, load_quickdraw, load_tuberlin
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.experiment_utils import git_sha, serialize_tensors
    from pdft_benchmarks.presets import get_preset
    from pdft_benchmarks.unfreeze import qft_unfreeze_orders, train_progressive_unfreeze

    chosen = jax.devices()[0]
    print(f"[qft_unfreeze] device: {chosen} (platform={chosen.platform!r})")
    if args.gpu is not None and chosen.platform not in ("gpu", "cuda"):
        print(f"[qft_unfreeze] FATAL: --gpu {args.gpu} requested but JAX sees "
              f"platform={chosen.platform!r} (NVML init failure?). Aborting.",
              file=sys.stderr)
        return 2

    DATASET_CFG = {
        "quickdraw_5q": (5, load_quickdraw, "img_size"),
        "div2k_8q": (8, load_div2k, "size"),
        "tuberlin_8q": (8, load_tuberlin, "size"),
    }
    m_q, loader, size_kw = DATASET_CFG[args.dataset]
    m = n = m_q
    preset = get_preset(args.dataset, args.preset)
    seed = args.seed if args.seed is not None else preset.seed
    init_seed = args.init_seed if args.init_seed is not None else preset.seed
    lr = args.lr if args.lr is not None else preset.lr_peak
    batch = args.batch if args.batch is not None else (preset.n_train if m == 5 else 50)

    train_imgs, test_imgs = loader(n_train=preset.n_train, n_test=preset.n_test,
                                   seed=seed, **{size_kw: 2 ** m})
    fixed_batch = [np.asarray(x) for x in train_imgs[:batch]]
    k_train = max(1, round(2 ** (m + n) * 0.1))
    loss = pdft.MSELoss(k=k_train)
    print(f"[qft_unfreeze] dataset={args.dataset} m=n={m} init={args.init} "
          f"init_seed={init_seed} batch={len(fixed_batch)} "
          f"k_train={k_train} lr={lr} max_steps={args.max_steps}")

    def make_basis():
        if args.init == "random":
            return family_random_basis("qft", m, n, init_seed)
        return qft_identity_basis(m=m, n=n)

    orders = qft_unfreeze_orders(m, n)
    keep_ratios = (0.05, 0.10, 0.15, 0.20)

    out_base = Path(args.out) if args.out else \
        Path(f"results/training/2_direct_training/unfreeze/{args.dataset}")
    out_base.mkdir(parents=True, exist_ok=True)

    manifest_orderings = {}
    for name in [s.strip() for s in args.orderings.split(",") if s.strip()]:
        order = orders[name]
        print(f"\n[qft_unfreeze] === ordering {name!r}: {len(order)} stages ===")
        basis = make_basis()
        n_stages = len(order)

        def stage_psnr(stage, tensors):
            # Skip the expensive full-test-set eval on intermediate stages when
            # --psnr-every > 1; always evaluate the final stage.
            if args.psnr_every > 1 and stage != n_stages and stage % args.psnr_every != 0:
                return {}
            b = pdft.QFTBasis(m=m, n=n, tensors=tensors)
            metrics, _ = evaluate_basis_shared(b, test_imgs, keep_ratios=keep_ratios)
            return {"psnr": {f"{r}": float(metrics[str(r)]["mean_psnr"]) for r in keep_ratios}}

        t0 = time.perf_counter()
        res = train_progressive_unfreeze(
            basis, fixed_batch, unfreeze_order=order, lr=lr,
            max_steps_per_stage=args.max_steps, loss=loss,
            grad_tol=args.grad_tol, loss_tol=args.loss_tol,
            min_steps_per_stage=args.min_steps, seed=seed,
            stage_callback=stage_psnr, grad_check_every=args.grad_check_every)
        elapsed = time.perf_counter() - t0
        total_steps = res.stages[-1].end_step
        print(f"[qft_unfreeze]   {name}: {total_steps} steps, {elapsed:.1f}s, "
              f"final PSNR@0.2={res.stages[-1].extra['psnr']['0.2']:.3f} dB")

        cell = out_base / name
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "trace.json").write_text(json.dumps({
            "dataset": args.dataset, "ordering": name, "m": m, "n": n,
            "lr": lr, "grad_tol": args.grad_tol, "loss_tol": args.loss_tol,
            "min_steps": args.min_steps, "max_steps": args.max_steps,
            "batch": len(fixed_batch), "k_train": k_train,
            "steps": res.trace,
            "stages": [vars(s) for s in res.stages],
            "git_sha": git_sha(short=False),
        }, indent=2))
        (cell / "trained_final.json").write_text(json.dumps({
            "ordering": name, "m": int(res.basis.m), "n": int(res.basis.n),
            "tensors": serialize_tensors(res.basis.tensors),
        }, indent=2))
        (cell / "env.json").write_text(json.dumps({
            "experiment": "qft_unfreeze", "dataset": args.dataset, "ordering": name,
            "init": args.init, "init_seed": init_seed, "lr": lr,
            "grad_tol": args.grad_tol, "loss_tol": args.loss_tol,
            "min_steps": args.min_steps, "max_steps": args.max_steps,
            "batch": len(fixed_batch), "seed": seed,
            "device": str(chosen), "git_sha": git_sha(short=False),
        }, indent=2))

        manifest_orderings[name] = {
            "n_stages": len(res.stages), "total_steps": total_steps,
            "elapsed_seconds": elapsed,
            "trigger_counts": {tr: sum(1 for s in res.stages if s.trigger == tr)
                               for tr in ("grad_norm", "loss_delta", "max_steps")},
            "final_loss": res.stages[-1].final_loss,
            "final_psnr": res.stages[-1].extra["psnr"],
        }

    (out_base / "manifest.json").write_text(json.dumps({
        "experiment": "qft_unfreeze", "dataset": args.dataset, "m": m, "n": n,
        "init": args.init, "init_seed": init_seed,
        "orderings": manifest_orderings, "git_sha": git_sha(short=False),
    }, indent=2))
    print(f"\n[qft_unfreeze] done. Manifest: {out_base / 'manifest.json'}")
    return 0


# ===========================================================================
# dispatch
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="experiment", required=True,
                                metavar="{l1-reg,l1-anchor,unfreeze}")
    _add_l1_reg(sub.add_parser(
        "l1-reg", help="identity-reg lambda sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter))
    _add_l1_anchor(sub.add_parser(
        "l1-anchor", help="L1-to-init-anchor across bases",
        formatter_class=argparse.RawDescriptionHelpFormatter))
    _add_unfreeze(sub.add_parser(
        "unfreeze", help="progressive gate-unfreezing",
        formatter_class=argparse.RawDescriptionHelpFormatter))
    args = parser.parse_args()

    # GPU isolation BEFORE the deferred pdft/JAX imports inside each _run_*.
    if args.gpu is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    return {
        "l1-reg": _run_l1_reg,
        "l1-anchor": _run_l1_anchor,
        "unfreeze": _run_unfreeze,
    }[args.experiment](args)


if __name__ == "__main__":
    sys.exit(main())
