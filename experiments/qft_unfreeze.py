#!/usr/bin/env python3
"""Progressive gate-unfreezing sweep for the identity-init QFT basis.

Runs `train_progressive_unfreeze` once per unfreeze ordering (bg/lr/rl) on a
chosen dataset, writing one output subtree per ordering plus an aggregate
manifest. See docs/superpowers/specs/2026-05-27-qft-unfreeze-design.md.

Usage:
    python experiments/qft_unfreeze.py --gpu 0 --dataset quickdraw_5q
    python experiments/qft_unfreeze.py --gpu 0 --dataset div2k_8q --max-steps 800
    python experiments/qft_unfreeze.py --gpu 1 --dataset tuberlin_8q --max-steps 800
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--dataset", default="quickdraw_5q",
                   choices=["quickdraw_5q", "div2k_8q", "tuberlin_8q"])
    p.add_argument("--orderings", default="bg,lr,rl")
    p.add_argument("--batch", type=int, default=None,
                   help="Fixed batch size. Default: full train set at m=5, else 50.")
    p.add_argument("--lr", type=float, default=None, help="Default: preset.lr_peak.")
    p.add_argument("--grad-tol", type=float, default=1e-5)
    p.add_argument("--loss-tol", type=float, default=1e-5)
    p.add_argument("--min-steps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--psnr-every", type=int, default=1,
                   help="Evaluate per-stage test PSNR only every K stages (the "
                        "final stage is always evaluated). K>1 cuts the dominant "
                        "m=8 eval cost; the loss/grad-norm staircase is unaffected.")
    p.add_argument("--grad-check-every", type=int, default=1,
                   help="Run the Riemannian grad-norm probe only every K steps "
                        "(loss-delta plateau uses the Adam step's free loss every "
                        "step). K>1 roughly halves the per-step cost at m=8.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--init", default="identity", choices=["identity", "random"],
                   help="Gate initialisation: QFT-family identity (default) or "
                        "Haar-random (H slots -> Haar U(2), CP slots -> uniform phase).")
    p.add_argument("--init-seed", type=int, default=None,
                   help="Seed for --init random (default: preset.seed). Shared "
                        "across orderings so they start from the same basis.")
    p.add_argument("--preset", default="generalized",
                   choices=["smoke", "moderate", "generalized"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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

    out_base = Path(args.out) if args.out else Path(f"results/qft_unfreeze/{args.dataset}")
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


if __name__ == "__main__":
    sys.exit(main())
