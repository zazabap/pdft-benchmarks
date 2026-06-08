#!/usr/bin/env python3
"""QFT training top-k sweep: does the training compression rate matter?

Trains analytic-init ``QFTBasis(m, n)`` at several *training-k* ratios — the
``MSELoss`` top-k truncation count expressed as a fraction of the ``2**(m+n)``
coefficients — and evaluates each trained basis at *all* eval keep-ratios. The
headline question: is each eval compression rate best served by a QFT trained at
a matching training-k, or does the current fixed 10% dominate everywhere?

Training-k ratio ``r`` -> ``k_train = max(1, round(2**(m+n) * r))``
(``experiment_utils.train_k_for``). Every other hyperparameter matches the
headline ``qft`` cell — analytic QFT init, the ``generalized`` preset, the
1008-step budget (``--epochs 112``, early stopping disabled), seed 42 — so the
``r = 0.10`` run reproduces the headline QFT PSNRs and acts as a built-in sanity
check (DIV2K-8q: 25.09 / 27.57 / 29.53 / 31.29 dB at rho = 0.05/0.10/0.15/0.20).

Standalone driver: does NOT use ``run_experiment``. One cell per training-k at
``results/training/3_training_topk/<dataset>/_runs/train_k<pct>/`` (standard cell
schema: metrics.json, env.json, trained_*.json, loss_history/). An aggregate
manifest with the full train-k x eval-rho PSNR matrix lands at
``results/training/3_training_topk/<dataset>/manifest.json``.

GPU isolation: ``--gpu`` sets ``CUDA_VISIBLE_DEVICES`` BEFORE importing
pdft_benchmarks (which transitively imports JAX, which preallocates GPU memory).

Usage:
    python experiments/qft_topk_sweep.py --gpu 0 --dataset div2k_8q
    python experiments/qft_topk_sweep.py --gpu 1 --dataset quickdraw_5q
    python experiments/qft_topk_sweep.py --gpu 0 --dataset div2k_8q \\
        --train-keep-ratios 0.05,0.10,0.15,0.20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

# Headline `qft` cell PSNRs (DIV2K-8q, trained at the historical 10% top-k).
# Used as the manifest reference anchor and to sanity-check the r=0.10 run.
DIV2K_QFT_ANCHOR = {"0.05": 25.093, "0.1": 27.572, "0.15": 29.532, "0.2": 31.294}


def _parse_ratios(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. Sets CUDA_VISIBLE_DEVICES before any pdft/jax import.")
    parser.add_argument("--dataset", default="div2k_8q",
                        choices=["div2k_8q", "quickdraw_5q"],
                        help="Dataset + qubit config: div2k_8q (m=n=8, 256x256) "
                             "or quickdraw_5q (m=n=5, 32x32).")
    parser.add_argument("--train-keep-ratios", default="0.05,0.10,0.15,0.20",
                        help="Comma-separated training-k ratios. One trained QFT "
                             "per ratio. Default 0.05,0.10,0.15,0.20.")
    parser.add_argument("--eval-keep-ratios", default="0.05,0.10,0.15,0.20",
                        help="Comma-separated eval keep-ratios; every trained QFT "
                             "is evaluated at all of them. Default matches the "
                             "headline eval grid.")
    parser.add_argument("--preset", default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--epochs", type=int, default=112,
                        help="Epoch budget per training-k (default 112 = 1008 "
                             "steps, the headline budget). Early stopping disabled.")
    parser.add_argument("--out-base", default=None,
                        help="Parent for per-ratio cells. Default "
                             "results/training/3_training_topk/<dataset>/_runs.")
    parser.add_argument("--force", action="store_true",
                        help="Retrain every ratio even if its cell already exists; "
                             "otherwise resume by reading existing metrics.json.")
    args = parser.parse_args()

    if args.gpu is not None:
        # Pin by nvidia-smi index. CUDA's default device order is fastest-first,
        # which scrambles indices on mixed-GPU hosts (this box has A6000 +
        # RTX 6000 Ada cards), so a bare CUDA_VISIBLE_DEVICES=N can land on a
        # different physical GPU than nvidia-smi's GPU N. PCI_BUS_ID makes
        # --gpu N select nvidia-smi's GPU N.
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # IMPORTANT: imports after env var so JAX picks up the isolated device.
    import jax
    import pdft
    import pdft.io  # noqa: F401 — needed by evaluate_basis_shared
    from pdft_benchmarks.datasets import load_div2k, load_quickdraw
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.experiment_utils import git_sha, serialize_tensors, train_k_for
    from pdft_benchmarks.presets import get_preset

    exp = "qft_topk_sweep"
    train_ratios = _parse_ratios(args.train_keep_ratios)
    eval_ratios = tuple(_parse_ratios(args.eval_keep_ratios))

    # GPU fail-fast: when --gpu N was passed, refuse to silently run on CPU
    # (CLAUDE.md notes NVML init failures can cause this).
    devices = jax.devices()
    chosen = devices[0]
    print(f"[{exp}] JAX devices: {devices}")
    print(f"[{exp}] chosen device: {chosen} (platform={chosen.platform!r})")
    if args.gpu is not None and chosen.platform not in ("gpu", "cuda"):
        print(
            f"[{exp}] FATAL: --gpu {args.gpu} requested but JAX sees only "
            f"platform={chosen.platform!r} (NVML init failure?). Aborting to "
            f"avoid a silent CPU run.",
            file=sys.stderr,
        )
        return 2

    preset = get_preset(args.dataset, args.preset)
    preset = replace(preset, epochs=args.epochs, early_stopping_patience=10**9)

    DATASET_CFG = {
        "div2k_8q":     (8, load_div2k, "size"),
        "quickdraw_5q": (5, load_quickdraw, "img_size"),
    }
    m_qubits, dataset_loader, size_kw = DATASET_CFG[args.dataset]
    m = n = m_qubits
    train_imgs_np, test_imgs_np = dataset_loader(
        n_train=preset.n_train, n_test=preset.n_test,
        seed=preset.seed, **{size_kw: 2 ** m},
    )
    print(f"[{exp}] dataset={args.dataset}, m=n={m}, epochs={preset.epochs}, "
          f"seed={preset.seed}, {len(train_imgs_np)} train / "
          f"{len(test_imgs_np)} test images")
    print(f"[{exp}] train-k ratios: {train_ratios}  ->  k = "
          f"{[train_k_for(m, n, r) for r in train_ratios]}")
    print(f"[{exp}] eval keep-ratios: {list(eval_ratios)}")

    out_base = Path(args.out_base) if args.out_base else \
        Path(f"results/training/3_training_topk/{args.dataset}/_runs")
    out_base.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for r in train_ratios:
        pct = round(r * 100)
        cell_tag = f"train_k{pct:02d}"
        basis_name = f"qft_topk{pct:02d}"
        out_dir = out_base / cell_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "loss_history").mkdir(parents=True, exist_ok=True)
        k_train = train_k_for(m, n, r)

        metrics_path = out_dir / "metrics.json"
        trained_path = out_dir / f"trained_{basis_name}.json"

        if metrics_path.exists() and trained_path.exists() and not args.force:
            # Resume: reuse the existing durable cell.
            cell = json.loads(metrics_path.read_text())[basis_name]
            psnr = {er: float(cell["metrics"][er]["mean_psnr"]) for er in cell["metrics"]}
            elapsed = float(cell["time"])
            steps = int(cell["_pdft_py"]["steps"])
            print(f"\n[{exp}] === train-k {r:.2f} (k={k_train}): RESUMED from "
                  f"existing cell ===")
        else:
            # Train analytic-init QFT at this training-k; everything else fixed.
            basis = pdft.QFTBasis(m=m, n=n)
            print(f"\n[{exp}] === train-k {r:.2f} (k={k_train}, {len(basis.tensors)} "
                  f"gates) -> {out_dir} ===")
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
            )
            elapsed = time.perf_counter() - t0
            steps = int(result.steps)
            epochs_completed = int(result.epochs_completed)
            print(f"[{exp}]   trained in {elapsed:.1f}s, steps={steps}, "
                  f"epochs={epochs_completed}")

            eval_metrics, _ = evaluate_basis_shared(
                result.basis, test_imgs_np, keep_ratios=eval_ratios,
            )
            psnr = {er: float(eval_metrics[er]["mean_psnr"]) for er in eval_metrics}
            print(f"[{exp}]   PSNR by eval-rho: " + ", ".join(
                f"{er}={p:.3f}" for er, p in psnr.items()))

            # Persist trained tensors first (durable checkpoint), then JSON.
            trained_path.write_text(json.dumps({
                "basis": basis_name,
                "train_ratio": r,
                "train_k": int(k_train),
                "m": int(result.basis.m), "n": int(result.basis.n),
                "tensors": serialize_tensors(result.basis.tensors),
            }, indent=2))
            metrics_path.write_text(json.dumps({
                basis_name: {
                    "metrics": eval_metrics,
                    "time": elapsed,
                    "_pdft_py": {
                        "train_ratio": r,
                        "train_k": int(k_train),
                        "steps": steps,
                        "epochs_completed": epochs_completed,
                        "device": str(jax.devices()[0]),
                        "n_test": int(len(test_imgs_np)),
                    },
                }
            }, indent=2))
            (out_dir / "loss_history" / f"{basis_name}_loss.json").write_text(json.dumps({
                "step_losses": [float(x) for x in result.loss_history],
                "val_losses": [float(x) for x in result.val_history],
                "epochs_completed": epochs_completed,
                "steps": steps,
            }, indent=2))
            (out_dir / "env.json").write_text(json.dumps({
                "experiment": exp,
                "dataset": args.dataset,
                "train_ratio": r,
                "train_k": int(k_train),
                "epochs_used": epochs_completed,
                "steps_used": steps,
                "preset_name": args.preset,
                "preset_epochs": int(args.epochs),
                "device": str(jax.devices()[0]),
                "git_sha": git_sha(short=False),
            }, indent=2))

        rows.append({
            "train_ratio": r,
            "train_k": int(k_train),
            "cell": cell_tag,
            "psnr": psnr,
            "steps": int(steps),
            "elapsed_seconds": float(elapsed),
        })

    manifest_path = out_base.parent / "manifest.json"
    manifest_path.write_text(json.dumps({
        "experiment": exp,
        "dataset": args.dataset,
        "m": m, "n": n,
        "train_keep_ratios": train_ratios,
        "eval_keep_ratios": list(eval_ratios),
        "epochs": int(args.epochs),
        "rows": rows,
        "headline_qft_anchor": DIV2K_QFT_ANCHOR if args.dataset == "div2k_8q" else {},
        "git_sha": git_sha(short=False),
    }, indent=2))

    # Summary: train-k (rows) x eval-rho (cols) PSNR matrix.
    eval_keys = [k for k in rows[0]["psnr"]] if rows else []
    print(f"\n[{exp}] sweep complete. Manifest: {manifest_path}")
    print(f"[{exp}] PSNR (dB) — rows: train-k, cols: eval-rho")
    print("  train-k \\ rho   " + "  ".join(f"{k:>7s}" for k in eval_keys))
    for row in rows:
        cells = "  ".join(f"{row['psnr'][k]:7.3f}" for k in eval_keys)
        print(f"  k={row['train_ratio']:.2f} (n={row['train_k']:>5d})  {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
