#!/usr/bin/env python3
"""Drive the 8-stage qft_progressive curriculum on DIV2K-8q.

Trains QFT(8, 8) from identity init via progressive unfreezing: at each
stage k=1..8, the basis is BlockedBasis(QFTBasis(k, k), 8-k, 8-k) for
k<8 and bare QFTBasis(8, 8) for k=8. Inner gates carry forward from the
previous stage via pdft_benchmarks.bases.qft_warm_from_smaller_qft;
newly introduced gates at each stage start at their identity element.

Standalone driver: does NOT use pdft_benchmarks.run_experiment. Cells
land at results/qft_progressive/div2k_8q/_runs/stage_k<k>/ with the
standard cell schema. An aggregate manifest is written at
results/qft_progressive/div2k_8q/manifest.json.

Usage:
    python experiments/qft_progressive.py --gpu 0 [--epochs-per-stage 56]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_trained_qftbasis_checkpoint(path: "Path", expected_k: int):
    """Reconstruct a `pdft.QFTBasis(k, k)` from a trained_qft_progressive_k<k>.json
    checkpoint file written by an earlier stage. Used by the resume-from-checkpoint
    path in main().

    The on-disk schema is:
      {"stage_k": k, "m": k, "n": k, "tensors": [{"real": [[...]], "imag": [[...]]}, ...]}
    """
    import jax.numpy as jnp
    import numpy as np
    import pdft

    data = json.loads(path.read_text())
    m, n = int(data["m"]), int(data["n"])
    if m != expected_k or n != expected_k:
        raise ValueError(
            f"checkpoint at {path} has m={m}, n={n}, expected m=n={expected_k}. "
            f"This indicates a corrupted or mismatched cell — delete "
            f"{path.parent} (or pass --force) before retrying."
        )
    tensors = [
        jnp.asarray(
            np.array(t["real"], dtype=np.float64)
            + 1j * np.array(t["imag"], dtype=np.float64),
            dtype=jnp.complex128,
        )
        for t in data["tensors"]
    ]
    return pdft.QFTBasis(m=m, n=n, tensors=tensors)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. Sets CUDA_VISIBLE_DEVICES before any pdft/jax import.")
    parser.add_argument("--epochs-per-stage", type=int, default=56,
                        help="Per-stage epoch budget. Default 56 -> 448 total epochs across 8 stages.")
    parser.add_argument("--out-base", type=str, default=None,
                        help="Parent for per-stage cells. Default results/qft_progressive/<dataset>/_runs.")
    parser.add_argument("--dataset", type=str, default="div2k_8q",
                        choices=["div2k_8q"],
                        help="Dataset + qubit config. div2k_8q only for now (spec scope).")
    parser.add_argument("--preset", type=str, default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--force", action="store_true", default=False,
                        help="Retrain every stage even if its trained_*.json already exists; "
                             "otherwise resume by loading existing trained tensors.")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # IMPORTANT: imports after env var so JAX picks up the device.
    import numpy as np
    import jax
    import pdft
    import pdft.io  # noqa: F401 — needed by evaluate_basis_shared
    from pdft_benchmarks.bases import (
        qft_identity_basis,
        qft_warm_from_smaller_qft,
    )
    from pdft_benchmarks.datasets import load_div2k
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.presets import get_preset

    # GPU fail-fast: when --gpu N was passed, refuse to silently fall back
    # to CPU (CLAUDE.md notes NVML init failures can cause this).
    devices = jax.devices()
    chosen = devices[0]
    print(f"[qft_progressive] JAX devices: {devices}")
    print(f"[qft_progressive] chosen device: {chosen} (platform={chosen.platform!r})")
    if args.gpu is not None and chosen.platform not in ("gpu", "cuda"):
        print(
            f"[qft_progressive] FATAL: --gpu {args.gpu} was requested but JAX "
            f"sees only platform={chosen.platform!r}. This typically means NVML "
            f"failed to initialise (see CLAUDE.md 'When something goes wrong'). "
            f"Aborting to avoid a silent CPU run.",
            file=sys.stderr,
        )
        return 2

    preset = get_preset(args.dataset, args.preset)
    preset = replace(preset, epochs=args.epochs_per_stage,
                     early_stopping_patience=10**9)
    print(f"[qft_progressive] dataset={args.dataset}, preset.epochs={preset.epochs} "
          f"per stage, early_stopping disabled, seed={preset.seed}")

    m = n = 8
    train_imgs_np, test_imgs_np = load_div2k(
        n_train=preset.n_train, n_test=preset.n_test,
        seed=preset.seed, size=2**m,
    )
    k_train = max(1, round(2 ** (m + n) * 0.1))
    print(f"[qft_progressive] m=n={m}, k_train={k_train}, "
          f"{len(train_imgs_np)} train images, {len(test_imgs_np)} test images")

    out_base = Path(args.out_base) if args.out_base else \
        Path(f"results/qft_progressive/{args.dataset}/_runs")
    out_base.mkdir(parents=True, exist_ok=True)

    prev_inner = None
    prev_cell_path = None
    prev_cell_sha = None
    stage_summaries: list[dict] = []

    for k in range(1, 9):
        stage_tag = f"stage_k{k}"
        basis_name = f"qft_progressive_k{k}"
        out_dir = out_base / stage_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "loss_history").mkdir(parents=True, exist_ok=True)

        trained_path = out_dir / f"trained_{basis_name}.json"
        metrics_path = out_dir / "metrics.json"

        if trained_path.exists() and metrics_path.exists() and not args.force:
            # Resume path: load existing trained inner tensors and existing metrics.
            inner_k = _load_trained_qftbasis_checkpoint(trained_path, expected_k=k)
            n_trainable = len(inner_k.tensors)
            existing_metrics = json.loads(metrics_path.read_text())
            if basis_name not in existing_metrics:
                raise RuntimeError(
                    f"metrics.json at {metrics_path} missing expected key "
                    f"{basis_name!r}; pass --force to retrain or delete the cell."
                )
            cell_data = existing_metrics[basis_name]
            psnr20 = float(cell_data["metrics"]["0.2"]["mean_psnr"])
            elapsed = float(cell_data["time"])
            steps = int(cell_data["_pdft_py"]["steps"])
            epochs_completed = int(cell_data["_pdft_py"]["epochs_completed"])
            print(f"\n[qft_progressive] === stage k={k}: RESUMED from existing cell "
                  f"({n_trainable} trainable gates, block size {2**k}x{2**k}) ===")
            print(f"[qft_progressive]   PSNR @ rho=0.20: {psnr20:.3f} dB "
                  f"(from existing metrics.json)")
            inner_trained = inner_k
            # Do NOT rewrite trained_*.json, metrics.json, env.json, or loss_history
            # — they already exist and are durable. The SHA chain will be computed
            # below from the on-disk file, so consistency is preserved.
        else:
            # Train path: build basis, train, evaluate, persist.
            if prev_inner is None:
                inner_k = qft_identity_basis(m=k, n=k)
            else:
                inner_k = qft_warm_from_smaller_qft(prev_inner)
            if k < 8:
                basis = pdft.BlockedBasis(inner=inner_k,
                                          block_log_m=8 - k,
                                          block_log_n=8 - k)
            else:
                basis = inner_k

            n_trainable = len(inner_k.tensors)
            print(f"\n[qft_progressive] === stage k={k} ({n_trainable} trainable gates, "
                  f"block size {2**k}x{2**k}) -> {out_dir} ===")

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
            print(f"[qft_progressive]   trained in {elapsed:.1f}s, "
                  f"steps={steps}, epochs={epochs_completed}")

            eval_metrics, _ = evaluate_basis_shared(
                result.basis, test_imgs_np,
                keep_ratios=(0.05, 0.10, 0.15, 0.20),
            )
            psnr20 = float(eval_metrics["0.2"]["mean_psnr"])
            print(f"[qft_progressive]   PSNR @ rho=0.20: {psnr20:.3f} dB")

            # Persist trained tensors FIRST so that even if subsequent JSON
            # writes fail, the durable checkpoint exists.
            if k < 8:
                inner_trained = result.basis.inner
            else:
                inner_trained = result.basis
            trained_path.write_text(json.dumps({
                "stage_k": k,
                "m": int(inner_trained.m),
                "n": int(inner_trained.n),
                "tensors": [{"real": np.asarray(t).real.tolist(),
                             "imag": np.asarray(t).imag.tolist()}
                            for t in inner_trained.tensors],
            }, indent=2))

            metrics_path.write_text(json.dumps({
                basis_name: {
                    "metrics": eval_metrics,
                    "time": elapsed,
                    "_pdft_py": {
                        "stage_k": k,
                        "n_trainable": int(n_trainable),
                        "block_size": int(2**k),
                        "steps": steps,
                        "epochs_completed": epochs_completed,
                        "device": str(jax.devices()[0]),
                        "n_test": int(len(test_imgs_np)),
                    }
                }
            }, indent=2))

            (out_dir / "loss_history" / f"{basis_name}_loss.json").write_text(json.dumps({
                "step_losses": [float(x) for x in result.loss_history],
                "val_losses": [float(x) for x in result.val_history],
                "epochs_completed": epochs_completed,
                "steps": steps,
            }, indent=2))

            (out_dir / "env.json").write_text(json.dumps({
                "experiment": "qft_progressive",
                "stage_k": k,
                "epochs_used": epochs_completed,
                "steps_used": steps,
                "n_trainable": int(n_trainable),
                "block_size": int(2**k),
                "prev_cell_path": prev_cell_path,
                "prev_cell_sha256": prev_cell_sha,
                "preset_name": args.preset,
                "preset_epochs_per_stage": int(args.epochs_per_stage),
                "device": str(jax.devices()[0]),
                "git_sha": _git_sha(),
            }, indent=2))

        # COMMON path (both resume and train): update carry-forward state + manifest.
        prev_inner = inner_trained
        prev_cell_path = str(trained_path)
        prev_cell_sha = _sha256_file(trained_path)

        stage_summaries.append({
            "k": k,
            "n_trainable": int(n_trainable),
            "block_size": int(2**k),
            "cell": stage_tag,
            "psnr_rho_020": float(psnr20),
            "steps": int(steps),
            "elapsed_seconds": float(elapsed),
        })

    manifest_path = out_base.parent / "manifest.json"
    manifest_path.write_text(json.dumps({
        "experiment": "qft_progressive",
        "dataset": args.dataset,
        "epochs_per_stage": int(args.epochs_per_stage),
        "total_epochs": int(args.epochs_per_stage * 8),
        "stages": stage_summaries,
        "anchors": {"qft": 31.29, "qft_identity": 31.66, "blocked_8": 32.26},
        "git_sha": _git_sha(),
    }, indent=2))

    print(f"\n[qft_progressive] sweep complete. Manifest: {manifest_path}")
    print("[qft_progressive] PSNR @ rho=0.20 by stage:")
    for s in stage_summaries:
        print(f"  k={s['k']} ({s['n_trainable']:>2d} gates): {s['psnr_rho_020']:.3f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
