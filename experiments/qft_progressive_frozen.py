#!/usr/bin/env python3
"""Drive the 8-stage qft_progressive block-size sweep via QFT(8,8) + frozen_indices.

Reproduces the existing `qft_progressive` experiment via a strictly-more-general
primitive: instead of wrapping QFTBasis(k, k) in BlockedBasis(inner, 8-k, 8-k),
each stage trains a FULL QFTBasis(8, 8) at identity init with the gates outside
the QFT(k, k) inner sub-circuit frozen via `pdft.train_basis_batched(frozen_indices=...)`.

By operator equivalence — putting gates outside the inner k-qubit set at identity
yields the BlockedBasis(QFTBasis(k, k), 8-k, 8-k) operator — each stage should
produce a training trajectory and PSNR identical (within numerical noise) to the
corresponding qft_progressive stage.

For each stage k = 1..8:
  - basis = QFTBasis(8, 8) at identity init (full circuit, constant across stages)
  - inner_indices, outer_indices = qft_inner_outer_indices(m=8, n=8, inner_m=k, inner_n=k)
  - frozen_indices = outer_indices   (k < 8)
                   = None            (k = 8; outer_indices == [], treated as None)
  - Trained under the same headline preset as qft_progressive.

Stages are INDEPENDENT — no warm-start chain. Each k starts from identity init
and trains only the inner k-qubit sub-circuit gates.

Cell layout mirrors qft_progressive so render_qft_progressive.py works with
a --results-base override:
  results/qft_progressive/frozen_replication/div2k_8q/_runs/stage_k<k>/

Usage:
    python experiments/qft_progressive_frozen.py --gpu 0 [--epochs-per-stage 112]
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


def _load_trained_full_qftbasis_checkpoint(path: Path):
    """Reconstruct a QFTBasis(8, 8) from a trained_qft_progressive_k<k>.json
    checkpoint written by an earlier stage of this driver.

    On-disk schema:
      {
        "stage_k": k,
        "m": 8, "n": 8,
        "inner_m": k, "inner_n": k,
        "frozen_indices": [...],
        "tensors": [{"real": [...], "imag": [...]}, ...]   # 72 entries
      }
    """
    import jax.numpy as jnp
    import numpy as np
    import pdft

    data = json.loads(path.read_text())
    m, n = int(data["m"]), int(data["n"])
    if m != 8 or n != 8:
        raise ValueError(
            f"checkpoint at {path} has m={m}, n={n}, expected m=n=8. "
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
    return pdft.QFTBasis(m=8, n=8, tensors=tensors)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. Sets CUDA_VISIBLE_DEVICES before any pdft/jax import.")
    parser.add_argument("--epochs-per-stage", type=int, default=112,
                        help="Per-stage epoch budget. Default 112 -> 1008 steps per stage.")
    parser.add_argument("--out-base", type=str, default=None,
                        help="Parent for per-stage cells. Default "
                             "results/qft_progressive/frozen_replication/<dataset>/_runs.")
    parser.add_argument("--dataset", type=str, default="div2k_8q",
                        choices=["div2k_8q"],
                        help="Dataset + qubit config. div2k_8q only (spec scope).")
    parser.add_argument("--preset", type=str, default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--force", action="store_true", default=False,
                        help="Retrain every stage even if its trained_*.json already exists; "
                             "otherwise resume by loading existing trained tensors.")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # IMPORTANT: imports after env var so JAX picks up the correct device.
    import numpy as np
    import jax
    import jax.numpy as jnp
    import pdft
    import pdft.io  # noqa: F401 — needed by evaluate_basis_shared
    from pdft_benchmarks.bases import qft_identity_basis, qft_inner_outer_indices
    from pdft_benchmarks.datasets import load_div2k
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.presets import get_preset

    # GPU fail-fast: when --gpu N was passed, refuse to silently fall back to CPU.
    devices = jax.devices()
    chosen = devices[0]
    print(f"[qft_progressive_frozen] JAX devices: {devices}")
    print(f"[qft_progressive_frozen] chosen device: {chosen} (platform={chosen.platform!r})")
    if args.gpu is not None and chosen.platform not in ("gpu", "cuda"):
        print(
            f"[qft_progressive_frozen] FATAL: --gpu {args.gpu} was requested but JAX "
            f"sees only platform={chosen.platform!r}. This typically means NVML "
            f"failed to initialise (see CLAUDE.md 'When something goes wrong'). "
            f"Aborting to avoid a silent CPU run.",
            file=sys.stderr,
        )
        return 2

    preset = get_preset(args.dataset, args.preset)
    preset = replace(preset, epochs=args.epochs_per_stage,
                     early_stopping_patience=10**9)
    print(f"[qft_progressive_frozen] dataset={args.dataset}, "
          f"preset.epochs={preset.epochs} per stage, "
          f"early_stopping disabled, seed={preset.seed}")

    m = n = 8
    train_imgs_np, test_imgs_np = load_div2k(
        n_train=preset.n_train, n_test=preset.n_test,
        seed=preset.seed, size=2**m,
    )
    k_train = max(1, round(2 ** (m + n) * 0.1))
    print(f"[qft_progressive_frozen] m=n={m}, k_train={k_train}, "
          f"{len(train_imgs_np)} train images, {len(test_imgs_np)} test images")

    out_base = Path(args.out_base) if args.out_base else \
        Path(f"results/qft_progressive/frozen_replication/{args.dataset}/_runs")
    out_base.mkdir(parents=True, exist_ok=True)

    # Pre-compute the identity reference tensors once — used for frozen-tensor
    # sanity checks after each stage.
    ref_basis = qft_identity_basis(m=m, n=n)
    ref_tensors = ref_basis.tensors  # 72 tensors in H-first canonical order

    # Each stage is INDEPENDENT — no carry-forward state.
    stage_summaries: list[dict] = []

    for k in range(1, 9):
        stage_tag = f"stage_k{k}"
        basis_name = f"qft_progressive_k{k}"
        out_dir = out_base / stage_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "loss_history").mkdir(parents=True, exist_ok=True)

        trained_path = out_dir / f"trained_{basis_name}.json"
        metrics_path = out_dir / "metrics.json"

        # Compute frozen/inner partition for this k.
        inner_indices, outer_indices = qft_inner_outer_indices(
            m=m, n=n, inner_m=k, inner_n=k
        )
        n_trainable = len(inner_indices)
        n_frozen = len(outer_indices)

        # For k=8: outer_indices == [] — pass None to avoid any validator
        # edge-cases (pdft treats [] as None anyway per the docstring, but
        # being explicit here is clearer).
        frozen_indices: list[int] | None = outer_indices if outer_indices else None

        if trained_path.exists() and metrics_path.exists() and not args.force:
            # Resume path: load existing trained tensors + metrics.
            full_trained = _load_trained_full_qftbasis_checkpoint(trained_path)
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
            print(f"\n[qft_progressive_frozen] === stage k={k}: RESUMED from existing cell "
                  f"({n_trainable} trainable / {n_frozen} frozen, "
                  f"inner block {2**k}x{2**k}) ===")
            print(f"[qft_progressive_frozen]   PSNR @ rho=0.20: {psnr20:.3f} dB "
                  f"(from existing metrics.json)")
        else:
            # Train path: build fresh QFTBasis(8, 8) at identity init, train
            # with outer gates frozen at identity.
            basis = qft_identity_basis(m=m, n=n)  # full QFT(8,8), all at identity

            print(f"\n[qft_progressive_frozen] === stage k={k} "
                  f"({n_trainable} trainable / {n_frozen} frozen gates, "
                  f"inner block {2**k}x{2**k}) -> {out_dir} ===")
            print(f"[qft_progressive_frozen]   inner_indices[:5]={inner_indices[:5]}, "
                  f"outer count={n_frozen}, frozen_indices={'None' if frozen_indices is None else f'[{n_frozen} indices]'}")

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
                frozen_indices=frozen_indices,
            )
            elapsed = time.perf_counter() - t0
            steps = int(result.steps)
            epochs_completed = int(result.epochs_completed)
            print(f"[qft_progressive_frozen]   trained in {elapsed:.1f}s, "
                  f"steps={steps}, epochs={epochs_completed}")

            # Sanity check: frozen tensors must be bit-exactly at identity init.
            if frozen_indices is not None:
                for i in frozen_indices:
                    diff = float(jnp.max(jnp.abs(result.basis.tensors[i] - ref_tensors[i])))
                    assert diff == 0.0, (
                        f"[qft_progressive_frozen] FATAL: frozen index {i} at stage k={k} "
                        f"drifted by {diff} — frozen_indices semantics broken!"
                    )
                print(f"[qft_progressive_frozen]   verified: all {n_frozen} frozen tensors "
                      f"are bit-exactly at identity.")
            else:
                # k=8: no frozen gates — whole circuit was trained freely.
                print(f"[qft_progressive_frozen]   k=8: frozen_count=0 "
                      f"(full QFT(8,8) trained without freezing).")

            eval_metrics, _ = evaluate_basis_shared(
                result.basis, test_imgs_np,
                keep_ratios=(0.05, 0.10, 0.15, 0.20),
            )
            psnr20 = float(eval_metrics["0.2"]["mean_psnr"])
            Lf = float(result.val_history[-1]) if len(result.val_history) > 0 else float("nan")
            print(f"[qft_progressive_frozen]   PSNR @ rho=0.20: {psnr20:.3f} dB, "
                  f"val MSE final: {Lf:.6f}")

            # Persist trained tensors FIRST — durable checkpoint even if
            # subsequent JSON writes fail.
            full_trained = result.basis  # QFTBasis(8, 8)
            trained_path.write_text(json.dumps({
                "stage_k": k,
                "m": int(full_trained.m),
                "n": int(full_trained.n),
                "inner_m": k,
                "inner_n": k,
                "frozen_indices": frozen_indices if frozen_indices is not None else [],
                "tensors": [{"real": np.asarray(t).real.tolist(),
                             "imag": np.asarray(t).imag.tolist()}
                            for t in full_trained.tensors],
            }, indent=2))

            metrics_path.write_text(json.dumps({
                basis_name: {
                    "metrics": eval_metrics,
                    "time": elapsed,
                    "_pdft_py": {
                        "stage_k": k,
                        "n_trainable": int(n_trainable),
                        "n_frozen": int(n_frozen),
                        "inner_m": k,
                        "inner_n": k,
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
                "experiment": "qft_progressive_frozen",
                "stage_k": k,
                "epochs_used": epochs_completed,
                "steps_used": steps,
                "n_trainable": int(n_trainable),
                "frozen_count": int(n_frozen),
                "inner_m": k,
                "inner_n": k,
                "block_size": int(2**k),
                "init_policy": "identity",
                "preset_name": args.preset,
                "preset_epochs_per_stage": int(args.epochs_per_stage),
                "device": str(jax.devices()[0]),
                "git_sha": _git_sha(),
            }, indent=2))

        # COMMON path (both resume and train): record manifest summary.
        stage_summaries.append({
            "k": k,
            "n_trainable": int(n_trainable),
            "n_frozen": int(n_frozen),
            "block_size": int(2**k),
            "cell": stage_tag,
            "psnr_rho_020": float(psnr20),
            "steps": int(steps),
            "elapsed_seconds": float(elapsed),
        })

    manifest_path = out_base.parent / "manifest.json"
    manifest_path.write_text(json.dumps({
        "experiment": "qft_progressive_frozen",
        "replication_of": "qft_progressive",
        "dataset": args.dataset,
        "epochs_per_stage": int(args.epochs_per_stage),
        "total_epochs": int(args.epochs_per_stage * 8),
        "stages": stage_summaries,
        "anchors": {"qft": 31.29, "qft_identity": 31.66, "blocked_8": 32.26},
        "git_sha": _git_sha(),
    }, indent=2))

    print(f"\n[qft_progressive_frozen] sweep complete. Manifest: {manifest_path}")
    print("[qft_progressive_frozen] PSNR @ rho=0.20 by stage:")
    for s in stage_summaries:
        print(f"  k={s['k']} ({s['n_trainable']:>2d} trainable / "
              f"{s['n_frozen']:>2d} frozen gates): {s['psnr_rho_020']:.3f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
