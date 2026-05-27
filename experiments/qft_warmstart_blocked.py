#!/usr/bin/env python3
"""Warm-start full QFT(m, n) from a TRAINED BlockedBasis(QFT(m_in, n_in), ...)
and re-train all m+n axis parameters from there.

Demonstrates that the trained blocked optimum is reachable from the larger
QFT family via warm-start: after re-training, the unblocked QFT(m, n) retains
the trained blocked PSNR (within run-to-run noise) — i.e. training from the
warm-start does not push the operator out of the blocked basin.

Pipeline (self-contained, does NOT use pdft_benchmarks.run_experiment):

  1. Load the previously-trained blocked basis from
       results/<src_experiment>/by_basis/<blocked_name>/trained_<blocked_name>.json
  2. Construct a QFTBasis(m, n) whose initial operator equals the trained
     blocked one bit-exactly (via bases.qft_warm_from_trained_blocked).
  3. Sanity-check bit-exactness on a random complex test input.
  4. Train under the same preset/budget the headline experiment used, with
     all m+n axis parameters trainable.
  5. Evaluate at the headline keep-ratios and write a cell layout matching
     the existing by_basis convention.

Outputs land at:
  results/qft_warmstart_from_trained_blocked/by_basis/<basis_name>/
    metrics.json, env.json, trained_<basis_name>.json, loss_history/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# (dataset, m, n, src trained_*.json, src dataset-loader name, output basis name)
DATASETS: dict[str, dict] = {
    "div2k_8q": {
        "m": 8, "n": 8,
        "load_dataset": "div2k",
        "preset_namespace": "div2k_8q",
        "trained_blocked_path": (
            "results/div2k_8q_pca_vs_block_dct/by_basis/blocked_8/"
            "trained_blocked_8.json"
        ),
        "output_basis_name": "qft_warmstart_blocked_8",
    },
    "quickdraw": {
        "m": 5, "n": 5,
        "load_dataset": "quickdraw",
        "preset_namespace": "quickdraw",
        "trained_blocked_path": (
            "results/quickdraw_pca_vs_block_dct/by_basis/blocked/"
            "trained_blocked.json"
        ),
        "output_basis_name": "qft_warmstart_blocked",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. Sets CUDA_VISIBLE_DEVICES before JAX import.")
    parser.add_argument(
        "--out",
        default="results/qft_warmstart_from_trained_blocked",
        help="Output root. Cell will land at <out>/by_basis/<basis_name>/.",
    )
    parser.add_argument("--epochs", type=int, default=112,
                        help="Headline budget = 112 epochs ≈ 1008 steps.")
    parser.add_argument("--preset", default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    args = parser.parse_args()

    # CRITICAL: set CUDA_VISIBLE_DEVICES BEFORE importing pdft_benchmarks.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = DATASETS[args.dataset]

    # Imports below trigger JAX device discovery; must come AFTER the env var.
    import json
    import time
    from dataclasses import replace
    from datetime import datetime, timezone

    import jax
    import jax.numpy as jnp
    import numpy as np

    import pdft  # noqa: F401  -- ensures jax_enable_x64

    from pdft_benchmarks._loading import load_trained_basis
    from pdft_benchmarks._training import train_one_basis_batched
    from pdft_benchmarks.bases import qft_warm_from_trained_blocked
    from pdft_benchmarks.datasets import load as load_dataset
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.pipeline import _select_device, _git_sha, _git_branch
    from pdft_benchmarks.presets import get_preset

    out_root = Path(args.out)
    basis_name = cfg["output_basis_name"]
    cell_dir = out_root / "by_basis" / basis_name
    cell_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trained blocked basis.
    trained_blocked_path = Path(cfg["trained_blocked_path"])
    if not trained_blocked_path.is_file():
        print(f"[warmstart] missing trained blocked file: {trained_blocked_path}",
              file=sys.stderr)
        return 2
    print(f"[warmstart] loading trained blocked basis from {trained_blocked_path}")
    trained_blocked = load_trained_basis(trained_blocked_path)
    print(f"[warmstart]   inner type={type(trained_blocked.inner).__name__}, "
          f"inner m={trained_blocked.inner.m} n={trained_blocked.inner.n}, "
          f"block_log_m={trained_blocked.block_log_m} block_log_n={trained_blocked.block_log_n}")

    # 2. Construct warm-start QFT.
    warm_qft_init = qft_warm_from_trained_blocked(trained_blocked)
    print(f"[warmstart] constructed warm-start QFTBasis: m={warm_qft_init.m} "
          f"n={warm_qft_init.n}, num_tensors={len(warm_qft_init.tensors)}")

    # 3. Sanity check: forward transforms must agree bit-exactly.
    rng = np.random.default_rng(0)
    side_m, side_n = 2 ** cfg["m"], 2 ** cfg["n"]
    x = rng.standard_normal((side_m, side_n)) + 1j * rng.standard_normal((side_m, side_n))
    x_jax = jnp.asarray(x, dtype=jnp.complex128)
    y_blocked = trained_blocked.forward_transform(x_jax)
    y_warm = warm_qft_init.forward_transform(x_jax)
    diff = float(jnp.max(jnp.abs(y_blocked - y_warm)))
    if diff > 1e-10:
        print(f"[warmstart] bit-exact check FAILED: max |diff| = {diff:.2e}",
              file=sys.stderr)
        return 3
    print(f"[warmstart] bit-exact check PASSED: max |diff| = {diff:.2e}")

    # 4. Resolve preset + dataset.
    preset = get_preset(cfg["preset_namespace"], args.preset)
    preset = replace(preset, epochs=args.epochs, early_stopping_patience=10**9)
    print(f"[warmstart] preset: epochs={preset.epochs}, batch_size={preset.batch_size}, "
          f"n_train={preset.n_train}, n_test={preset.n_test}, "
          f"optimizer={preset.optimizer}")

    train_imgs, test_imgs = load_dataset(
        cfg["load_dataset"],
        n_train=preset.n_train,
        n_test=preset.n_test,
        seed=preset.seed,
    )
    print(f"[warmstart] loaded train={np.asarray(train_imgs).shape if hasattr(train_imgs, 'shape') else len(train_imgs)}, "
          f"test={np.asarray(test_imgs).shape if hasattr(test_imgs, 'shape') else len(test_imgs)}")

    selected_device = _select_device("auto")
    print(f"[warmstart] device={selected_device}")

    # 5. Train. The factory must reconstruct the warm-start basis FRESH
    # because train_one_basis_batched is called twice (warmup + real run)
    # and we want the same starting point both times.
    def factory():
        return qft_warm_from_trained_blocked(trained_blocked)

    t0 = time.perf_counter()
    res = train_one_basis_batched(factory, train_imgs, preset, device=selected_device)
    train_elapsed = time.perf_counter() - t0
    print(f"[warmstart] training done in {train_elapsed:.1f}s "
          f"(epochs_completed={res.epochs_completed}, steps={res.steps})")

    # 6. Evaluate at headline keep-ratios.
    host_basis = jax.tree_util.tree_map(jax.device_get, res.basis)
    kr_metrics, nan_counts = evaluate_basis_shared(host_basis, test_imgs, preset.keep_ratios)
    psnr_summary = {kr: m["mean_psnr"] for kr, m in kr_metrics.items()}
    print(f"[warmstart] PSNR by keep-ratio: " + ", ".join(
        f"ρ={kr}: {p:.2f} dB" for kr, p in psnr_summary.items()
    ))

    # 7. Write cell outputs in the same format the headline pipeline uses.
    (cell_dir / "loss_history").mkdir(parents=True, exist_ok=True)
    (cell_dir / "loss_history" / f"{basis_name}_loss.json").write_text(json.dumps({
        "step_losses": list(res.loss_history),
        "val_losses": list(res.val_history),
        "epochs_completed": res.epochs_completed,
        "steps": res.steps,
    }))

    host_tensors = [jax.device_get(t) for t in res.basis.tensors]
    (cell_dir / f"trained_{basis_name}.json").write_text(json.dumps({
        "type": type(res.basis).__name__,
        "m": int(res.basis.m),
        "n": int(res.basis.n),
        "tensors": [
            [[float(v.real), float(v.imag)] for v in np.asarray(t).flatten(order="F")]
            for t in host_tensors
        ],
    }, indent=2))

    metrics_payload = {
        basis_name: {
            "metrics": kr_metrics,
            "time": res.time,
            "_pdft_py": {
                "warmup_s": res.warmup_s,
                "device": str(selected_device),
                "epochs_completed": res.epochs_completed,
                "steps": res.steps,
                "n_test": len(test_imgs),
                "eval_failed_count": nan_counts,
                "warm_started_from": str(trained_blocked_path),
                "warmstart_bit_exact_max_diff": diff,
            },
        }
    }
    (cell_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    env = {
        "jax_version": jax.__version__,
        "default_backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "active_device": str(selected_device),
        "git_sha": _git_sha(),
        "git_branch": _git_branch(),
        "pdft_upstream_ref": pdft.__upstream_ref__,
        "preset": preset.name,
        "preset_dataclass": {
            "epochs": preset.epochs,
            "n_train": preset.n_train,
            "n_test": preset.n_test,
            "optimizer": preset.optimizer,
            "batch_size": preset.batch_size,
            "warmup_frac": preset.warmup_frac,
            "lr_peak": preset.lr_peak,
            "lr_final": preset.lr_final,
            "max_grad_norm": preset.max_grad_norm,
            "validation_split": preset.validation_split,
            "early_stopping_patience": preset.early_stopping_patience,
            "seed": preset.seed,
            "keep_ratios": list(preset.keep_ratios),
        },
        "warm_started_from": str(trained_blocked_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (cell_dir / "env.json").write_text(json.dumps(env, indent=2))

    print(f"[warmstart] wrote cell -> {cell_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
