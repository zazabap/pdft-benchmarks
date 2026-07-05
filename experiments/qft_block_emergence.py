#!/usr/bin/env python3
"""Train a Haar-random QFTBasis(8,8) on DIV2K and snapshot the basis during
training, so we can watch the emergent block structure form.

The snapshots are taken WITHIN ONE continuous, faithful training run (correct
cosine-LR schedule + persistent Adam state) by wrapping the library's jitted
Adam step; restarting training every N steps would reset the schedule and the
optimizer moments and distort the dynamics. We dump the live basis tensors at
step 0 (the random init) and every --every steps through the final step, in the
trained_*.json {m, n, seed, tensors:[{real, imag}]} schema.

Usage:
    python experiments/qft_block_emergence.py --gpu 7 --seed 0 --epochs 112 \
        --out results/training/1_structure_inclusion/block_emergence
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0, help="Haar-random init seed.")
    ap.add_argument("--epochs", type=int, default=112, help="112 -> 1008 steps.")
    ap.add_argument("--every", type=int, default=50, help="snapshot cadence (steps).")
    ap.add_argument("--out", default="results/training/1_structure_inclusion/block_emergence")
    args = ap.parse_args()

    # Pin the GPU BEFORE any pdft / jax import (mirrors the DIV2K entry points).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pdft

    from pdft_benchmarks.bases import family_random_basis
    from pdft_benchmarks.datasets.div2k import load_div2k
    from pdft_benchmarks.experiment_utils import serialize_tensors
    from pdft_benchmarks.presets import get_preset

    m = n = 8
    preset = get_preset("div2k_8q", "generalized")
    out = Path(args.out)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_imgs, _ = load_div2k(n_train=preset.n_train, n_test=preset.n_test,
                               seed=preset.seed, size=2 ** m)
    basis = family_random_basis("qft", m, n, args.seed)

    # Total optimizer steps under the batched trainer's bookkeeping.
    import math
    n_train_eff = len(train_imgs) - int(np.clip(
        round(len(train_imgs) * preset.validation_split), 0, len(train_imgs) - 1))
    n_batches = math.ceil(n_train_eff / preset.batch_size)
    total_steps = max(1, args.epochs * n_batches)
    snap_steps = set(range(args.every, total_steps + 1, args.every)) | {total_steps}
    print(f"[emerge] seed={args.seed} epochs={args.epochs} "
          f"steps/epoch={n_batches} total_steps={total_steps} "
          f"snapshots={sorted({0} | snap_steps)}")

    def _dump(tensors, step):
        host = [jax.device_get(t) for t in tensors]
        (ckpt_dir / f"step_{step:05d}.json").write_text(json.dumps({
            "dataset": "div2k_8q", "key": "qft", "seed": args.seed,
            "m": m, "n": n, "step": step,
            "tensors": serialize_tensors(host),
        }))
        print(f"[emerge] checkpoint step {step}")

    # Wrap the jitted Adam step so we snapshot the live tensors mid-run.
    import pdft.training.batched as B
    _orig_build = B._build_jit_adam_step

    def _patched_build(*a, **k):
        step_fn = _orig_build(*a, **k)

        def wrapped(ct, ms, vs, stacked, lr_t, gstep):
            out = step_fn(ct, ms, vs, stacked, lr_t, gstep)
            s = int(gstep)
            if s in snap_steps:
                _dump(out[0], s)
            return out
        return wrapped

    B._build_jit_adam_step = _patched_build

    device = jax.devices()[0]
    with jax.default_device(device):
        dataset = [jax.device_put(np.asarray(img).astype(np.complex128), device)
                   for img in train_imgs]
        _dump(basis.tensors, 0)                          # step-0 random init
        result = pdft.train_basis_batched(
            basis,
            dataset=dataset,
            loss=pdft.MSELoss(k=max(1, round(2 ** (m + n) * 0.1))),
            epochs=args.epochs,
            batch_size=preset.batch_size,
            optimizer=preset.optimizer,
            validation_split=preset.validation_split,
            early_stopping_patience=10 ** 9,             # never early-stop
            warmup_frac=preset.warmup_frac,
            lr_peak=preset.lr_peak,
            lr_final=preset.lr_final,
            max_grad_norm=preset.max_grad_norm,
            shuffle=True,
            seed=preset.seed,
            val_every_k_epochs=preset.val_every_k_epochs,
        )

    B._build_jit_adam_step = _orig_build                 # restore
    (out / "loss_history.json").write_text(json.dumps({
        "seed": args.seed, "epochs": args.epochs, "total_steps": total_steps,
        "steps_per_epoch": n_batches, "every": args.every,
        "loss_history": [float(x) for x in result.loss_history],
    }))
    print(f"[emerge] done: {result.steps} steps, final loss "
          f"{result.loss_history[-1]:.6g}; checkpoints in {ckpt_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
