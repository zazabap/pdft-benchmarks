#!/usr/bin/env python3
"""Train one basis on DIV2K from its canonical init and save the loss history,
so the per-epoch validation-MSE loss curve (structure writeup Fig. 2) can be
re-rendered. Matches the headline objective: rho=0.10 (K=6554), 1008 steps
(epochs=112), seed 42, generalized preset.

Usage:
    python experiments/misc/train_basis_loss.py --gpu 7 --basis qft \
        --out results/structure/div2k_8q_pca_vs_block_dct/by_basis/qft/loss_history
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
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--basis", required=True,
                    help="qft | entangled_qft | mera | tebd | rich")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=112, help="112 -> 1008 steps.")
    ap.add_argument("--out", required=True, help="loss_history dir for this basis.")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import numpy as np
    import pdft

    from pdft_benchmarks.bases import BASIS_FACTORIES
    from pdft_benchmarks.datasets.div2k import load_div2k
    from pdft_benchmarks.presets import get_preset

    m = n = 8
    preset = get_preset("div2k_8q", "generalized")
    train_imgs, _ = load_div2k(n_train=preset.n_train, n_test=preset.n_test,
                               seed=preset.seed, size=2 ** m)
    basis = BASIS_FACTORIES[args.basis](m, n, seed=args.seed)   # canonical init

    device = jax.devices()[0]
    with jax.default_device(device):
        dataset = [jax.device_put(np.asarray(img).astype(np.complex128), device)
                   for img in train_imgs]
        result = pdft.train_basis_batched(
            basis,
            dataset=dataset,
            loss=pdft.MSELoss(k=max(1, round(2 ** (m + n) * 0.1))),   # rho=0.10
            epochs=args.epochs,
            batch_size=preset.batch_size,
            optimizer=preset.optimizer,
            validation_split=preset.validation_split,
            early_stopping_patience=10 ** 9,
            warmup_frac=preset.warmup_frac,
            lr_peak=preset.lr_peak,
            lr_final=preset.lr_final,
            max_grad_norm=preset.max_grad_norm,
            shuffle=True,
            seed=preset.seed,
            val_every_k_epochs=preset.val_every_k_epochs,
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.basis}_loss.json").write_text(json.dumps({
        "basis": args.basis, "seed": args.seed,
        "step_losses": [float(x) for x in result.loss_history],
        "val_losses": [float(x) for x in result.val_history],
        "epochs_completed": int(result.epochs_completed),
        "steps": int(result.steps),
    }))
    print(f"[loss] {args.basis}: {result.steps} steps, final val "
          f"{result.val_history[-1]:.6g}; wrote {out}/{args.basis}_loss.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
