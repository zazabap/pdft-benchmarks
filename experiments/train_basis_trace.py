#!/usr/bin/env python3
"""Train one DCT-IV variant on DIV2K-8q and save its full loss trajectory
(per-step training loss + per-epoch validation loss) for a training-dynamics plot.

  python experiments/train_basis_trace.py --basis dct4_controlled --gpu <g> --steps 1008
  python experiments/train_basis_trace.py --basis dct4_o4         --gpu <g> --steps 1008

The loss trajectory is card-independent, so the two variants can run in parallel
on any two GPUs. Writes results/dct4_controlled/trace_<basis>.json.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", choices=["dct4_o4", "dct4_controlled"], required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--steps", type=int, default=1008)
    ap.add_argument("--out", default="results/dct4_controlled")
    args = ap.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax, jax.numpy as jnp, numpy as np, pdft
    import pdft.io  # noqa: F401
    from pdft_benchmarks.bases import dct4_controlled_basis
    from pdft_benchmarks.datasets.div2k import load_div2k
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.presets import get_preset

    dev = jax.devices()[0]
    if dev.platform not in ("gpu", "cuda"):
        print("FATAL: not on GPU", file=sys.stderr); return 2
    m = n = 8
    preset = get_preset("div2k_8q", "generalized")
    k = max(1, round(2 ** (m + n) * 0.10))
    loss = pdft.MSELoss(k=k)
    train, test = load_div2k(n_train=preset.n_train, n_test=preset.n_test, seed=42, size=2 ** m)
    ds = [jax.device_put(np.asarray(x).astype(np.complex128), dev) for x in train]

    n_val = int(np.clip(round(len(ds) * preset.validation_split), 0, len(ds) - 1))
    n_batches = max(1, -(-(len(ds) - n_val) // preset.batch_size))
    epochs = max(1, round(args.steps / n_batches))

    if args.basis == "dct4_controlled":
        basis, frozen = dct4_controlled_basis(m, n)
    else:
        basis, frozen = pdft.DCT4Basis(m=m, n=n), []

    t0 = time.perf_counter()
    res = pdft.train_basis_batched(
        basis, dataset=ds, loss=loss, epochs=epochs, batch_size=preset.batch_size,
        optimizer=preset.optimizer, validation_split=preset.validation_split,
        early_stopping_patience=10 ** 9, warmup_frac=preset.warmup_frac,
        lr_peak=preset.lr_peak, lr_final=preset.lr_final, max_grad_norm=preset.max_grad_norm,
        shuffle=True, seed=42, val_every_k_epochs=preset.val_every_k_epochs,
        frozen_indices=(frozen or None))
    elapsed = time.perf_counter() - t0

    metrics, _ = evaluate_basis_shared(res.basis, test, keep_ratios=(0.05, 0.10, 0.15, 0.20))
    psnr = {f"{r}": float(metrics[str(r)]["mean_psnr"]) for r in (0.05, 0.10, 0.15, 0.20)}
    fs = set(frozen)
    n_trainable = sum(int(jnp.asarray(t).size) for i, t in enumerate(basis.tensors) if i not in fs)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"trace_{args.basis}.json").write_text(json.dumps({
        "basis": args.basis, "device": str(dev), "gpu_name": os.popen(
            "nvidia-smi --query-gpu=name --format=csv,noheader -i "
            + os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] + " 2>/dev/null").read().strip(),
        "steps": int(res.steps), "epochs": epochs, "elapsed_s": elapsed,
        "s_per_step": elapsed / max(1, res.steps), "n_trainable_params": n_trainable,
        "final_loss": float(res.loss_history[-1]), "psnr": psnr,
        "step_losses": [float(x) for x in res.loss_history],
        "val_losses": [float(x) for x in res.val_history],
    }))
    print(f"[{args.basis}] {res.steps} steps {elapsed:.1f}s ({elapsed/res.steps:.3f}s/step) "
          f"loss {res.loss_history[0]:.3f} -> {res.loss_history[-1]:.3f} "
          f"PSNR@.10={psnr['0.1']:.3f} params={n_trainable}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
