#!/usr/bin/env python3
"""Compare structured-O(2) DCT-IV (frozen mirror) vs dense-O(4) DCT-IV on
DIV2K-8q: per-step time, trainable params, and reconstruction PSNR.

Run:
  CUDA_VISIBLE_DEVICES=<gpu> CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=/workspaces/pdft-benchmarks-dct4exp/src:/workspaces/parametric-dft-paper/pdft-dct4main/src \
    /workspaces/pdft-benchmarks/.venv/bin/python experiments/dct4/dct4_controlled_compare.py --gpu <gpu> --steps 200
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # repo root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--steps", type=int, default=200, help="short training budget")
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
    keep_ratios = (0.05, 0.10, 0.15, 0.20)
    train, test = load_div2k(n_train=preset.n_train, n_test=preset.n_test, seed=42, size=2 ** m)
    ds = [jax.device_put(np.asarray(x).astype(np.complex128), dev) for x in train]

    # train_basis_batched counts EPOCHS, not optimizer steps. Convert the
    # requested step budget to whole epochs given the per-epoch batch count
    # (n_train after the validation split, ceil-divided by batch_size).
    n_val = int(np.clip(round(len(ds) * preset.validation_split), 0, len(ds) - 1))
    n_batches = max(1, -(-(len(ds) - n_val) // preset.batch_size))
    epochs = max(1, round(args.steps / n_batches))

    def n_trainable(basis, frozen):
        fs = set(frozen)
        return sum(int(jnp.asarray(t).size) for i, t in enumerate(basis.tensors) if i not in fs)

    def run(tag, basis, frozen):
        t0 = time.perf_counter()
        res = pdft.train_basis_batched(
            basis, dataset=ds, loss=loss, epochs=epochs, batch_size=preset.batch_size,
            optimizer=preset.optimizer, validation_split=preset.validation_split,
            early_stopping_patience=10 ** 9, warmup_frac=preset.warmup_frac,
            lr_peak=preset.lr_peak, lr_final=preset.lr_final, max_grad_norm=preset.max_grad_norm,
            shuffle=True, seed=42, val_every_k_epochs=preset.val_every_k_epochs,
            frozen_indices=(frozen or None))
        elapsed = time.perf_counter() - t0
        metrics, _ = evaluate_basis_shared(res.basis, test, keep_ratios=keep_ratios)
        psnr = {f"{r}": float(metrics[str(r)]["mean_psnr"]) for r in keep_ratios}
        return {"tag": tag, "steps": int(res.steps), "elapsed_s": elapsed,
                "s_per_step": elapsed / max(1, res.steps),
                "n_trainable_params": n_trainable(basis, frozen or []),
                "final_loss": float(res.loss_history[-1]), "psnr": psnr}

    o4 = run("dct4_o4", pdft.DCT4Basis(m=m, n=n), [])
    cb, frozen = dct4_controlled_basis(m, n)
    ctl = run("dct4_controlled", cb, frozen)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = {"device": str(dev), "target_steps": args.steps, "epochs": epochs, "k": k,
               "o4": o4, "controlled": ctl,
               "speedup_per_step": o4["s_per_step"] / ctl["s_per_step"],
               "param_ratio": o4["n_trainable_params"] / max(1, ctl["n_trainable_params"])}
    (out / "compare.json").write_text(json.dumps(summary, indent=2))
    for r in (o4, ctl):
        print(f"{r['tag']:16s} {r['s_per_step']*1000:7.1f} ms/step  "
              f"params={r['n_trainable_params']:6d}  PSNR@.10={r['psnr']['0.1']:.3f} dB")
    print(f"per-step speedup {summary['speedup_per_step']:.2f}x  "
          f"param ratio {summary['param_ratio']:.1f}x  -> {out/'compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
