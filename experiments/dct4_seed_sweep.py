#!/usr/bin/env python3
"""Random-seed variance sweep for the learnable DCT-IV basis (DIV2K-8q), trained
with NORMAL batched training (not the gate-unfreeze "sweep").

The DCT-IV analog of `experiments/qft_seed_sweep.py`. Each seed `s` reseeds
EVERYTHING trainable, exactly as the QFT seed study does — only the training
procedure differs (`pdft.train_basis_batched`, one cosine-LR top-k run, instead
of `train_progressive_unfreeze`):

  - the random init   : `dct4_random_basis("dct4", 8, 8, s)` — a fresh Haar
                        real-orthogonal init of the DCT-IV gate set (SO(2)/SO(4)
                        on the gates, random {0,pi} sign on the CP "Delta" gate),
                        NOT the canonical exact-DCT-IV init.
  - the training batch: 50 images subsampled from the fixed 500-image train pool
                        with `np.random.default_rng(s)`.
  - the training RNG  : `train_basis_batched(seed=s)`.

The held-out TEST set is held FIXED (canonical seed-42 50 images) so endpoint
PSNR variance reflects the trained model, not which test images were drawn —
directly comparable to the QFT seed writeup.

Each seed is an independent, atomically-checkpointed cell:

    <out>/_runs/seed_<NNN>.json        committed endpoint cell (+ loss trace)

A finished cell is skipped on re-invocation (unless --force), so the parallel
dispatcher (tools/run_dct4_seed_sweep.py) can fan jobs across idle GPUs, crash,
and resume by filling only the gaps. `--aggregate-only` rolls the cells up into
<out>/seed_sweep.json without any training.

This driver needs `pdft.DCT4Basis` (pdft >= 0.2.2). If the installed pdft is
stale, put the latest pdft `src` first on PYTHONPATH; `pdft_benchmarks` is
imported from THIS repo's `src` (added to sys.path below) regardless of any
editable-install location.

Usage:
    python experiments/dct4_seed_sweep.py --gpu 1 --seeds 7            # one cell
    python experiments/dct4_seed_sweep.py --gpu 1 --seeds 1-5          # a slice
    python experiments/dct4_seed_sweep.py --aggregate-only --seeds 1-100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Resolve `pdft_benchmarks` from THIS repo's src (the editable install may point
# at a different worktree that lacks the dct4 helper).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _parse_seeds(spec: str) -> list[int]:
    """Parse "1-100" / "1,2,5" / "1-10,42" into a sorted unique seed list."""
    seeds: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.update(range(int(lo), int(hi) + 1))
        else:
            seeds.add(int(part))
    return sorted(seeds)


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _cell_path(out_base: Path, seed: int) -> Path:
    return out_base / "_runs" / f"seed_{seed:03d}.json"


def _try_claim(cell: Path, stale_seconds: float = 14400) -> bool:
    """Atomically claim a seed so two dispatchers don't redundantly run it.
    Conservative: on any doubt returns True (a redundant run is harmless,
    atomic last-writer-wins; a wrongly-skipped seed would be missing)."""
    claim = cell.with_suffix(".claim")
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, str(int(time.time())).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            fresh = (time.time() - os.path.getmtime(claim)) < stale_seconds
        except OSError:
            return True
        if fresh:
            return False
        try:
            os.utime(claim, None)
        except OSError:
            pass
        return True
    except OSError:
        return True


def _release_claim(cell: Path) -> None:
    try:
        cell.with_suffix(".claim").unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Aggregation (no GPU).
# ---------------------------------------------------------------------------
def aggregate(out_base: Path, seeds: list[int], keep_ratios: tuple[float, ...],
              dataset: str, topk_ratio: float, epochs: int, data_seed: int,
              method: str = "dct4_random_normal") -> dict:
    import numpy as np

    def _stats(vals: list[float]) -> dict:
        a = np.asarray(vals, dtype=np.float64)
        out = {"mean": float(a.mean()),
               "std": float(a.std(ddof=1) if a.size > 1 else 0.0),
               "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}
        out["shapiro_p"] = None
        if a.size >= 3 and a.std() > 0:
            try:
                from scipy.stats import shapiro
                out["shapiro_p"] = float(shapiro(a).pvalue)
            except Exception:  # noqa: BLE001
                pass
        return out

    per_seed: dict[str, dict] = {}
    init_losses: dict[str, float] = {}
    missing: list[int] = []
    for s in seeds:
        cell = _cell_path(out_base, s)
        if not cell.exists():
            missing.append(s)
            continue
        data = json.loads(cell.read_text())
        per_seed[str(s)] = data["psnr"]
        if data.get("init_loss") is not None:
            init_losses[str(s)] = data["init_loss"]

    agg = {}
    for kr in keep_ratios:
        key = str(kr)
        vals = [v[key] for v in per_seed.values() if key in v]
        if vals:
            agg[key] = _stats(vals)

    summary = {
        "dataset": dataset, "method": method, "init": "haar_so",
        "topk_ratio": topk_ratio, "epochs": epochs,
        "data_seed_fixed_test": data_seed, "keep_ratios": list(keep_ratios),
        "seeds": seeds, "n_seeds": len(seeds),
        "per_seed": per_seed, "agg": agg,
        "init_loss_per_seed": init_losses,
        "missing": missing,
    }
    _atomic_write_json(out_base / "seed_sweep.json", summary)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU index. Sets CUDA_VISIBLE_DEVICES before any pdft/jax import.")
    p.add_argument("--dataset", default="div2k_8q",
                   choices=["div2k_8q", "quickdraw_5q", "tuberlin_8q"])
    p.add_argument("--seeds", default="1-100",
                   help='Seed spec: "1-100" range, "1,2,5" list, or a mix.')
    p.add_argument("--epochs", type=int, default=1008,
                   help="Fixed gradient-step budget (batch==len(subsample) -> "
                        "1 step/epoch, so epochs == steps). Headline: 1008.")
    p.add_argument("--topk-ratio", type=float, default=0.20,
                   help="Training objective: keep top fraction of coefficients "
                        "in the MSE loss. Default 0.20 (matches the QFT seed study).")
    p.add_argument("--batch", type=int, default=None,
                   help="Training subsample size (how many of the 500-image pool "
                        "to train on). Default: full train set at m=5, else 50.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="SGD mini-batch size, decoupled from --batch. Default: "
                        "equal to --batch (full-batch on the subsample, the "
                        "original behavior). Set e.g. --batch 500 --batch-size 50 "
                        "to train on the full pool with mini-batches.")
    p.add_argument("--lr", type=float, default=None, help="Default: preset.lr_peak.")
    p.add_argument("--out", default=None,
                   help="Output base. Default results/training/2_direct_training/"
                        "random_seed/dct_<dataset>.")
    p.add_argument("--no-trace", action="store_true", default=False,
                   help="Skip the per-step loss trace (keep only the endpoint cell).")
    p.add_argument("--no-tensors", action="store_true", default=False,
                   help="Skip saving the trained DCT-IV operator. By DEFAULT the "
                        "trained basis is saved per seed to trained_seed_<NNN>.json "
                        "(serialize_tensors), matching the QFT seed study so the "
                        "operator is recoverable for every finished training.")
    p.add_argument("--parametrization", default="o4", choices=["o4", "controlled"],
                   help="DCT-IV twiddle parametrization. 'o4' (default): dense "
                        "O(4) twiddle + trainable mirror CNOTs. 'controlled': "
                        "single-angle O(2) CRY twiddle + fixed CX-flip mirror "
                        "(7x fewer params, ~2x faster step). Controlled cells "
                        "default to the dct_<dataset>_controlled output dir.")
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--aggregate-only", action="store_true", default=False)
    args = p.parse_args()
    _controlled = args.parametrization == "controlled"
    _method = "dct4ctl_random_normal" if _controlled else "dct4_random_normal"

    DATASET_CFG = {
        "quickdraw_5q": (5, "load_quickdraw", "img_size"),
        "div2k_8q": (8, "load_div2k", "size"),
        "tuberlin_8q": (8, "load_tuberlin", "size"),
    }
    m_q, loader_name, size_kw = DATASET_CFG[args.dataset]
    m = n = m_q
    seeds = _parse_seeds(args.seeds)
    keep_ratios = (0.01, 0.05, 0.10, 0.20)
    _dataset_tag = f"{args.dataset}_controlled" if _controlled else args.dataset
    out_base = Path(args.out) if args.out else \
        Path(f"results/training/2_direct_training/random_seed/dct_{_dataset_tag}")
    out_base.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        s = aggregate(out_base, seeds, keep_ratios, args.dataset,
                      args.topk_ratio, args.epochs, data_seed=42, method=_method)
        done = len(s["per_seed"])
        print(f"[dct4_sweep] aggregated {done} cells -> {out_base/'seed_sweep.json'}")
        a = s["agg"].get("0.2")
        if a:
            print(f"  PSNR@.20 mean={a['mean']:.3f} std={a['std']:.3f} "
                  f"min={a['min']:.3f} max={a['max']:.3f} n={a['n']} "
                  f"shapiro_p={a['shapiro_p']}")
        return 0

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # IMPORTANT: imports AFTER the env var so JAX binds the requested device.
    import jax
    import numpy as np
    import pdft
    import pdft.io  # noqa: F401 — needed by evaluate_basis_shared
    from pdft_benchmarks import datasets as ds_mod
    from pdft_benchmarks.bases import dct4_random_basis, dct4_random_controlled_basis
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.experiment_utils import git_sha, serialize_tensors
    from pdft_benchmarks.presets import get_preset

    if not hasattr(pdft, "DCT4Basis"):
        print(f"[dct4_sweep] FATAL: pdft at {pdft.__file__} lacks DCT4Basis "
              f"(need >= 0.2.2). Put the latest pdft src first on PYTHONPATH.",
              file=sys.stderr)
        return 3

    chosen = jax.devices()[0]
    print(f"[dct4_sweep] device: {chosen} (platform={chosen.platform!r}) "
          f"pdft={pdft.__version__} @ {Path(pdft.__file__).parent}")
    if args.gpu is not None and chosen.platform not in ("gpu", "cuda"):
        print(f"[dct4_sweep] FATAL: --gpu {args.gpu} requested but JAX sees "
              f"platform={chosen.platform!r} (NVML init failure?). Aborting to "
              f"avoid a silent CPU run.", file=sys.stderr)
        return 2

    preset = get_preset(args.dataset, "generalized")
    lr = args.lr if args.lr is not None else preset.lr_peak
    batch = args.batch if args.batch is not None else (preset.n_train if m == 5 else 50)
    batch_size = args.batch_size if args.batch_size is not None else batch
    loader = getattr(ds_mod, loader_name)

    # Canonical split (seed 42): fixes BOTH the 500-image train pool and the
    # 50-image held-out test set. Per-seed runs subsample the training batch
    # from `train_pool` but always evaluate on this same `test_imgs`.
    train_pool, test_imgs = loader(n_train=preset.n_train, n_test=preset.n_test,
                                   seed=42, **{size_kw: 2 ** m})
    k_train = max(1, round(2 ** (m + n) * args.topk_ratio))
    loss = pdft.MSELoss(k=k_train)
    print(f"[dct4_sweep] dataset={args.dataset} m=n={m} topk_ratio={args.topk_ratio} "
          f"k_train={k_train} batch={batch} lr={lr} epochs={args.epochs} "
          f"seeds=[{seeds[0]}..{seeds[-1]}] n={len(seeds)}")

    cell_dir = out_base / "_runs"
    cell_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        cell = _cell_path(out_base, seed)
        if cell.exists() and not args.force:
            print(f"[dct4_sweep] skip seed_{seed:03d} (exists)")
            continue
        if not args.force and not _try_claim(cell):
            print(f"[dct4_sweep] skip seed_{seed:03d} (claimed by another worker)")
            continue
        if cell.exists() and not args.force:
            _release_claim(cell)
            continue

        # Per-seed: Haar real-orthogonal init + 50-image training subsample.
        basis = (dct4_random_controlled_basis if _controlled
                 else dct4_random_basis)(m, n, seed)
        rng = np.random.default_rng(seed)
        batch_idx = sorted(int(i) for i in
                           rng.choice(len(train_pool), size=batch, replace=False))
        dataset = [jax.device_put(np.asarray(train_pool[i]).astype(np.complex128), chosen)
                   for i in batch_idx]

        # --epochs is a STEP budget (the original 1-step/epoch convention). With
        # a decoupled mini-batch there are ceil(batch/batch_size) steps/epoch, so
        # convert the step budget to whole epochs to keep total steps ~ --epochs.
        steps_per_epoch = max(1, -(-batch // batch_size))
        epochs_to_run = max(1, -(-args.epochs // steps_per_epoch))
        t0 = time.perf_counter()
        res = pdft.train_basis_batched(
            basis, dataset=dataset, loss=loss,
            epochs=epochs_to_run, batch_size=batch_size, optimizer=preset.optimizer,
            validation_split=0.0, early_stopping_patience=10 ** 9,
            warmup_frac=preset.warmup_frac, lr_peak=lr, lr_final=preset.lr_final,
            max_grad_norm=preset.max_grad_norm, shuffle=True, seed=seed,
            val_every_k_epochs=preset.val_every_k_epochs)
        elapsed = time.perf_counter() - t0

        metrics, _ = evaluate_basis_shared(res.basis, test_imgs, keep_ratios=keep_ratios)
        psnr = {f"{r}": float(metrics[str(r)]["mean_psnr"]) for r in keep_ratios}
        init_loss = float(res.loss_history[0]) if res.loss_history else None
        final_loss = float(res.loss_history[-1]) if res.loss_history else None
        print(f"[dct4_sweep] seed_{seed:03d}: {res.steps} steps, {elapsed:.1f}s, "
              f"loss {init_loss:.4e} -> {final_loss:.4e}, "
              f"PSNR@.20={psnr['0.2']:.3f} dB")

        # Save the trained operator + trace BEFORE the endpoint cell: the cell
        # is the resume/skip marker, so writing it last guarantees that any
        # seed counted as "done" already has its basis on disk (the trained
        # basis is saved for every finished training).
        if not args.no_tensors:
            _atomic_write_json(cell.with_name(f"trained_seed_{seed:03d}.json"), {
                "seed": seed, "m": int(res.basis.m), "n": int(res.basis.n),
                "tensors": serialize_tensors(res.basis.tensors),
            })
        if not args.no_trace:
            _atomic_write_json(cell.with_name(f"seed_{seed:03d}_trace.json"), {
                "seed": seed, "loss_history": [float(x) for x in res.loss_history],
            })
        _atomic_write_json(cell, {
            "dataset": args.dataset, "seed": seed, "m": m, "n": n,
            "init": "haar_so", "method": _method,
            "topk_ratio": args.topk_ratio, "k_train": k_train,
            "lr": lr, "batch": batch, "batch_size": batch_size, "epochs": args.epochs,
            "train_batch_idx": batch_idx,
            "psnr": psnr, "init_loss": init_loss, "final_loss": final_loss,
            "total_steps": int(res.steps), "epochs_completed": int(res.epochs_completed),
            "elapsed_seconds": elapsed,
            "device": str(chosen), "git_sha": git_sha(short=False),
        })
        _release_claim(cell)

    aggregate(out_base, seeds, keep_ratios, args.dataset, args.topk_ratio,
              args.epochs, data_seed=42, method=_method)
    print(f"[dct4_sweep] done. Summary: {out_base/'seed_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
