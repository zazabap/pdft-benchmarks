#!/usr/bin/env python3
"""Random-seed variance sweep for the progressive gate-unfreeze QFT (DIV2K-8q).

Trains the QFT(m, n) gate-unfreeze operator from Haar-random init, top-20%
coefficient objective, over many seeds and the three unfreeze orderings
(`bg`/`lr`/`rl`). Each seed `s` reseeds EVERYTHING trainable:

  - the Haar gate init               (family_random_basis("qft", m, n, s))
  - the 50-image training batch      (subsampled from the fixed 500-image train
                                      pool with np.random.default_rng(s))
  - the training RNG                 (train_progressive_unfreeze seed=s)

The held-out TEST set is held FIXED (canonical seed-42 50 images) so endpoint
PSNR variance reflects the trained model, not which test images were drawn,
and stays directly comparable to the unfreeze writeup.

Each (ordering, seed) run is an independent, atomically-checkpointed cell:

    <out>/_runs/<ordering>/seed_<NNN>.json        committed endpoint cell
    <out>/_runs/<ordering>/seed_<NNN>_trace.json  full per-step loss trace
                                                  (local; default on; --no-trace)

A finished cell is skipped on re-invocation (unless --force), so the parallel
dispatcher (tools/run_seed_sweep.py) can fan 300 jobs across GPUs, crash, and
resume by filling only the gaps. `--aggregate-only` rolls the cells up into
<out>/seed_sweep.json without any training (mean / std / min / max / Shapiro-p
per ordering per keep ratio).

Usage:
    # one cell (what the dispatcher calls)
    python experiments/qft/qft_seed_sweep.py --gpu 0 --orderings bg --seeds 7
    # a slice, sequentially, on one GPU
    python experiments/qft/qft_seed_sweep.py --gpu 0 --orderings bg,lr,rl --seeds 1-5
    # roll up whatever cells exist (no GPU needed)
    python experiments/qft/qft_seed_sweep.py --aggregate-only --seeds 1-100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


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
    """Write JSON to `path` atomically (tmp + os.replace) so a crash never
    leaves a half-written checkpoint that the resume path would trust."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _cell_path(out_base: Path, ordering: str, seed: int) -> Path:
    return out_base / "_runs" / ordering / f"seed_{seed:03d}.json"


def _try_claim(cell: Path, stale_seconds: float = 14400) -> bool:
    """Atomically claim a (ordering, seed) so a *second* dispatcher's driver
    won't redundantly run the same seed where the forward/reverse sweeps cross
    (mainly on `lr`). Returns True if we may run it, False only if another LIVE
    driver currently holds the claim.

    Conservative by design: on ANY doubt (claim is stale from a crashed run,
    unreadable, or a race) it returns True. A redundant run is harmless (atomic
    last-writer-wins), whereas a wrongly-skipped seed would be missing from the
    300 — so we always err toward running.
    """
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
            return False            # a live driver owns it -> skip
        try:
            os.utime(claim, None)   # stale (crashed run) -> take it over
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
# Aggregation (no GPU; reusable from --aggregate-only and from the dispatcher).
# ---------------------------------------------------------------------------
def aggregate(out_base: Path, orderings: list[str], seeds: list[int],
              keep_ratios: tuple[float, ...], dataset: str,
              topk_ratio: float, data_seed: int) -> dict:
    """Roll up every existing seed cell into a single summary dict and write it
    to <out_base>/seed_sweep.json. Missing cells are skipped (and counted)."""
    import numpy as np

    def _stats(vals: list[float]) -> dict:
        a = np.asarray(vals, dtype=np.float64)
        out = {"mean": float(a.mean()), "std": float(a.std(ddof=1) if a.size > 1 else 0.0),
               "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}
        # Shapiro-Wilk normality (needs >=3 points and some spread).
        out["shapiro_p"] = None
        if a.size >= 3 and a.std() > 0:
            try:
                from scipy.stats import shapiro
                out["shapiro_p"] = float(shapiro(a).pvalue)
            except Exception:  # noqa: BLE001 — scipy optional / degenerate input
                pass
        return out

    per_ordering: dict[str, dict] = {}
    missing: dict[str, list[int]] = {}
    for ordering in orderings:
        per_seed: dict[str, dict] = {}
        miss: list[int] = []
        for s in seeds:
            cell = _cell_path(out_base, ordering, s)
            if not cell.exists():
                miss.append(s)
                continue
            data = json.loads(cell.read_text())
            per_seed[str(s)] = data["psnr"]
        agg = {}
        for kr in keep_ratios:
            key = str(kr)
            vals = [v[key] for v in per_seed.values() if key in v]
            if vals:
                agg[key] = _stats(vals)
        per_ordering[ordering] = {"per_seed": per_seed, "agg": agg}
        if miss:
            missing[ordering] = miss

    summary = {
        "dataset": dataset, "init": "random", "topk_ratio": topk_ratio,
        "data_seed_fixed_test": data_seed, "keep_ratios": list(keep_ratios),
        "seeds": seeds, "n_seeds": len(seeds),
        "per_ordering": per_ordering,
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
    p.add_argument("--orderings", default="bg,lr,rl")
    p.add_argument("--seeds", default="1-100",
                   help='Seed spec: "1-100" range, "1,2,5" list, or a mix.')
    p.add_argument("--topk-ratio", type=float, default=0.20,
                   help="Training objective: keep top fraction of coefficients "
                        "in the MSE loss. Default 0.20 (the headline is 0.10).")
    p.add_argument("--batch", type=int, default=None,
                   help="Training batch size (subsampled per seed). Default: full "
                        "train set at m=5, else 50.")
    p.add_argument("--lr", type=float, default=None, help="Default: preset.lr_peak.")
    p.add_argument("--grad-tol", type=float, default=1e-5)
    p.add_argument("--loss-tol", type=float, default=1e-5)
    p.add_argument("--min-steps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--grad-check-every", type=int, default=5,
                   help="Riemannian grad-norm probe cadence (loss-delta plateau "
                        "uses the free per-step loss). 5 ~halves per-step cost.")
    p.add_argument("--out", default=None,
                   help="Output base. Default results/training/2_direct_training/"
                        "random_seed/<dataset>.")
    p.add_argument("--no-trace", action="store_true", default=False,
                   help="Skip the full per-step trace file (keeps only the "
                        "compact endpoint cell with per-stage final losses).")
    p.add_argument("--no-tensors", action="store_true", default=False,
                   help="Skip saving the trained QFT operator (trained_seed_*.json). "
                        "Each is ~10-20 KB and regenerable from the seed.")
    p.add_argument("--force", action="store_true", default=False,
                   help="Retrain cells even if their seed_<NNN>.json exists.")
    p.add_argument("--aggregate-only", action="store_true", default=False,
                   help="Skip training; just roll existing cells into "
                        "seed_sweep.json. No GPU needed.")
    args = p.parse_args()

    DATASET_CFG = {
        "quickdraw_5q": (5, "load_quickdraw", "img_size"),
        "div2k_8q": (8, "load_div2k", "size"),
        "tuberlin_8q": (8, "load_tuberlin", "size"),
    }
    m_q, loader_name, size_kw = DATASET_CFG[args.dataset]
    m = n = m_q
    orderings = [s.strip() for s in args.orderings.split(",") if s.strip()]
    seeds = _parse_seeds(args.seeds)
    keep_ratios = (0.05, 0.10, 0.15, 0.20)
    out_base = Path(args.out) if args.out else \
        Path(f"results/training/2_direct_training/random_seed/{args.dataset}")
    out_base.mkdir(parents=True, exist_ok=True)

    # --- aggregate-only short-circuit: no JAX, no GPU. -----------------------
    if args.aggregate_only:
        s = aggregate(out_base, orderings, seeds, keep_ratios,
                      args.dataset, args.topk_ratio, data_seed=42)
        done = sum(len(v["per_seed"]) for v in s["per_ordering"].values())
        print(f"[seed_sweep] aggregated {done} cells -> {out_base/'seed_sweep.json'}")
        for o in orderings:
            a = s["per_ordering"][o]["agg"].get("0.2")
            if a:
                print(f"  {o}: PSNR@.20 mean={a['mean']:.3f} std={a['std']:.3f} "
                      f"min={a['min']:.3f} max={a['max']:.3f} n={a['n']} "
                      f"shapiro_p={a['shapiro_p']}")
        return 0

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    # IMPORTANT: imports AFTER the env var so JAX binds the requested device.
    import jax
    import numpy as np
    import pdft
    import pdft.io  # noqa: F401 — needed by evaluate_basis_shared
    from pdft_benchmarks import datasets as ds_mod
    from pdft_benchmarks.bases import family_random_basis
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.experiment_utils import git_sha, serialize_tensors
    from pdft_benchmarks.presets import get_preset
    from pdft_benchmarks.unfreeze import qft_unfreeze_orders, train_progressive_unfreeze

    chosen = jax.devices()[0]
    print(f"[seed_sweep] device: {chosen} (platform={chosen.platform!r})")
    if args.gpu is not None and chosen.platform not in ("gpu", "cuda"):
        print(f"[seed_sweep] FATAL: --gpu {args.gpu} requested but JAX sees "
              f"platform={chosen.platform!r} (NVML init failure?). Aborting to "
              f"avoid a silent CPU run.", file=sys.stderr)
        return 2

    preset = get_preset(args.dataset, "generalized")
    lr = args.lr if args.lr is not None else preset.lr_peak
    batch = args.batch if args.batch is not None else (preset.n_train if m == 5 else 50)
    loader = getattr(ds_mod, loader_name)

    # Canonical split (seed 42): fixes BOTH the 500-image train pool and the
    # 50-image held-out test set. Per-seed runs subsample the training batch
    # from `train_pool` but always evaluate on this same `test_imgs`.
    train_pool, test_imgs = loader(n_train=preset.n_train, n_test=preset.n_test,
                                   seed=42, **{size_kw: 2 ** m})
    k_train = max(1, round(2 ** (m + n) * args.topk_ratio))
    loss = pdft.MSELoss(k=k_train)
    orders = qft_unfreeze_orders(m, n)
    print(f"[seed_sweep] dataset={args.dataset} m=n={m} topk_ratio={args.topk_ratio} "
          f"k_train={k_train} batch={batch} lr={lr} "
          f"orderings={orderings} seeds=[{seeds[0]}..{seeds[-1]}] n={len(seeds)}")

    for ordering in orderings:
        order = orders[ordering]
        n_stages = len(order)
        cell_dir = out_base / "_runs" / ordering
        cell_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            cell = _cell_path(out_base, ordering, seed)
            if cell.exists() and not args.force:
                print(f"[seed_sweep] skip {ordering}/seed_{seed:03d} (exists)")
                continue
            if not args.force and not _try_claim(cell):
                print(f"[seed_sweep] skip {ordering}/seed_{seed:03d} (claimed by another worker)")
                continue
            if cell.exists() and not args.force:  # cell landed while we claimed
                _release_claim(cell)
                continue

            # Per-seed training batch: subsample `batch` of the fixed train pool.
            rng = np.random.default_rng(seed)
            batch_idx = sorted(int(i) for i in
                               rng.choice(len(train_pool), size=batch, replace=False))
            fixed_batch = [np.asarray(train_pool[i]) for i in batch_idx]
            basis = family_random_basis("qft", m, n, seed)

            def stage_psnr(stage, tensors, _n=n_stages):
                if stage != _n:  # endpoint only — skip ~71 full-test evals/run
                    return {}
                b = pdft.QFTBasis(m=m, n=n, tensors=tensors)
                metrics, _ = evaluate_basis_shared(b, test_imgs, keep_ratios=keep_ratios)
                return {"psnr": {f"{r}": float(metrics[str(r)]["mean_psnr"])
                                 for r in keep_ratios}}

            t0 = time.perf_counter()
            res = train_progressive_unfreeze(
                basis, fixed_batch, unfreeze_order=order, lr=lr,
                max_steps_per_stage=args.max_steps, loss=loss,
                grad_tol=args.grad_tol, loss_tol=args.loss_tol,
                min_steps_per_stage=args.min_steps, seed=seed,
                stage_callback=stage_psnr, grad_check_every=args.grad_check_every)
            elapsed = time.perf_counter() - t0
            psnr = res.stages[-1].extra["psnr"]
            total_steps = res.stages[-1].end_step
            print(f"[seed_sweep] {ordering}/seed_{seed:03d}: {total_steps} steps, "
                  f"{elapsed:.1f}s, PSNR@.20={psnr['0.2']:.3f} dB")

            _atomic_write_json(cell, {
                "dataset": args.dataset, "ordering": ordering, "seed": seed,
                "m": m, "n": n, "init": "random",
                "topk_ratio": args.topk_ratio, "k_train": k_train,
                "lr": lr, "batch": batch, "train_batch_idx": batch_idx,
                "max_steps": args.max_steps, "grad_check_every": args.grad_check_every,
                "psnr": psnr,
                "final_loss": res.stages[-1].final_loss,
                "total_steps": total_steps,
                "per_stage_final_loss": [s.final_loss for s in res.stages],
                "trigger_counts": {tr: sum(1 for s in res.stages if s.trigger == tr)
                                   for tr in ("grad_norm", "loss_delta", "max_steps")},
                "elapsed_seconds": elapsed,
                "device": str(chosen), "git_sha": git_sha(short=False),
            })
            if not args.no_trace:
                _atomic_write_json(cell.with_name(f"seed_{seed:03d}_trace.json"), {
                    "ordering": ordering, "seed": seed,
                    "steps": res.trace,
                })
            if not args.no_tensors:
                # The trained QFT operator itself (~10-20 KB). Regenerable from
                # the seed, but saved so reconstruction / re-eval don't need a
                # 2 h rerun. Schema matches the unfreeze trained_*.json cells.
                _atomic_write_json(cell.with_name(f"trained_seed_{seed:03d}.json"), {
                    "ordering": ordering, "seed": seed,
                    "m": int(res.basis.m), "n": int(res.basis.n),
                    "tensors": serialize_tensors(res.basis.tensors),
                })
            _release_claim(cell)

    # Refresh the rolled-up summary so a single-process run also produces it.
    aggregate(out_base, orderings, seeds, keep_ratios,
              args.dataset, args.topk_ratio, data_seed=42)
    print(f"[seed_sweep] done. Summary: {out_base/'seed_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
