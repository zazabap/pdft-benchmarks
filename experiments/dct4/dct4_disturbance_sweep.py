#!/usr/bin/env python3
"""Exact-init disturbance sweep for the controlled O(2)-twiddle DCT-IV (DIV2K-8q).

Start from the *exact* analytic DCT-IV init, jitter a random fraction f of its
2200 real gate entries on-manifold (Gaussian, sigma), train, and record final
PSNR at rho in {0.01,0.05,0.10,0.20}. Each (f, seed) is an atomic, resumable
cell (the same layout the seed-sweep drivers on `dev` use):

    <out>/_runs/f<frac>_seed<NNN>.json

f=0 is the undisturbed exact-init reference (deterministic; recorded once). The
held-out TEST set is the canonical seed-42 50 images (fixed). Training is on the
FULL 500-image pool with mini-batch 50 and a FIXED shuffle seed, so the only
randomness at a given f is the perturbation draw.

Needs pdft.DCT4Basis pinned at commit 5365a5a (sliced CRY apply). Put that pdft
src first on PYTHONPATH; pdft_benchmarks is imported from THIS repo's src.

Usage:
    python experiments/dct4/dct4_disturbance_sweep.py --gpu 2 --fractions 0.01 --seeds 1
    python experiments/dct4/dct4_disturbance_sweep.py --gpu 2 --seeds 1-3        # full grid
    python experiments/dct4/dct4_disturbance_sweep.py --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # repo root

FRACTIONS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)
KEEP_RATIOS = (0.01, 0.05, 0.10, 0.20)
RHO_KEYS = ("0.01", "0.05", "0.1", "0.2")  # evaluate_basis_shared uses str(kr)


def _fkey(f: float) -> str:
    return f"{f:g}"


def _parse_seeds(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _parse_fractions(specs: list[str] | None) -> tuple[float, ...]:
    if not specs:
        return FRACTIONS
    return tuple(float(x) for x in specs)


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _cell_path(out_base: Path, f: float, seed: int) -> Path:
    return out_base / "_runs" / f"f{_fkey(f)}_seed{seed:03d}.json"


def _try_claim(cell: Path, stale_seconds: float = 14400) -> bool:
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


def _seed_for(f: float, seed: int):
    import numpy as np
    return np.random.default_rng([int(round(f * 1_000_000)), int(seed)])


def _stats(vals):
    import numpy as np
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1) if a.size > 1 else 0.0),
            "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def aggregate(out_base: Path, seeds, sigma, topk_ratio, epochs) -> dict:
    per_cell: dict[str, dict] = {}
    baseline = None
    found: set[float] = set()
    for cell in sorted((out_base / "_runs").glob("f*_seed*.json")):
        data = json.loads(cell.read_text())
        per_cell[cell.stem] = data
        fv = data.get("f", None)
        if fv == 0.0:
            baseline = data
        elif fv is not None:
            found.add(fv)
    fractions = sorted(found)

    def agg_over(kind: str) -> dict:
        out: dict[str, dict] = {}
        for f in fractions:
            fk = _fkey(f)
            out[fk] = {}
            for rk in RHO_KEYS:
                vals = [c[kind][rk] for c in per_cell.values()
                        if c.get("f") == f and kind in c and rk in c[kind]]
                if vals:
                    out[fk][rk] = _stats(vals)
        return out

    def agg_scalar(field: str) -> dict:
        """Aggregate a per-cell scalar (e.g. init_loss/final_loss) over seeds."""
        out: dict[str, dict] = {}
        for f in fractions:
            vals = [c[field] for c in per_cell.values()
                    if c.get("f") == f and c.get(field) is not None]
            if vals:
                out[_fkey(f)] = _stats(vals)
        return out

    summary = {
        "dataset": "div2k_8q", "parametrization": "controlled", "sigma": sigma,
        "topk_ratio": topk_ratio, "epochs": epochs,
        "fractions": fractions, "seeds": seeds,
        "keep_ratios": list(KEEP_RATIOS), "rho_keys": list(RHO_KEYS),
        "data_seed_fixed_test": 42, "train_shuffle_seed": 42,
        "baseline": baseline,
        "agg_trained": agg_over("psnr_trained"),
        "agg_untrained": agg_over("psnr_untrained"),
        "agg_init_loss": agg_scalar("init_loss"),
        "agg_final_loss": agg_scalar("final_loss"),
        "per_cell": per_cell,
    }
    _atomic_write_json(out_base / "disturbance_sweep.json", summary)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--fractions", nargs="*", default=None,
                   help="Disturbance fractions. Default: 0.001..0.10 (7 points). "
                        "Pass 0.0 to (re)compute the undisturbed baseline cell.")
    p.add_argument("--seeds", default="1-3")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1008, help="Step budget.")
    p.add_argument("--topk-ratio", type=float, default=0.10)
    p.add_argument("--batch", type=int, default=500, help="Train-pool size (full pool).")
    p.add_argument("--batch-size", type=int, default=50, help="SGD mini-batch.")
    p.add_argument("--out", default="results/training/4_exact_disturbance")
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--aggregate-only", action="store_true", default=False)
    args = p.parse_args()

    fractions = _parse_fractions(args.fractions)
    seeds = _parse_seeds(args.seeds)
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)
    (out_base / "_runs").mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        s = aggregate(out_base, seeds, args.sigma, args.topk_ratio, args.epochs)
        nt = sum(len(v) for v in s["agg_trained"].values())
        print(f"[dct4_dist] aggregated {len(s['per_cell'])} cells "
              f"({nt} f x rho trained points) -> {out_base/'disturbance_sweep.json'}")
        return 0

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import numpy as np
    import pdft
    import pdft.io  # noqa: F401
    from pdft_benchmarks.datasets.div2k import load_div2k
    from pdft_benchmarks.disturbance import disturb_controlled_dct4, flat_entry_count
    from pdft_benchmarks.evaluation import evaluate_basis_shared
    from pdft_benchmarks.experiment_utils import git_sha
    from pdft_benchmarks.presets import get_preset

    if not hasattr(pdft, "DCT4Basis"):
        print(f"[dct4_dist] FATAL: pdft at {pdft.__file__} lacks DCT4Basis.", file=sys.stderr)
        return 3

    chosen = jax.devices()[0]
    print(f"[dct4_dist] device={chosen} platform={chosen.platform!r} "
          f"pdft={pdft.__version__} @ {Path(pdft.__file__).parent}")
    if args.gpu is not None and chosen.platform not in ("gpu", "cuda"):
        print("[dct4_dist] FATAL: --gpu requested but JAX sees CPU. Aborting.", file=sys.stderr)
        return 2

    m = n = 8
    preset = get_preset("div2k_8q", "generalized")
    train_pool, test_imgs = load_div2k(n_train=preset.n_train, n_test=preset.n_test,
                                       seed=42, size=2 ** m)
    k_train = max(1, round(2 ** (m + n) * args.topk_ratio))
    loss = pdft.MSELoss(k=k_train)
    exact = pdft.DCT4Basis(m, n, parametrization="controlled")
    ntot = flat_entry_count(exact)
    dataset = [jax.device_put(np.asarray(train_pool[i]).astype(np.complex128), chosen)
               for i in range(len(train_pool))]
    steps_per_epoch = max(1, -(-args.batch // args.batch_size))
    epochs_to_run = max(1, -(-args.epochs // steps_per_epoch))
    print(f"[dct4_dist] m=n={m} entries={ntot} sigma={args.sigma} topk={args.topk_ratio} "
          f"k={k_train} batch={args.batch}/{args.batch_size} epochs={epochs_to_run} "
          f"fractions={fractions} seeds={seeds}")

    def _psnr(basis) -> dict:
        metrics, _ = evaluate_basis_shared(basis, test_imgs, keep_ratios=KEEP_RATIOS)
        return {rk: float(metrics[rk]["mean_psnr"]) for rk in RHO_KEYS}

    # f=0 is a single deterministic baseline cell; keep it out of the seed loop.
    grid = [(0.0, 0)] if 0.0 in fractions else []
    grid += [(f, s) for f in fractions if f != 0.0 for s in seeds]

    for f, seed in grid:
        cell = _cell_path(out_base, f, seed)
        if cell.exists() and not args.force:
            print(f"[dct4_dist] skip f{_fkey(f)} seed{seed:03d} (exists)")
            continue
        if not args.force and not _try_claim(cell):
            print(f"[dct4_dist] skip f{_fkey(f)} seed{seed:03d} (claimed)")
            continue

        rng = _seed_for(f, seed)
        perturbed, n_sel = disturb_controlled_dct4(exact, f, rng, sigma=args.sigma)
        psnr_untrained = _psnr(perturbed)

        t0 = time.perf_counter()
        res = pdft.train_basis_batched(
            perturbed, dataset=dataset, loss=loss, epochs=epochs_to_run,
            batch_size=args.batch_size, optimizer=preset.optimizer,
            validation_split=0.0, early_stopping_patience=10 ** 9,
            warmup_frac=preset.warmup_frac, lr_peak=preset.lr_peak,
            lr_final=preset.lr_final, max_grad_norm=preset.max_grad_norm,
            shuffle=True, seed=42, val_every_k_epochs=preset.val_every_k_epochs)
        elapsed = time.perf_counter() - t0
        psnr_trained = _psnr(res.basis)
        init_loss = float(res.loss_history[0]) if res.loss_history else None
        final_loss = float(res.loss_history[-1]) if res.loss_history else None
        print(f"[dct4_dist] f{_fkey(f)} seed{seed:03d}: n_pert={n_sel}/{ntot} "
              f"{res.steps} steps {elapsed:.0f}s untr@.20={psnr_untrained['0.2']:.3f} "
              f"tr@.20={psnr_trained['0.2']:.3f} dB")

        _atomic_write_json(cell, {
            "f": f, "seed": seed, "sigma": args.sigma, "n_perturbed": n_sel,
            "n_entries": ntot, "m": m, "n": n, "parametrization": "controlled",
            "topk_ratio": args.topk_ratio, "k_train": k_train,
            "batch": args.batch, "batch_size": args.batch_size, "epochs": args.epochs,
            "psnr_untrained": psnr_untrained, "psnr_trained": psnr_trained,
            "init_loss": init_loss, "final_loss": final_loss,
            "total_steps": int(res.steps), "elapsed_seconds": elapsed,
            "device": str(chosen), "git_sha": git_sha(short=False),
        })
        _release_claim(cell)

    aggregate(out_base, seeds, args.sigma, args.topk_ratio, args.epochs)
    print(f"[dct4_dist] done. Summary: {out_base/'disturbance_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
