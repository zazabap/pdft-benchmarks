#!/usr/bin/env python3
"""Parallel dispatcher for the random-seed unfreeze sweep (experiments/qft_seed_sweep.py).

Fans (ordering, seed) jobs across many GPUs with a dynamic free-list: each GPU
slot pulls the next pending job, so no GPU idles while work remains. Jobs whose
checkpoint cell already exists are skipped up front, so the pool is fully
resumable — kill it, re-run, and only the gaps refill. A heartbeat
`<out>/_progress.json` records done / running / remaining while it runs.

Each job is a `qft_seed_sweep.py` subprocess pinned to one GPU via
CUDA_VISIBLE_DEVICES (+ XLA_PYTHON_CLIENT_PREALLOCATE=false, the shared-box
requirement). `--seeds-per-job` batches several seeds into one process to
amortise JAX startup — keep it 1 for DIV2K (runs dwarf startup) and large for
QuickDraw (seconds/run, startup dominates). Per-seed checkpoint granularity is
preserved regardless (the driver writes one cell per seed).

Usage:
    # DIV2K full run: 3 orderings x 100 seeds across 8 GPUs, 1 seed/process
    python tools/run_seed_sweep.py --dataset div2k_8q --seeds 1-100 \
        --gpus 0,1,3,7,8,9 --procs-per-gpu 1 --seeds-per-job 1
    # QuickDraw pilot: batch 25 seeds/process so startup doesn't dominate
    python tools/run_seed_sweep.py --dataset quickdraw_5q --seeds 1-100 \
        --gpus 0,1 --seeds-per-job 25 --no-trace
    python tools/run_seed_sweep.py --dataset div2k_8q --seeds 1-100 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "experiments" / "qft_seed_sweep.py"
DEFAULT_PY = REPO / ".venv" / "bin" / "python"


def _parse_seeds(spec: str) -> list[int]:
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


def _parse_gpus(spec: str) -> list[int]:
    return _parse_seeds(spec)  # same grammar (ranges + lists)


def _cell_path(out_base: Path, ordering: str, seed: int) -> Path:
    return out_base / "_runs" / ordering / f"seed_{seed:03d}.json"


def _chunk(seq: list[int], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="div2k_8q",
                   choices=["div2k_8q", "quickdraw_5q", "tuberlin_8q"])
    p.add_argument("--orderings", default="bg,lr,rl")
    p.add_argument("--seeds", default="1-100")
    p.add_argument("--gpus", default="0-9", help='GPU ids, e.g. "0,1,3" or "0-9".')
    p.add_argument("--procs-per-gpu", type=int, default=1,
                   help="Concurrent processes per GPU. Raise to 2 only if a run's "
                        "peak memory leaves headroom on the 48 GB cards.")
    p.add_argument("--seeds-per-job", type=int, default=1,
                   help="Seeds batched into one process (amortises JAX startup). "
                        "1 for DIV2K; large (e.g. 25) for QuickDraw.")
    p.add_argument("--topk-ratio", type=float, default=0.20)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--grad-check-every", type=int, default=5)
    p.add_argument("--out", default=None)
    p.add_argument("--no-trace", action="store_true", default=False)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--python", default=str(DEFAULT_PY))
    p.add_argument("--cache-dir", default="/tmp/jax_compile_cache_seed_sweep",
                   help="Shared XLA persistent-compilation-cache dir. Stage-k of "
                        "an ordering compiles identically for every seed, so the "
                        "first seed populates the cache and the rest hit it "
                        "(huge win — the 72 per-stage recompiles are the hidden "
                        "cost). Empty string disables.")
    p.add_argument("--stagger", type=float, default=3.0,
                   help="Seconds between launches (dodges the cuBLASLt init race).")
    p.add_argument("--reverse", action="store_true", default=False,
                   help="Process the job list back-to-front. Lets a SECOND "
                        "dispatcher share the same GPUs (2 procs/GPU) and sweep "
                        "from the opposite end — the driver's skip-existing guard "
                        "makes the two sweeps converge with minimal overlap.")
    p.add_argument("--dry-run", action="store_true", default=False)
    args = p.parse_args()

    orderings = [s.strip() for s in args.orderings.split(",") if s.strip()]
    seeds = _parse_seeds(args.seeds)
    gpus = _parse_gpus(args.gpus)
    out_base = Path(args.out) if args.out else \
        REPO / "results/training/2_direct_training/random_seed" / args.dataset
    out_base.mkdir(parents=True, exist_ok=True)

    # Build the job list: (ordering, [seed,...]) chunks, dropping cells already done.
    jobs: deque = deque()
    n_total = n_skipped = 0
    for ordering in orderings:
        pending = []
        for s in seeds:
            n_total += 1
            if not args.force and _cell_path(out_base, ordering, s).exists():
                n_skipped += 1
                continue
            pending.append(s)
        for chunk in _chunk(pending, args.seeds_per_job):
            jobs.append((ordering, chunk))

    if args.reverse:
        jobs.reverse()

    n_jobs = len(jobs)
    n_runs = sum(len(c) for _, c in jobs)
    print(f"[dispatch] dataset={args.dataset} orderings={orderings} "
          f"seeds={len(seeds)} -> {n_total} cells, {n_skipped} already done, "
          f"{n_runs} to run in {n_jobs} jobs across GPUs {gpus} "
          f"x{args.procs_per_gpu} ({len(gpus)*args.procs_per_gpu} slots)")
    if args.dry_run:
        for ordering, chunk in list(jobs)[:12]:
            print(f"  job: {ordering} seeds {chunk[0]}..{chunk[-1]} ({len(chunk)})")
        if n_jobs > 12:
            print(f"  ... +{n_jobs - 12} more jobs")
        return 0
    if n_jobs == 0:
        print("[dispatch] nothing to do — all cells present.")
        return 0

    # Slots = (gpu, slot_index). A free-list assigns the next idle slot to the
    # next pending job; dynamic, so fast jobs don't strand a GPU.
    slots = [(g, k) for g in gpus for k in range(args.procs_per_gpu)]
    free = deque(slots)
    running: dict = {}  # pid-key -> (proc, slot, ordering, chunk, t0)
    t_start = time.time()
    done = 0

    def write_progress():
        out_base.joinpath("_progress.json").write_text(json.dumps({
            "dataset": args.dataset, "n_runs": n_runs, "n_jobs": n_jobs,
            "jobs_done": done, "jobs_running": len(running),
            "jobs_remaining": len(jobs),
            "elapsed_seconds": round(time.time() - t_start, 1),
            "running": [{"gpu": slot[0], "ordering": o,
                         "seeds": [c[0], c[-1]], "elapsed_s": round(time.time() - t0, 1)}
                        for (_pk, (_pr, slot, o, c, t0, _fh)) in running.items()],
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))

    def launch(slot, ordering, chunk):
        gpu = slot[0]
        seed_arg = ",".join(str(s) for s in chunk)
        cmd = [args.python, str(DRIVER), "--gpu", str(gpu),
               "--dataset", args.dataset, "--orderings", ordering,
               "--seeds", seed_arg, "--topk-ratio", str(args.topk_ratio),
               "--max-steps", str(args.max_steps),
               "--grad-check-every", str(args.grad_check_every),
               "--out", str(out_base)]
        if args.no_trace:
            cmd.append("--no-trace")
        if args.force:
            cmd.append("--force")
        env = dict(os.environ)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if args.cache_dir:
            # Persistent XLA compilation cache, shared across all job processes.
            env["JAX_COMPILATION_CACHE_DIR"] = args.cache_dir
            env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
        log = out_base / "_runs" / ordering / f"_job_{chunk[0]:03d}_{chunk[-1]:03d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = log.open("w")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        print(f"[dispatch] launch gpu{gpu} {ordering} seeds {chunk[0]}..{chunk[-1]} "
              f"-> {log.name} (pid {proc.pid})")
        return (proc, fh)

    while jobs or running:
        # Fill free slots.
        while jobs and free:
            slot = free.popleft()
            ordering, chunk = jobs.popleft()
            proc, fh = launch(slot, ordering, chunk)
            running[proc.pid] = (proc, slot, ordering, chunk, time.time(), fh)
            write_progress()
            time.sleep(args.stagger)
        # Poll runners.
        time.sleep(2.0)
        for pid in list(running.keys()):
            proc, slot, ordering, chunk, t0, fh = running[pid]
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            done += 1
            free.append(slot)
            del running[pid]
            tag = f"{ordering} seeds {chunk[0]}..{chunk[-1]}"
            if rc == 0:
                print(f"[dispatch] DONE  gpu{slot[0]} {tag} "
                      f"({time.time()-t0:.0f}s)  [{done}/{n_jobs}]")
            else:
                print(f"[dispatch] FAIL  gpu{slot[0]} {tag} rc={rc} — "
                      f"see log; cells will refill on re-run")
            write_progress()

    # Final roll-up (no GPU).
    subprocess.run([args.python, str(DRIVER), "--aggregate-only",
                    "--dataset", args.dataset, "--orderings", ",".join(orderings),
                    "--seeds", args.seeds, "--topk-ratio", str(args.topk_ratio),
                    "--out", str(out_base)], check=False)
    print(f"[dispatch] all jobs finished in {time.time()-t_start:.0f}s. "
          f"Summary: {out_base/'seed_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
