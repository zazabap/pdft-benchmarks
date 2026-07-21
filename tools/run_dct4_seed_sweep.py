#!/usr/bin/env python3
"""Parallel dispatcher for the DCT-IV normal-training seed sweep
(experiments/dct4_seed_sweep.py).

Fans per-seed jobs across IDLE GPUs with a dynamic free-list: each GPU slot
pulls the next pending seed, so no GPU idles while work remains. Jobs whose
checkpoint cell already exists are skipped up front, so the pool is fully
resumable — kill it, re-run, and only the gaps refill. A heartbeat
`<out>/_progress.json` records done / running / remaining while it runs.

By default it auto-detects idle GPUs (memory.used below --idle-mib) so it never
packs onto a card running another tenant's job; pass --gpus to pin a set.

Each job is a `dct4_seed_sweep.py` subprocess pinned to one GPU via
CUDA_VISIBLE_DEVICES (+ XLA_PYTHON_CLIENT_PREALLOCATE=false, CUDA_DEVICE_ORDER=
PCI_BUS_ID). `--pdft-src PATH` prepends a pdft checkout to the subprocess
PYTHONPATH (use the latest pdft with DCT4Basis if the installed one is stale).

Usage:
    python tools/run_dct4_seed_sweep.py --seeds 1-100 \
        --python /opt/conda/envs/pdft/bin/python \
        --pdft-src /workspaces/parametric-dft-paper/pdft-dct4main/src
    python tools/run_dct4_seed_sweep.py --seeds 1-100 --gpus 1,3,4,6 --dry-run
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
DRIVER = REPO / "experiments" / "dct4_seed_sweep.py"


def _parse_ints(spec: str) -> list[int]:
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


def _detect_idle_gpus(idle_mib: int) -> list[int]:
    """GPUs whose used memory is below `idle_mib` (never pack onto a tenant)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dispatch] nvidia-smi failed ({e}); pass --gpus explicitly.",
              file=sys.stderr)
        return []
    idle = []
    for line in out.strip().splitlines():
        idx, used = (x.strip() for x in line.split(","))
        if int(used) < idle_mib:
            idle.append(int(idx))
    return idle


def _cell_path(out_base: Path, seed: int) -> Path:
    return out_base / "_runs" / f"seed_{seed:03d}.json"


def _chunk(seq: list[int], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="div2k_8q",
                   choices=["div2k_8q", "quickdraw_5q", "tuberlin_8q"])
    p.add_argument("--seeds", default="1-100")
    p.add_argument("--gpus", default=None,
                   help='GPU ids, e.g. "1,3,4,6" or "0-9". Default: auto-detect idle.')
    p.add_argument("--idle-mib", type=int, default=1000,
                   help="A GPU is 'idle' if memory.used < this (MiB). Default 1000.")
    p.add_argument("--procs-per-gpu", type=int, default=1,
                   help="Concurrent processes per GPU. A batch-50 DIV2K run peaks "
                        "~33 GB, so 1 per 48 GB card; raise only with headroom.")
    p.add_argument("--seeds-per-job", type=int, default=1,
                   help="Seeds batched per process (amortises JAX startup). 1 for DIV2K.")
    p.add_argument("--epochs", type=int, default=1008)
    p.add_argument("--topk-ratio", type=float, default=0.20)
    p.add_argument("--out", default=None)
    p.add_argument("--no-trace", action="store_true", default=False)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--pdft-src", default=None,
                   help="Prepend this pdft checkout 'src' to the subprocess "
                        "PYTHONPATH (use the latest pdft with DCT4Basis).")
    p.add_argument("--cache-dir", default="/tmp/jax_compile_cache_dct4_seed",
                   help="Shared XLA persistent-compilation-cache dir. Empty disables.")
    p.add_argument("--stagger", type=float, default=3.0,
                   help="Seconds between launches (dodges the cuBLASLt init race).")
    p.add_argument("--dry-run", action="store_true", default=False)
    args = p.parse_args()

    seeds = _parse_ints(args.seeds)
    gpus = _parse_ints(args.gpus) if args.gpus else _detect_idle_gpus(args.idle_mib)
    if not gpus:
        print("[dispatch] no GPUs available (auto-detect found none idle). "
              "Pass --gpus.", file=sys.stderr)
        return 1
    out_base = Path(args.out) if args.out else \
        REPO / "results/training/2_direct_training/random_seed" / f"dct_{args.dataset}"
    out_base.mkdir(parents=True, exist_ok=True)

    jobs: deque = deque()
    n_total = n_skipped = 0
    pending = []
    for s in seeds:
        n_total += 1
        if not args.force and _cell_path(out_base, s).exists():
            n_skipped += 1
            continue
        pending.append(s)
    for chunk in _chunk(pending, args.seeds_per_job):
        jobs.append(chunk)

    n_jobs = len(jobs)
    n_runs = sum(len(c) for c in jobs)
    print(f"[dispatch] dataset={args.dataset} seeds={len(seeds)} -> {n_total} cells, "
          f"{n_skipped} already done, {n_runs} to run in {n_jobs} jobs across GPUs "
          f"{gpus} x{args.procs_per_gpu} ({len(gpus)*args.procs_per_gpu} slots), "
          f"epochs={args.epochs}")
    if args.dry_run:
        for chunk in list(jobs)[:12]:
            print(f"  job: seeds {chunk[0]}..{chunk[-1]} ({len(chunk)})")
        if n_jobs > 12:
            print(f"  ... +{n_jobs - 12} more jobs")
        return 0
    if n_jobs == 0:
        print("[dispatch] nothing to do — all cells present.")
        return 0

    slots = [(g, k) for g in gpus for k in range(args.procs_per_gpu)]
    free = deque(slots)
    running: dict = {}
    t_start = time.time()
    done = 0

    def write_progress():
        out_base.joinpath("_progress.json").write_text(json.dumps({
            "dataset": args.dataset, "n_runs": n_runs, "n_jobs": n_jobs,
            "jobs_done": done, "jobs_running": len(running),
            "jobs_remaining": len(jobs), "gpus": gpus,
            "elapsed_seconds": round(time.time() - t_start, 1),
            "running": [{"gpu": slot[0], "seeds": [c[0], c[-1]],
                         "elapsed_s": round(time.time() - t0, 1)}
                        for (_pk, (_pr, slot, c, t0, _fh)) in running.items()],
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))

    def launch(slot, chunk):
        gpu = slot[0]
        seed_arg = ",".join(str(s) for s in chunk)
        cmd = [args.python, str(DRIVER), "--gpu", str(gpu),
               "--dataset", args.dataset, "--seeds", seed_arg,
               "--epochs", str(args.epochs), "--topk-ratio", str(args.topk_ratio),
               "--out", str(out_base)]
        if args.no_trace:
            cmd.append("--no-trace")
        if args.force:
            cmd.append("--force")
        env = dict(os.environ)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["PYTHONUNBUFFERED"] = "1"  # flush driver prints to the per-job log live
        if args.pdft_src:
            prev = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = args.pdft_src + (os.pathsep + prev if prev else "")
        if args.cache_dir:
            env["JAX_COMPILATION_CACHE_DIR"] = args.cache_dir
            env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
        log = out_base / "_runs" / f"_job_{chunk[0]:03d}_{chunk[-1]:03d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = log.open("w")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        print(f"[dispatch] launch gpu{gpu} seeds {chunk[0]}..{chunk[-1]} "
              f"-> {log.name} (pid {proc.pid})")
        return (proc, fh)

    while jobs or running:
        while jobs and free:
            slot = free.popleft()
            chunk = jobs.popleft()
            proc, fh = launch(slot, chunk)
            running[proc.pid] = (proc, slot, chunk, time.time(), fh)
            write_progress()
            time.sleep(args.stagger)
        time.sleep(2.0)
        for pid in list(running.keys()):
            proc, slot, chunk, t0, fh = running[pid]
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            done += 1
            free.append(slot)
            del running[pid]
            tag = f"seeds {chunk[0]}..{chunk[-1]}"
            if rc == 0:
                print(f"[dispatch] DONE  gpu{slot[0]} {tag} "
                      f"({time.time()-t0:.0f}s)  [{done}/{n_jobs}]")
            else:
                print(f"[dispatch] FAIL  gpu{slot[0]} {tag} rc={rc} — "
                      f"see log; cells will refill on re-run")
            write_progress()

    subprocess.run([args.python, str(DRIVER), "--aggregate-only",
                    "--dataset", args.dataset, "--seeds", args.seeds,
                    "--topk-ratio", str(args.topk_ratio), "--out", str(out_base)],
                   check=False)
    print(f"[dispatch] all jobs finished in {time.time()-t_start:.0f}s. "
          f"Summary: {out_base/'seed_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
