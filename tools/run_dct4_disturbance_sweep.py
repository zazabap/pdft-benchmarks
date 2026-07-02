#!/usr/bin/env python3
"""Parallel dispatcher for the DCT-IV exact-init disturbance sweep
(experiments/dct4_disturbance_sweep.py). Fans (fraction, seed) cells across idle
GPUs; resumable (skips existing cells). See tools/run_dct4_seed_sweep.py for the
scheduler design this is copied from.

Usage:
    python tools/run_dct4_disturbance_sweep.py --seeds 1-3 --gpus 2,3,5,6 \
        --python /opt/conda/envs/pdft/bin/python \
        --pdft-src /workspaces/parametric-dft-paper/pdft-pr24/src
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
DRIVER = REPO / "experiments" / "dct4_disturbance_sweep.py"
FRACTIONS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10)


def _fkey(f):
    return f"{f:g}"


def _parse_ints(spec):
    out = set()
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


def _detect_idle_gpus(idle_mib):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dispatch] nvidia-smi failed ({e}); pass --gpus.", file=sys.stderr)
        return []
    idle = []
    for line in out.strip().splitlines():
        idx, used = (x.strip() for x in line.split(","))
        if int(used) < idle_mib:
            idle.append(int(idx))
    return idle


def _cell_path(out_base, f, seed):
    return out_base / "_runs" / f"f{_fkey(f)}_seed{seed:03d}.json"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="1-3")
    p.add_argument("--fractions", nargs="*", type=float, default=list(FRACTIONS))
    p.add_argument("--with-baseline", action="store_true", default=True,
                   help="Include the f=0 undisturbed baseline cell.")
    p.add_argument("--gpus", default=None)
    p.add_argument("--idle-mib", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=1008)
    p.add_argument("--topk-ratio", type=float, default=0.10)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--out", default=str(REPO / "results/training/4_exact_disturbance"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--pdft-src", default="/workspaces/parametric-dft-paper/pdft-pr24/src")
    p.add_argument("--cache-dir", default="/tmp/jax_compile_cache_dct4_dist")
    p.add_argument("--stagger", type=float, default=3.0)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    args = p.parse_args()

    seeds = _parse_ints(args.seeds)
    gpus = _parse_ints(args.gpus) if args.gpus else _detect_idle_gpus(args.idle_mib)
    if not gpus:
        print("[dispatch] no idle GPUs; pass --gpus.", file=sys.stderr)
        return 1
    out_base = Path(args.out)
    (out_base / "_runs").mkdir(parents=True, exist_ok=True)

    # Build the (f, seed) job list; f=0 is a single baseline job.
    pairs = []
    if args.with_baseline:
        pairs.append((0.0, 0))
    pairs += [(f, s) for f in args.fractions for s in seeds]
    jobs = deque()
    n_skipped = 0
    for f, s in pairs:
        if not args.force and _cell_path(out_base, f, s).exists():
            n_skipped += 1
            continue
        jobs.append((f, s))
    n_jobs = len(jobs)
    print(f"[dispatch] {len(pairs)} cells, {n_skipped} done, {n_jobs} to run "
          f"on GPUs {gpus}, epochs={args.epochs}")
    if args.dry_run:
        for f, s in list(jobs)[:16]:
            print(f"  job: f={_fkey(f)} seed={s}")
        if n_jobs > 16:
            print(f"  ... +{n_jobs - 16} more")
        return 0
    if n_jobs == 0:
        print("[dispatch] nothing to do.")
        return 0

    free = deque(gpus)
    running = {}
    t_start = time.time()
    done = 0

    def write_progress():
        out_base.joinpath("_progress.json").write_text(json.dumps({
            "n_jobs": n_jobs, "jobs_done": done, "jobs_running": len(running),
            "jobs_remaining": len(jobs), "gpus": gpus,
            "elapsed_seconds": round(time.time() - t_start, 1),
            "running": [{"gpu": g, "f": _fkey(f), "seed": s,
                         "elapsed_s": round(time.time() - t0, 1)}
                        for (_pid, (_pr, g, f, s, t0, _fh)) in running.items()],
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))

    def launch(gpu, f, s):
        cmd = [args.python, str(DRIVER), "--gpu", str(gpu),
               "--fractions", _fkey(f), "--seeds", str(s),
               "--epochs", str(args.epochs), "--topk-ratio", str(args.topk_ratio),
               "--sigma", str(args.sigma), "--out", str(out_base)]
        if args.force:
            cmd.append("--force")
        env = dict(os.environ)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["PYTHONUNBUFFERED"] = "1"
        prev = env.get("PYTHONPATH", "")
        # pdft (5365a5a) first, then this repo's src for pdft_benchmarks.
        env["PYTHONPATH"] = os.pathsep.join(
            [args.pdft_src, str(REPO / "src")] + ([prev] if prev else []))
        if args.cache_dir:
            env["JAX_COMPILATION_CACHE_DIR"] = args.cache_dir
            env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
        log = out_base / "_runs" / f"_job_f{_fkey(f)}_seed{s:03d}.log"
        fh = log.open("w")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        print(f"[dispatch] launch gpu{gpu} f={_fkey(f)} seed={s} -> {log.name} (pid {proc.pid})")
        return proc, fh

    while jobs or running:
        while jobs and free:
            gpu = free.popleft()
            f, s = jobs.popleft()
            proc, fh = launch(gpu, f, s)
            running[proc.pid] = (proc, gpu, f, s, time.time(), fh)
            write_progress()
            time.sleep(args.stagger)
        time.sleep(2.0)
        for pid in list(running):
            proc, gpu, f, s, t0, fh = running[pid]
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            done += 1
            free.append(gpu)
            del running[pid]
            tag = f"f={_fkey(f)} seed={s}"
            print(f"[dispatch] {'DONE ' if rc == 0 else 'FAIL '} gpu{gpu} {tag} "
                  f"({time.time()-t0:.0f}s) rc={rc} [{done}/{n_jobs}]")
            write_progress()

    subprocess.run([args.python, str(DRIVER), "--aggregate-only",
                    "--seeds", args.seeds, "--sigma", str(args.sigma),
                    "--topk-ratio", str(args.topk_ratio), "--out", str(out_base)],
                   check=False)
    print(f"[dispatch] finished in {time.time()-t_start:.0f}s. "
          f"Summary: {out_base/'disturbance_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
