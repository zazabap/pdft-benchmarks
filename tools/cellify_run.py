#!/usr/bin/env python3
"""Split a flat `run_experiment` output dir into a by_basis/<basis>/ tree.

`pdft_benchmarks.pipeline.run_experiment` writes a flat output dir:

    <in>/metrics.json                # top-level dict keyed by basis name
    <in>/env.json
    <in>/trained_<basis>.json        # one per basis
    <in>/loss_history/<basis>_loss.json
    <in>/<baseline>_eigenbasis.npz   # for PCA-class baselines

This tool produces, per basis listed in `metrics.json`:

    <out>/<basis>/metrics.json       # the aggregate's [basis] subtree
    <out>/<basis>/env.json           # copy of the shared run env
    <out>/<basis>/trained_<basis>.json   # move
    <out>/<basis>/loss_history/<basis>_loss.json   # move
    <out>/<basis>/<baseline>_eigenbasis.npz        # copy (shared across cells)

Pure-Python: no JAX, no pdft_benchmarks import — runs without a GPU.

CLI:
    python tools/cellify_run.py --in <flat-dir> --out <by_basis-dst>
                                [--bases comma,sep] [--keep-source]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def cellify(in_dir: Path, out_dir: Path, bases: list[str] | None,
            keep_source: bool) -> int:
    metrics_path = in_dir / "metrics.json"
    env_path = in_dir / "env.json"

    if not metrics_path.exists():
        print(f"[cellify] FAIL: {metrics_path} not found", file=sys.stderr)
        return 1
    if not env_path.exists():
        print(f"[cellify] FAIL: {env_path} not found", file=sys.stderr)
        return 1

    metrics = json.loads(metrics_path.read_text())
    env = env_path.read_text()  # copied verbatim into each cell

    target_bases = bases if bases else list(metrics.keys())
    missing = [b for b in target_bases if b not in metrics]
    if missing:
        print(f"[cellify] FAIL: bases not in metrics.json: {missing}",
              file=sys.stderr)
        return 1

    # Eigenbasis files (shared across cells; copied so cells are self-contained)
    eigenbasis_files = list(in_dir.glob("*_eigenbasis.npz"))

    out_dir.mkdir(parents=True, exist_ok=True)
    for basis in target_bases:
        cell = out_dir / basis
        cell.mkdir(parents=True, exist_ok=True)

        # 1. metrics.json — single-basis subset
        (cell / "metrics.json").write_text(
            json.dumps({basis: metrics[basis]}, indent=2)
        )

        # 2. env.json — copy
        (cell / "env.json").write_text(env)

        # 3. trained_<basis>.json — move (or copy if --keep-source)
        trained = in_dir / f"trained_{basis}.json"
        if trained.exists():
            target = cell / f"trained_{basis}.json"
            if keep_source:
                shutil.copy2(trained, target)
            else:
                shutil.move(str(trained), str(target))

        # 4. loss_history/<basis>_loss.json — move
        loss = in_dir / "loss_history" / f"{basis}_loss.json"
        if loss.exists():
            (cell / "loss_history").mkdir(parents=True, exist_ok=True)
            target = cell / "loss_history" / f"{basis}_loss.json"
            if keep_source:
                shutil.copy2(loss, target)
            else:
                shutil.move(str(loss), str(target))

        # 5. eigenbasis files — copy (shared across cells)
        for eb in eigenbasis_files:
            shutil.copy2(eb, cell / eb.name)

        print(f"[cellify] wrote {cell}/")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", required=True, type=Path,
                    help="Flat run_experiment output dir.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Destination by_basis/ root.")
    ap.add_argument("--bases", default=None,
                    help="Optional comma-separated subset to cellify.")
    ap.add_argument("--keep-source", action="store_true",
                    help="Copy instead of move; leaves <in> intact.")
    args = ap.parse_args()
    bases = [b.strip() for b in args.bases.split(",")] if args.bases else None
    return cellify(args.in_dir, args.out, bases, args.keep_source)


if __name__ == "__main__":
    sys.exit(main())
