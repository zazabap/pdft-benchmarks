#!/usr/bin/env python3
"""Append PCA / KLT baseline metrics to existing archived source runs.

Avoids retraining bases from scratch: each archived source run already has
trained-basis metrics in its metrics.json, but lacks the new "pca" and
"block_pca_8" entries (those baselines didn't exist when the runs were
generated). This script:

  1. Walks each ARCHIVED_RUN below.
  2. Reads env.json to recover (n_train, n_test, seed) of the original run.
  3. Re-loads the same dataset slice (deterministic from those kwargs).
  4. Fits and evaluates block_pca_8 (always) + global pca (skip on div2k_10q).
  5. Writes the new entries into source metrics.json, preserving existing
     bases + classical baselines. For block_pca_8, also writes the
     eigenbasis as <run>/block_pca_8_eigenbasis.npz.
  6. The standard extraction pipeline (scripts/extract_canonical_cells.py)
     then propagates the new entries into results/published/<cell>/.

Usage:
    python scripts/append_pca_to_archived_runs.py
    python scripts/append_pca_to_archived_runs.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import pdft  # noqa: F401  -- enables jax_enable_x64 for the SVD path

from pdft_benchmarks.baselines import BASELINE_FACTORIES
from pdft_benchmarks.datasets import load as load_dataset
from pdft_benchmarks.evaluation import evaluate_baseline
from pdft_benchmarks.pca import fingerprint as pca_fingerprint

# Source-run dir → (dataset, m, n). Every archived run referenced by
# scripts/extract_canonical_cells.py must appear here.
ARCHIVED_RUNS: list[tuple[str, str, int, int]] = [
    # (relative_path, dataset, m, n)
    ("div2k_8q_generalized_20260425-102013_gpu0",       "div2k", 8, 8),
    ("div2k_8q_generalized_20260425-102013_gpu1",       "div2k", 8, 8),
    ("div2k_8q_blocked_generalized_20260426-085726",    "div2k", 8, 8),
    ("div2k_8q_blocked_rich_generalized_20260426-110840", "div2k", 8, 8),
    ("div2k_8q_REAL_20260426-123029",                   "div2k", 8, 8),
    ("div2k_10q_generalized_20260426-055335_gpu0_bs2",  "div2k", 10, 10),
    ("div2k_10q_generalized_20260426-055335_gpu1_bs2",  "div2k", 10, 10),
    ("quickdraw_5q_generalized_20260427-073744",        "quickdraw", 5, 5),
]

KEEP_RATIOS = (0.05, 0.1, 0.15, 0.2)


def _baselines_for(dataset: str, m: int) -> list[str]:
    """Which classical baselines to (re)evaluate for a given run.

    Includes the new PCA + rank-truncation variants. The policy skip for
    `pca` / `pca_rank` on div2k_10q is handled inside the loop so the
    source metrics.json gets `{"skipped": ...}` entries that downstream
    extraction can propagate.
    """
    return [
        "block_pca_8",
        "pca",
        "block_pca_8_rank",
        "pca_rank",
        "dct_rank",
        "block_dct_8_rank",
    ]


def _process_run(run_dir: Path, dataset: str, m: int, n: int, *, dry: bool) -> None:
    env = json.loads((run_dir / "env.json").read_text())
    pd = env["preset_dataclass"]
    n_train, n_test, seed = pd["n_train"], pd["n_test"], pd["seed"]
    keep_ratios = tuple(pd.get("keep_ratios", KEEP_RATIOS))

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())

    print(f"\n=== {run_dir.name} ({dataset}, m={m}, n={n}, n_train={n_train}, seed={seed}) ===")
    print(f"  existing keys: {sorted(metrics.keys())}")

    # Per-dataset loader kwargs to recover the original run's image size.
    loader_kwargs: dict = {"n_train": n_train, "n_test": n_test, "seed": seed}
    if dataset == "div2k":
        # Original experiments pass size=2**m (e.g. 256 at m=8, 1024 at m=10).
        # The loader's default is 256, so DIV2K-10q would silently load a wrong
        # crop size if we don't override here.
        loader_kwargs["size"] = 2 ** m

    print(f"  loading {dataset} (kwargs={loader_kwargs})…")
    train_imgs, test_imgs = load_dataset(dataset, **loader_kwargs)
    print(f"  train_imgs.shape={train_imgs.shape}, test_imgs.shape={test_imgs.shape}")

    new_entries: dict = {}
    for baseline_name in _baselines_for(dataset, m):
        # Policy skip: any global PCA variant on div2k_10q.
        if baseline_name in ("pca", "pca_rank") and dataset == "div2k" and m == 10:
            new_entries[baseline_name] = {"skipped": "pca_intractable_at_1m_dim"}
            print(f"  [skip] {baseline_name} (intractable at d={(2 ** m)*(2 ** n)})")
            continue
        t0 = time.perf_counter()
        builder = BASELINE_FACTORIES[baseline_name]
        fn = builder(train_imgs)
        fit_s = time.perf_counter() - t0
        kr_metrics, eval_s = evaluate_baseline(fn, test_imgs, keep_ratios)
        basis = getattr(fn, "_pca_basis", None)
        payload: dict = {"metrics": kr_metrics, "time": float(fit_s + eval_s)}
        if basis is not None:
            payload["_pdft_py"] = {
                "pca_fingerprint": pca_fingerprint(basis),
                "fit_s": float(fit_s),
                "eval_s": float(eval_s),
            }
            if basis.block == 8 and not dry:
                npz_path = run_dir / f"{baseline_name}_eigenbasis.npz"
                np.savez(
                    npz_path,
                    eigenbasis=basis.eigenbasis,
                    mean=basis.mean,
                    eigenvalues=basis.eigenvalues,
                )
                print(f"  [save] {npz_path.name} ({basis.eigenbasis.shape[0]}x{basis.d})")
        # Pretty-print PSNR at the four headline keep ratios.
        psnr_str = ", ".join(
            f"kr={kr}: {kr_metrics[str(kr)]['mean_psnr']:.2f}dB"
            for kr in keep_ratios if str(kr) in kr_metrics
        )
        print(f"  [eval] {baseline_name}: fit={fit_s:.1f}s eval={eval_s:.1f}s -- {psnr_str}")
        new_entries[baseline_name] = payload

    if dry:
        print(f"  [dry-run] would write {len(new_entries)} new keys to {metrics_path.name}")
        return

    metrics.update(new_entries)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  [write] {metrics_path.name} now has keys: {sorted(metrics.keys())}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate + print but do not modify metrics.json or write .npz")
    parser.add_argument("--only", default="",
                        help="comma-separated substring filter on run dirname")
    args = parser.parse_args(argv)

    archive_root = args.results_root / "_archive"
    if not archive_root.is_dir():
        print(f"ERROR: archive root not found: {archive_root}", file=sys.stderr)
        return 2

    only_filters = [s for s in args.only.split(",") if s]
    t_total = time.perf_counter()
    for rel_path, dataset, m, n in ARCHIVED_RUNS:
        if only_filters and not any(s in rel_path for s in only_filters):
            continue
        run_dir = archive_root / rel_path
        if not run_dir.is_dir():
            print(f"SKIP: {run_dir} not found", file=sys.stderr)
            continue
        _process_run(run_dir, dataset, m, n, dry=args.dry_run)
    print(f"\nTotal wall: {time.perf_counter() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
