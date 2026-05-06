#!/usr/bin/env python3
"""Independent rerun of the DIV2K-8q baselines (n_train=500, m=n=8).

Runs every registered baseline — including PCA, block_PCA_8, and the
classical FFT/DCT family — against a fresh load of the DIV2K-8q test
set, with no trained bases. Produces a single metrics.json + a
Markdown report so the numbers in
results/div2k_8q_pca_vs_block_dct/writeup.typ can be verified.

Sets CUDA_VISIBLE_DEVICES before importing pdft_benchmarks (matches
the entry-point pattern); independent rerun is single-process so this
is mostly defensive.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="results/div2k_8q_pca_vs_block_dct/independent_reruns/seed_default",
                    help="Output directory")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--img-size", type=int, default=256)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Imports trigger JAX initialisation; must come AFTER the env var.
    from pdft_benchmarks.baselines import BASELINE_FACTORIES
    from pdft_benchmarks.datasets import load_div2k
    from pdft_benchmarks.evaluation import evaluate_baseline

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[indep] loading DIV2K n_train={args.n_train}, n_test={args.n_test}, "
          f"img_size={args.img_size}, seed={args.seed}…")
    train, test = load_div2k(
        args.n_train, args.n_test, seed=args.seed, size=args.img_size
    )
    print(f"[indep] train shape={train.shape}, test shape={test.shape}")

    keep_ratios = (0.05, 0.10, 0.15, 0.20)

    metrics: dict = {}
    for name in sorted(BASELINE_FACTORIES):
        print(f"[indep] fitting + evaluating baseline '{name}'…")
        t0 = time.perf_counter()
        builder = BASELINE_FACTORIES[name]
        fn = builder(list(train))
        kr_metrics, elapsed = evaluate_baseline(fn, test, keep_ratios)
        metrics[name] = {"metrics": kr_metrics, "time_s": elapsed}
        print(f"[indep]   done in {elapsed:.1f}s")

    out_metrics = out / "metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2))
    print(f"[indep] wrote {out_metrics}")

    # Report
    report = ["# DIV2K-8q independent baseline rerun",
              "",
              f"- seed: {args.seed}",
              f"- n_train: {args.n_train}, n_test: {args.n_test}",
              f"- img_size: {args.img_size}",
              f"- baselines: {sorted(BASELINE_FACTORIES)}",
              "",
              "## PSNR (mean over test set, keep ratios = 0.05/0.10/0.15/0.20)",
              "",
              "| baseline | 0.05 | 0.10 | 0.15 | 0.20 |",
              "|---|---|---|---|---|"]
    for name in sorted(metrics):
        m = metrics[name]["metrics"]
        psnrs = [f"{m[str(k)]['mean_psnr']:.2f}" for k in (0.05, 0.10, 0.15, 0.20)]
        report.append(f"| {name} | {' | '.join(psnrs)} |")
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"[indep] wrote {out / 'REPORT.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
