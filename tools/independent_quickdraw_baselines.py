#!/usr/bin/env python3
"""Independent rerun of the QuickDraw baselines (n_train=500, m=n=5).

Runs every registered baseline — including PCA and the rank-truncation
variants — against a fresh load of the QuickDraw test set, with no
trained bases. Produces a single metrics.json + a Markdown report so
the numbers in docs/global_pca_vs_block_dct.typ can be verified end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/structure/quickdraw_pca_vs_block_dct/independent_reruns/seed_default",
                    help="Output directory")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--img-size", type=int, default=32)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from pdft_benchmarks.baselines import BASELINE_FACTORIES
    from pdft_benchmarks.datasets import load_quickdraw
    from pdft_benchmarks.evaluation import evaluate_baseline

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[indep] loading QuickDraw n_train={args.n_train}, n_test={args.n_test}, "
          f"img_size={args.img_size}, seed={args.seed}…")
    train, test = load_quickdraw(
        args.n_train, args.n_test, seed=args.seed, img_size=args.img_size
    )
    print(f"[indep] train shape={train.shape}, test shape={test.shape}")

    keep_ratios = (0.05, 0.10, 0.15, 0.20)

    metrics: dict = {}
    for name in sorted(BASELINE_FACTORIES):
        print(f"[indep] fitting + evaluating baseline '{name}'…")
        builder = BASELINE_FACTORIES[name]
        fn = builder(list(train))
        kr_metrics, elapsed = evaluate_baseline(fn, test, keep_ratios)
        metrics[name] = {"metrics": kr_metrics, "time_s": elapsed}
        print(f"[indep]   done in {elapsed:.1f}s")

    out_metrics = out / "metrics.json"
    out_metrics.write_text(json.dumps({
        "config": {
            "dataset": "quickdraw",
            "n_train": args.n_train,
            "n_test": args.n_test,
            "img_size": args.img_size,
            "seed": args.seed,
            "keep_ratios": list(keep_ratios),
        },
        "metrics": metrics,
    }, indent=2))
    print(f"[indep] wrote {out_metrics}")

    # Build a small Markdown report for human reading.
    report = out / "REPORT.md"
    lines = []
    lines.append("# Independent QuickDraw baseline rerun\n")
    lines.append(f"- dataset: quickdraw\n- n_train: {args.n_train}\n- n_test: {args.n_test}\n"
                 f"- img_size: {args.img_size}\n- seed: {args.seed}\n"
                 f"- baselines: {sorted(BASELINE_FACTORIES)}\n\n")

    lines.append("## PSNR (dB) by keep ratio\n")
    header = "| baseline | " + " | ".join(f"{r:.2f}" for r in keep_ratios) + " | time (s) |"
    sep = "|" + "---|" * (len(keep_ratios) + 2)
    lines.append(header)
    lines.append(sep)
    for name in sorted(metrics):
        m = metrics[name]
        row = [name] + [f"{m['metrics'][f'{r}']['mean_psnr']:.2f}" for r in keep_ratios] + \
              [f"{m['time_s']:.1f}"]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n## MSE by keep ratio\n")
    lines.append(header.replace("time (s)", "time (s)"))
    lines.append(sep)
    for name in sorted(metrics):
        m = metrics[name]
        row = [name] + [f"{m['metrics'][f'{r}']['mean_mse']:.5f}" for r in keep_ratios] + \
              [f"{m['time_s']:.1f}"]
        lines.append("| " + " | ".join(row) + " |")
    report.write_text("\n".join(lines) + "\n")
    print(f"[indep] wrote {report}")


if __name__ == "__main__":
    main()
