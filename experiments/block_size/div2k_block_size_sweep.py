#!/usr/bin/env python3
"""DIV2K-8q block-size sweep (m=n=8, 256×256). Trains the 12-cell trained
grid: {blocked, rich, real_rich} × {4, 8, 16, 32}, plus extended classical
b-sweep baselines.

Two-GPU split — set CUDA_VISIBLE_DEVICES BEFORE importing pdft_benchmarks
(JAX preallocates ~75% on first call):

    # Terminal A — 6 cells on GPU 0
    python experiments/block_size/div2k_block_size_sweep.py --gpu 0 \\
        --bases blocked_4,rich_4,real_rich_4,blocked_8,rich_8,real_rich_8 \\
        --out results/block_size_sweep/div2k_8q/_runs/g0

    # Terminal B — 6 cells on GPU 1
    python experiments/block_size/div2k_block_size_sweep.py --gpu 1 \\
        --bases blocked_16,rich_16,real_rich_16,blocked_32,rich_32,real_rich_32 \\
        --out results/block_size_sweep/div2k_8q/_runs/g1

After both finish, cellify both run dirs into the canonical
`results/block_size_sweep/div2k_8q/by_basis/` tree.
"""

import argparse
import os
import sys

ALL_BASES = (
    "blocked_4",    "rich_4",    "real_rich_4",
    "blocked_8",    "rich_8",    "real_rich_8",
    "blocked_16",   "rich_16",   "real_rich_16",
    "blocked_32",   "rich_32",   "real_rich_32",
)
BASELINES = (
    "fft", "dct", "pca", "bd_pca",
    "block_fft_4",   "block_fft_8",   "block_fft_16",   "block_fft_32",   "block_fft_64",   "block_fft_128",
    "block_dct_4",   "block_dct_8",   "block_dct_16",   "block_dct_32",   "block_dct_64",   "block_dct_128",
    "block_pca_4",   "block_pca_8",   "block_pca_16",   "block_pca_32",   "block_pca_64",   "block_pca_128",
    "block_bd_pca_4","block_bd_pca_8","block_bd_pca_16","block_bd_pca_32","block_bd_pca_64","block_bd_pca_128",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DIV2K-8q block-size sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--preset", default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--bases", default=",".join(ALL_BASES),
                        help=f"Comma-separated subset of {ALL_BASES}.")
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    bases = [b.strip() for b in args.bases.split(",") if b.strip()]
    unknown = [b for b in bases if b not in ALL_BASES]
    if unknown:
        print(f"[div2k-sweep] unknown bases: {unknown}; valid: {ALL_BASES}", file=sys.stderr)
        return 2

    from pdft_benchmarks.experiment_utils import apply_preset_overrides
    from pdft_benchmarks.pipeline import run_experiment

    # Note: the preset registry key is "div2k_8q" (the experiment) but the
    # data-loader key passed to run_experiment below is "div2k" (the dataset).
    # The asymmetry is intentional and matches div2k_8q_pca_vs_block_dct.py.
    preset = apply_preset_overrides(
        args.preset, dataset="div2k_8q", tag="div2k-sweep",
        no_early_stop=args.no_early_stop, epochs=args.epochs,
    )

    res = run_experiment(
        dataset="div2k",
        m=8, n=8,
        bases=bases,
        baselines=list(BASELINES),
        preset=preset,
        output_dir=args.out,
        # When CUDA_VISIBLE_DEVICES already isolated GPU N, the JAX-visible
        # device is index 0 inside this process. Pass "auto" so JAX picks it.
        device="auto",
    )
    print(f"\nDone. Results: {res.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
