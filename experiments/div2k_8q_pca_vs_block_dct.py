#!/usr/bin/env python3
"""DIV2K-8q PCA-vs-block-DCT benchmark (m=n=8, 256×256 grayscale).

Trains the seven registered parametric bases (qft, entangled_qft, tebd,
mera, blocked_8, rich_8, real_rich_8) and evaluates each against six
classical baselines: global FFT, global DCT, 8×8-block FFT, 8×8-block
DCT, global PCA, and 8×8-block PCA. Includes PCA + block-PCA in the
trained run so the writeup's headline numbers come from a single source.

MERA actually runs at this geometry (m+n=16 = 2^4, unlike QuickDraw
where m+n=10 silently skips MERA). Block bases use the *_8 factories
(`blocked_8`, `rich_8`, `real_rich_8`) which pin inner_m=inner_n=3 →
8×8 pixel blocks → 32×32 grid on 256×256 images. This is
apples-to-apples with classical block_dct_8 / block_fft_8 (also 8×8
patches), unlike the default `blocked` / `rich` / `real_rich` factories
which would give 16×16 blocks at m=8.

GPU isolation: the --gpu flag sets CUDA_VISIBLE_DEVICES BEFORE
importing pdft_benchmarks (which transitively imports JAX). JAX
preallocates ~75% of every visible GPU on first call; without this
guard, two parallel invocations OOM.

Outputs land in `--out` directly. To use both GPUs:

    # Terminal A — GPU 0 — 4 unblocked bases
    python experiments/div2k_8q_pca_vs_block_dct.py \\
        --gpu 0 --bases qft,entangled_qft,tebd,mera \\
        --out results/div2k_8q_pca_vs_block_dct/_runs/unblocked

    # Terminal B — GPU 1 — 3 block-wrapped bases (8×8 blocks)
    python experiments/div2k_8q_pca_vs_block_dct.py \\
        --gpu 1 --bases blocked_8,rich_8,real_rich_8 \\
        --out results/div2k_8q_pca_vs_block_dct/_runs/blocked

After both finish, cellify with `tools/cellify_run.py` to assemble the
canonical `by_basis/<basis>/` tree.
"""

import argparse
import os
import sys

DEFAULT_BASES = ("qft", "entangled_qft", "tebd", "mera",
                 "blocked_8", "rich_8", "real_rich_8")
# Ablation variants — selectable via --bases but not in the default
# headline grid.
EXTRA_BASES = ("qft_identity", "dct4_ctl")
ALL_BASES = DEFAULT_BASES + EXTRA_BASES
BASELINES = ("fft", "dct", "block_fft_8", "block_dct_8", "pca", "block_pca_8",
             "dct_rank", "block_dct_8_rank", "pca_rank", "block_pca_8_rank",
             "bd_pca", "block_bd_pca_8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DIV2K-8q PCA-vs-block-DCT benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--preset", default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index. Sets CUDA_VISIBLE_DEVICES before JAX import.")
    parser.add_argument("--out", default=None,
                        help="Output directory (None → timestamped default).")
    parser.add_argument("--bases", default=",".join(DEFAULT_BASES),
                        help=f"Comma-separated subset of {ALL_BASES}. "
                             f"Default: {DEFAULT_BASES}.")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early-stopping-on-validation-plateau. "
                             "Train for the preset's full epoch budget. Useful "
                             "for fair cross-basis comparison of training "
                             "trajectories on the same x-axis.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override preset.epochs. With batch_size=50 and "
                             "n_train=500 (val_split=0.15 → 425 train), each "
                             "epoch is 9 optimizer steps; epochs=223 ≈ 2000 "
                             "steps total.")
    args = parser.parse_args()

    # CRITICAL: set CUDA_VISIBLE_DEVICES BEFORE importing pdft_benchmarks.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    bases = [b.strip() for b in args.bases.split(",") if b.strip()]
    unknown = [b for b in bases if b not in ALL_BASES]
    if unknown:
        print(f"unknown basis name(s): {unknown}; "
              f"choices: {sorted(ALL_BASES)}", file=sys.stderr)
        return 2

    # Imports below trigger JAX device discovery; must come AFTER the env var.
    # (experiment_utils is JAX-free, but kept here alongside the pipeline import
    # for locality.)
    from pdft_benchmarks.experiment_utils import apply_preset_overrides
    from pdft_benchmarks.pipeline import run_experiment

    # Preset-registry key is "div2k_8q"; the dataset-loader key below is "div2k".
    preset = apply_preset_overrides(
        args.preset, dataset="div2k_8q", tag="div2k-8q",
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
    sys.exit(main())
