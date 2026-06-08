#!/usr/bin/env python3
"""QuickDraw block-size sweep (m=n=5, 32×32). Trains the 9 trained-basis
cells of the sweep grid: {blocked, rich, real_rich} × {4, 8, 16}.

Mirrors `experiments/quickdraw_pca_vs_block_dct.py` (single-process,
all bases on one GPU). Runs at the headline 1008-step preset by default
(`--epochs 112 --no-early-stop`) so each cell is comparable against the
existing b=8 headline.

Outputs land in `--out` directly. After the run, cellify into the canonical
`results/block_size_sweep/quickdraw/by_basis/` tree with `tools/cellify_run.py`.
"""

# Note: this is a single-process script. --gpu selects the JAX device for
# this Python process; it does NOT set CUDA_VISIBLE_DEVICES (unlike the
# DIV2K sweep which splits across two GPUs and needs OS-level isolation).
# QuickDraw is small enough that one process holds both training and the
# 20+ classical baselines, so a single visible GPU is fine.

import argparse

from pdft_benchmarks.experiment_utils import apply_preset_overrides
from pdft_benchmarks.pipeline import run_experiment


TRAINED_BASES = (
    "blocked_4", "blocked_8", "blocked_16",
    "rich_4",    "rich_8",    "rich_16",
    "real_rich_4", "real_rich_8", "real_rich_16",
)
# Classical baselines at the full QuickDraw classical grid (b ∈ {2,4,8,16,32}).
# b=32 = full image = global dct/fft/pca/bd_pca.
BASELINES = (
    "fft", "dct", "pca", "bd_pca",
    "block_fft_2", "block_fft_4", "block_fft_8", "block_fft_16",
    "block_dct_2", "block_dct_4", "block_dct_8", "block_dct_16",
    "block_pca_2", "block_pca_4", "block_pca_8", "block_pca_16",
    "block_bd_pca_2", "block_bd_pca_4", "block_bd_pca_8", "block_bd_pca_16",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early-stopping (recommended for headline runs).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override preset.epochs (headline = 112).")
    parser.add_argument("--bases", default=",".join(TRAINED_BASES),
                        help=f"Comma-separated subset of {TRAINED_BASES}.")
    args = parser.parse_args()

    bases = [b.strip() for b in args.bases.split(",") if b.strip()]
    unknown = [b for b in bases if b not in TRAINED_BASES]
    if unknown:
        import sys
        print(f"[quickdraw-sweep] unknown bases: {unknown}; valid: {TRAINED_BASES}", file=sys.stderr)
        raise SystemExit(2)

    preset = apply_preset_overrides(
        args.preset, dataset="quickdraw", tag="quickdraw-sweep",
        no_early_stop=args.no_early_stop, epochs=args.epochs,
    )

    res = run_experiment(
        dataset="quickdraw",
        m=5, n=5,
        bases=bases,
        baselines=list(BASELINES),
        preset=preset,
        output_dir=args.out,
        device=f"gpu:{args.gpu}" if args.gpu is not None else "auto",
    )
    print(f"\nDone. Results: {res.output_dir}")


if __name__ == "__main__":
    main()
