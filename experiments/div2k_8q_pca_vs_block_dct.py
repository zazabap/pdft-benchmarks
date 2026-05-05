#!/usr/bin/env python3
"""DIV2K-8q PCA-vs-block-DCT benchmark (m=n=8, 256×256 grayscale).

Trains the seven registered parametric bases (qft, entangled_qft, tebd,
mera, blocked, rich, real_rich) and evaluates each against six classical
baselines: global FFT, global DCT, 8×8-block FFT, 8×8-block DCT, global
PCA, and 8×8-block PCA. Includes PCA + block-PCA in the trained run so
the writeup's headline numbers come from a single source.

MERA actually runs at this geometry (m+n=16 = 2^4, unlike QuickDraw
where m+n=10 silently skips MERA). Block bases (blocked, rich,
real_rich) train at m=8 with the symmetric `_blocked` split
(inner_m=4, block_log_m=4), giving 16×16 grids of 16×16 blocks on
256×256 images. Note: classical block_dct_8 uses 8×8 blocks — the
trained block bases (16×16) and classical block-DCT (8×8) are NOT at
the same block scale; this mirrors the QuickDraw pattern (4×4 trained
vs 8×8 classical).

GPU isolation: the --gpu flag sets CUDA_VISIBLE_DEVICES BEFORE
importing pdft_benchmarks (which transitively imports JAX). JAX
preallocates ~75% of every visible GPU on first call; without this
guard, two parallel invocations OOM.

Outputs land in `--out` directly. To use both GPUs:

    # Terminal A — GPU 0 — 4 unblocked bases
    python experiments/div2k_8q_pca_vs_block_dct.py \\
        --gpu 0 --bases qft,entangled_qft,tebd,mera \\
        --out results/div2k_8q_pca_vs_block_dct/_runs/unblocked

    # Terminal B — GPU 1 — 3 block-wrapped bases
    python experiments/div2k_8q_pca_vs_block_dct.py \\
        --gpu 1 --bases blocked,rich,real_rich \\
        --out results/div2k_8q_pca_vs_block_dct/_runs/blocked

After both finish, cellify with `tools/cellify_run.py` to assemble the
canonical `by_basis/<basis>/` tree.
"""

import argparse
import os
import sys

ALL_BASES = ("qft", "entangled_qft", "tebd", "mera",
             "blocked", "rich", "real_rich")
BASELINES = ("fft", "dct", "block_fft_8", "block_dct_8", "pca", "block_pca_8")


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
    parser.add_argument("--bases", default=",".join(ALL_BASES),
                        help=f"Comma-separated subset of {ALL_BASES}.")
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
    from pdft_benchmarks.pipeline import run_experiment

    res = run_experiment(
        dataset="div2k",
        m=8, n=8,
        bases=bases,
        baselines=list(BASELINES),
        preset=args.preset,
        output_dir=args.out,
        # When CUDA_VISIBLE_DEVICES already isolated GPU N, the JAX-visible
        # device is index 0 inside this process. Pass "auto" so JAX picks it.
        device="auto",
    )
    print(f"\nDone. Results: {res.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
