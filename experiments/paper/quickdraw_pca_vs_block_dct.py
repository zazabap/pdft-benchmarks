#!/usr/bin/env python3
"""QuickDraw PCA-vs-block-DCT benchmark (m=n=5, 32×32).

Trains the seven registered parametric bases (qft, entangled_qft, tebd,
mera, blocked, rich, real_rich) and evaluates each against the four
classical baselines: global FFT, global DCT, 8×8-block FFT, 8×8-block
DCT. PCA + block-DCT are the comparison anchors for the paper; the
parametric bases are the candidates being assessed against them.

Outputs land in `results/quickdraw_pca_vs_block_dct/` when run with the
default --out (None → uses the run_experiment default which writes
under results/<dataset>_<preset>_<timestamp>/; pass --out explicitly
to drop straight into the canonical paper directory).

`mera` is silently skipped by run_experiment because m+n=10 is not a
power of 2. The block bases (blocked, rich, real_rich) train at m=5
thanks to the asymmetric `_blocked` split (inner_m=3, block_log_m=2)
in pdft_benchmarks.bases — a 4×4 grid of 8×8 blocks fitting a 32×32
image.
"""

import argparse
import os
import sys


DEFAULT_BASES = ("qft", "entangled_qft", "tebd", "mera",
                 "blocked", "rich", "real_rich")
# Ablation variants — selectable via --bases but not in the default grid.
EXTRA_BASES = ("dct4_ctl", "tebd_u4", "rich_full", "real_rich_full")
ALL_BASES = DEFAULT_BASES + EXTRA_BASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--bases", default=",".join(DEFAULT_BASES),
                        help=f"Comma-separated subset of {DEFAULT_BASES}. "
                             f"Default: all.")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early-stopping-on-validation-plateau.")
    parser.add_argument("--keep-ratios", default=None,
                        help="Comma-separated keep ratios to evaluate at, "
                             "overriding preset.keep_ratios. The paper's "
                             "headline table also reports 0.01.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override preset.epochs.")
    args = parser.parse_args()

    # CRITICAL: set CUDA_VISIBLE_DEVICES BEFORE importing pdft_benchmarks
    # (which transitively imports JAX). Selecting the device afterwards via
    # jax.devices()[i] is not enough on a mixed-GPU host: JAX compiles for the
    # default device and then refuses to run the executable on a card of a
    # different type ("executable is built for device CUDA:0 of type ...").
    # The DIV2K driver has always done this; this one had not.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from pdft_benchmarks.experiment_utils import apply_preset_overrides
    from pdft_benchmarks.pipeline import run_experiment

    bases = [b.strip() for b in args.bases.split(",") if b.strip()]
    unknown = [b for b in bases if b not in ALL_BASES]
    if unknown:
        raise SystemExit(f"unknown basis name(s): {unknown}; "
                         f"choices: {sorted(ALL_BASES)}")

    preset = apply_preset_overrides(
        args.preset, dataset="quickdraw", tag="quickdraw",
        no_early_stop=args.no_early_stop, epochs=args.epochs,
        keep_ratios=(
            tuple(float(x) for x in args.keep_ratios.split(","))
            if args.keep_ratios else None
        ),
    )

    res = run_experiment(
        dataset="quickdraw",
        m=5, n=5,
        bases=bases,
        baselines=["fft", "dct", "block_fft_8", "block_dct_8"],
        preset=preset,
        output_dir=args.out,
        # CUDA_VISIBLE_DEVICES already isolated the card, so the JAX-visible
        # device is index 0 in this process; "auto" picks it.
        device="auto",
    )
    failed = {
        name: entry["failed"]
        for name, entry in res.metrics.items()
        if isinstance(entry, dict) and "failed" in entry
    }
    if failed:
        for name, info in failed.items():
            print(f"FAILED {name}: {info.get('phase')}: {info.get('error')}",
                  file=sys.stderr)
        print(f"\n{len(failed)}/{len(bases)} requested bases failed; "
              f"partial results in {res.output_dir}", file=sys.stderr)
        return 1

    print(f"\nDone. Results: {res.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
