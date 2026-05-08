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
from dataclasses import replace

from pdft_benchmarks.pipeline import run_experiment
from pdft_benchmarks.presets import get_preset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early-stopping-on-validation-plateau.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override preset.epochs.")
    args = parser.parse_args()

    preset = args.preset
    if args.no_early_stop or args.epochs is not None:
        base = get_preset("quickdraw", preset) if isinstance(preset, str) else preset
        kwargs = {}
        if args.no_early_stop:
            kwargs["early_stopping_patience"] = 10**9
        if args.epochs is not None:
            kwargs["epochs"] = args.epochs
        preset = replace(base, **kwargs)
        print(f"[quickdraw] preset overrides: epochs={preset.epochs}, "
              f"patience={preset.early_stopping_patience}")

    res = run_experiment(
        dataset="quickdraw",
        m=5, n=5,
        bases=["qft", "entangled_qft", "tebd", "mera",
               "blocked", "rich", "real_rich"],
        baselines=["fft", "dct", "block_fft_8", "block_dct_8"],
        preset=preset,
        output_dir=args.out,
        device=f"gpu:{args.gpu}" if args.gpu is not None else "auto",
    )
    print(f"\nDone. Results: {res.output_dir}")


if __name__ == "__main__":
    main()
