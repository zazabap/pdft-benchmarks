#!/usr/bin/env python3
"""QuickDraw benchmark (m=n=5, 32×32) — all 7 registered bases at `generalized`.

`mera` is silently skipped by run_experiment because m+n=10 is not a
power of 2. The block bases (`blocked`, `rich`, `real_rich`) train at
m=5 thanks to the asymmetric `_blocked` split (inner_m=3, block_log_m=2)
in `pdft_benchmarks.bases` — a 4×4 grid of 8×8 blocks fitting a 32×32 image.
"""

import argparse

from pdft_benchmarks.pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    res = run_experiment(
        dataset="quickdraw",
        m=5, n=5,
        bases=["qft", "entangled_qft", "tebd", "mera",
               "blocked", "rich", "real_rich"],
        baselines=["fft", "dct", "block_fft_8", "block_dct_8"],
        preset=args.preset,
        output_dir=args.out,
        device=f"gpu:{args.gpu}" if args.gpu is not None else "auto",
    )
    print(f"\nDone. Results: {res.output_dir}")


if __name__ == "__main__":
    main()
