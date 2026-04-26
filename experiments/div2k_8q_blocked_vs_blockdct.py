#!/usr/bin/env python3
"""3 blocked bases vs blockDCT on DIV2K @ m=n=8.

- Bases:     blocked, rich, real_rich  (each wraps its inner topology in BlockedBasis)
- Baselines: block_dct_8                (the natural classical comparator)
- Preset:    generalized (500 train images)
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
        dataset="div2k",
        m=8, n=8,
        bases=["blocked", "rich", "real_rich"],
        baselines=["block_dct_8"],
        preset=args.preset,
        output_dir=args.out,
        device=f"gpu:{args.gpu}" if args.gpu is not None else "auto",
    )
    print(f"\nDone. Results: {res.output_dir}")


if __name__ == "__main__":
    main()
