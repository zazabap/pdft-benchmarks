#!/usr/bin/env python3
"""QuickDraw benchmark (m=n=5, 32×32). Mirrors the original run_quickdraw.py."""

import argparse

from pdft_benchmarks.pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="moderate",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    res = run_experiment(
        dataset="quickdraw",
        m=5, n=5,
        bases=["qft", "entangled_qft", "tebd"],            # mera skipped at m+n=10
        baselines=["fft", "dct"],
        preset=args.preset,
        output_dir=args.out,
        device=f"gpu:{args.gpu}" if args.gpu is not None else "auto",
    )
    print(f"\nDone. Results: {res.output_dir}")


if __name__ == "__main__":
    main()
