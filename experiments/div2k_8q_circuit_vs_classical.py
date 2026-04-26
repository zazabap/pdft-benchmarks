#!/usr/bin/env python3
"""Core experiment: 4 circuit bases vs DCT/FFT/blockDCT on DIV2K @ m=n=8.

Reproduces the canonical comparison:
- Bases:     qft, entangled_qft, tebd, mera     (m+n=16 is power of 2 → mera runs)
- Baselines: dct, fft, block_dct_8
- Preset:    generalized (500 train images, 60 epochs, batch 50, Adam)

Outputs to ./results/div2k_8q_<preset>_<timestamp>/.
"""

import argparse

from pdft_benchmarks.pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index (None = auto)")
    parser.add_argument("--out", default=None, help="output dir")
    args = parser.parse_args()

    res = run_experiment(
        dataset="div2k",
        m=8, n=8,
        bases=["qft", "entangled_qft", "tebd", "mera"],
        baselines=["fft", "dct", "block_dct_8"],
        preset=args.preset,
        output_dir=args.out,
        device=f"gpu:{args.gpu}" if args.gpu is not None else "auto",
    )
    print(f"\nDone. Results: {res.output_dir}")


if __name__ == "__main__":
    main()
