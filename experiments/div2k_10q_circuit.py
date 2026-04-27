#!/usr/bin/env python3
"""DIV2K-10q (m=n=10, 1024×1024) — kept variant.

Same shape as the 8q circuit experiment but at m=n=10. MERA is silently
skipped (m+n=20 is not a power of 2). Uses the DIV2K-10q-specific preset
(batch_size=4, val_every_k_epochs=2) due to memory constraints.
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
        m=10, n=10,
        bases=["qft", "entangled_qft", "tebd"],            # mera skipped at m+n=20
        baselines=["fft", "dct", "block_dct_8"],
        preset=args.preset,
        output_dir=args.out,
        device=f"gpu:{args.gpu}" if args.gpu is not None else "auto",
        dataset_kwargs={"size": 1024},  # match m=n=10 → 2^10 = 1024
    )
    print(f"\nDone. Results: {res.output_dir}")


if __name__ == "__main__":
    main()
