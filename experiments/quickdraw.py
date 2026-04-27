#!/usr/bin/env python3
"""QuickDraw benchmark (m=n=5, 32×32) — circuit bases at `generalized`.

`mera` is silently skipped by run_experiment because m+n=10 is not a
power of 2. The block bases (`blocked`, `rich`, `real_rich`) are NOT
trained here: the registry's `_blocked` helper does `m // 2` for both
inner_m and block_log_m, which loses a qubit at odd outer m=5
(yielding a basis at m_outer=4, expecting 16×16 input). For the
benchmark matrix these cells are marked SKIPPED with reason
`block_factory_odd_m_unsupported`.
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
        bases=["qft", "entangled_qft", "tebd", "mera"],
        baselines=["fft", "dct", "block_fft_8", "block_dct_8"],
        preset=args.preset,
        output_dir=args.out,
        device=f"gpu:{args.gpu}" if args.gpu is not None else "auto",
    )
    print(f"\nDone. Results: {res.output_dir}")


if __name__ == "__main__":
    main()
