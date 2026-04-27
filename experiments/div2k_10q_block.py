#!/usr/bin/env python3
"""DIV2K-10q (m=n=10, 1024×1024) — block bases.

Sibling of div2k_10q_circuit.py. Trains the three block bases registered
in pdft_benchmarks.bases.BASIS_FACTORIES — `blocked`, `rich`, `real_rich`
— under the `generalized` preset overridden to batch_size=2 (matches the
existing canonical 10q circuit runs).

MERA is not trained here (the circuit script doesn't either; m+n=20 is
not a power of 2).
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from pdft_benchmarks.pipeline import run_experiment
from pdft_benchmarks.presets import get_preset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base_preset = get_preset("div2k_10q", "generalized")
    # Match the existing canonical 10q runs: bs=2, lr_peak=0.003.
    preset = replace(base_preset, batch_size=2, lr_peak=0.003)

    res = run_experiment(
        dataset="div2k",
        m=10, n=10,
        bases=["blocked", "rich", "real_rich"],
        baselines=["fft", "dct", "block_fft_8", "block_dct_8"],
        preset=preset,
        output_dir=args.out,
        device=f"gpu:{args.gpu}" if args.gpu is not None else "auto",
    )
    print(f"\nDone. Results: {res.output_dir}")


if __name__ == "__main__":
    main()
