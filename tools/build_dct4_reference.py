#!/usr/bin/env python3
"""Build the DCT-IV study's reference JSON: the canonical (untrained) exact
DCT-IV init PSNR + the block-FFT 8x8 / block-DCT 8x8 classical references.

The canonical DCT-IV is `DCT4Basis(8,8)` at its exact init (deterministic),
evaluated on the fixed seed-42 DIV2K-8q test set at the keep ratios
(0.01/0.05/0.10/0.20) via `evaluate_basis_shared` (CPU — no GPU needed). The
block-FFT 8x8 / block-DCT 8x8 references are computed fresh at the same ratios
from `BASELINE_FACTORIES` (the QFT seed-study reference only stored
0.05/0.10/0.15/0.20, so it cannot supply rho=0.01). Writes
<out>/reference/classical_dct4.json.

Usage:
    python tools/build_dct4_reference.py \
        --out results/training/2_direct_training/random_seed/dct_div2k_8q
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]
QFT_CLASSICAL = REPO / "results/training/2_direct_training/random_seed/reference/classical_div2k.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="div2k_8q")
    args = ap.parse_args()

    import numpy as np  # noqa: F401
    import pdft
    import pdft.io  # noqa: F401
    from pdft_benchmarks.baselines import BASELINE_FACTORIES
    from pdft_benchmarks.datasets.div2k import load_div2k
    from pdft_benchmarks.evaluation import evaluate_baseline, evaluate_basis_shared
    from pdft_benchmarks.presets import get_preset

    if not hasattr(pdft, "DCT4Basis"):
        print(f"[ref] FATAL: pdft at {pdft.__file__} lacks DCT4Basis; put latest "
              f"pdft src on PYTHONPATH.", file=sys.stderr)
        return 3

    m = n = 8
    preset = get_preset(args.dataset, "generalized")
    train_imgs, test_imgs = load_div2k(n_train=preset.n_train, n_test=preset.n_test,
                                       seed=42, size=2 ** m)
    keep_ratios = (0.01, 0.05, 0.10, 0.20)

    basis = pdft.DCT4Basis(m=m, n=n)   # exact DCT-IV (deterministic)
    metrics, _ = evaluate_basis_shared(basis, test_imgs, keep_ratios=keep_ratios)
    canon = {f"{r}": round(float(metrics[str(r)]["mean_psnr"]), 3) for r in keep_ratios}
    print(f"[ref] canonical DCT-IV PSNR: {canon}")

    ref = {"canonical_dct4": {"label": "DCT-IV (untrained)", "psnr": canon}}
    # Block references computed fresh at the chosen ratios (the QFT seed-study
    # reference only stored 0.05/0.10/0.15/0.20, so it cannot supply rho=0.01).
    block_labels = {"block_fft_8": "block-FFT 8×8", "block_dct_8": "block-DCT 8×8"}
    for name, label in block_labels.items():
        fn = BASELINE_FACTORIES[name](list(train_imgs))
        kr_metrics, _ = evaluate_baseline(fn, test_imgs, keep_ratios)
        psnr = {f"{r}": round(float(kr_metrics[str(r)]["mean_psnr"]), 3) for r in keep_ratios}
        ref[name] = {"label": label, "psnr": psnr}
        print(f"[ref] {name} PSNR: {psnr}")

    out_dir = Path(args.out) / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classical_dct4.json").write_text(json.dumps(ref, indent=2))
    print(f"[ref] wrote {out_dir/'classical_dct4.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
