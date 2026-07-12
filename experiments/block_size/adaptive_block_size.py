#!/usr/bin/env python3
"""Per-image oracle adaptive block-size experiment.

Evaluates the HEVC-style content-adaptive selector at image granularity on
QuickDraw and DIV2K-8q across three candidate families:

  block_dct     HEVC-style classical (top-k DCT per-block)
  block_bd_pca  Dataset-fitted bilateral 2D-PCA per-block
  real_rich     Trained headline basis at each candidate block size

For each family the "adaptive" strategy picks, per test image, the block
size that minimises MSE (oracle).  The "fixed_best" reference is the single
block size that maximises mean PSNR across the full test set.

Usage
-----
    # QuickDraw (CPU-safe, 32×32)
    python experiments/block_size/adaptive_block_size.py --dataset quickdraw

    # DIV2K-8q (GPU recommended, 256×256)
    python experiments/block_size/adaptive_block_size.py --dataset div2k_8q --gpu 0

Outputs land in --out (default results/adaptive_block_size/<dataset>/per_image/)
as metrics.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "quickdraw": {
        "classical_b":  [2, 4, 8, 16],
        "trained_b":    [4, 8, 16],
        "img_size": 32,
        "m": 5, "n": 5,
        "cells_root": "results/block_size_sweep/quickdraw/by_basis",
    },
    "div2k_8q": {
        "classical_b":  [4, 8, 16, 32],
        "trained_b":    [4, 8, 16, 32],
        "img_size": 256,
        "m": 8, "n": 8,
        "cells_root": "results/block_size_sweep/div2k_8q/by_basis",
    },
}

KEEP_RATIO = 0.2  # headline ρ


# ---------------------------------------------------------------------------
# PSNR helper
# ---------------------------------------------------------------------------

def _psnr(original: np.ndarray, recovered: np.ndarray) -> float:
    rec = np.clip(np.real(recovered), 0.0, 1.0)
    mse = float(np.mean((original.astype(np.float64) - rec.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Per-family evaluation
# ---------------------------------------------------------------------------

def _evaluate_family(
    candidates: list[tuple[str, object]],
    test_images: np.ndarray,
    keep_ratio: float,
) -> dict:
    """Run the per-image oracle over all test images.

    Returns a dict with:
      fixed_best:  {"b": name, "psnr": float}
      adaptive:    {"psnr": float, "gain_over_fixed_best_db": float,
                   "chosen_per_image": [str, ...]}
      n_test:      int
      candidate_b: [names, ...]
    """
    from pdft_benchmarks.adaptive import adaptive_compress

    n_test = len(test_images)
    candidate_names = [name for name, _ in candidates]

    # Compute per-image PSNR for every candidate (needed for fixed_best).
    # Shape: (n_test, n_candidates)
    adaptive_psnrs: list[float] = []
    chosen_per_image: list[str] = []

    # We run adaptive_compress for the adaptive column, and
    # also collect the per-candidate PSNR matrix for fixed_best.
    psnr_matrix = np.full((n_test, len(candidates)), float("nan"))

    for i, img in enumerate(test_images):
        img_f64 = np.asarray(img, dtype=np.float64)

        # Per-candidate
        for j, (name, fn) in enumerate(candidates):
            try:
                recon = fn(img_f64, keep_ratio)
                psnr_matrix[i, j] = _psnr(img_f64, recon)
            except Exception as e:
                print(f"  [warn] candidate {name} failed on image {i}: {e}", file=sys.stderr)

        # Adaptive (oracle per image)
        try:
            _, info = adaptive_compress(img_f64, candidates, keep_ratio, granularity="image")
            chosen_name = info["chosen"]
            j_chosen = candidate_names.index(chosen_name)
            adaptive_psnrs.append(psnr_matrix[i, j_chosen])
            chosen_per_image.append(chosen_name)
        except Exception as e:
            print(f"  [warn] adaptive_compress failed on image {i}: {e}", file=sys.stderr)
            adaptive_psnrs.append(float("nan"))
            chosen_per_image.append("")

    # fixed_best: per-candidate mean PSNR across test set
    mean_psnr_per_cand = np.nanmean(psnr_matrix, axis=0)
    best_idx = int(np.nanargmax(mean_psnr_per_cand))
    fixed_best_name = candidate_names[best_idx]
    fixed_best_psnr = float(mean_psnr_per_cand[best_idx])

    adaptive_psnr = float(np.nanmean(adaptive_psnrs))
    gain = adaptive_psnr - fixed_best_psnr

    return {
        "candidate_b": candidate_names,
        "fixed_best": {"b": fixed_best_name, "psnr": round(fixed_best_psnr, 4)},
        "adaptive": {
            "psnr": round(adaptive_psnr, 4),
            "gain_over_fixed_best_db": round(gain, 4),
            "chosen_per_image": chosen_per_image,
        },
        "n_test": n_test,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["quickdraw", "div2k_8q"], required=True)
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU index. For DIV2K: sets CUDA_VISIBLE_DEVICES. "
                         "For QuickDraw: passed as JAX device hint.")
    ap.add_argument("--n-test",  type=int, default=50)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--out",     default=None,
                    help="Output directory. Default: results/adaptive_block_size/<dataset>/per_image/")
    args = ap.parse_args()

    # Set GPU BEFORE importing pdft_benchmarks (JAX preallocates on first use).
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = DATASET_CONFIG[args.dataset]
    out_dir = Path(args.out) if args.out else \
        Path(f"results/adaptive_block_size/{args.dataset}/per_image")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    from pdft_benchmarks.baselines import BASELINE_FACTORIES
    from pdft_benchmarks._loading import load_trained_basis, make_compress_fn

    if args.dataset == "quickdraw":
        from pdft_benchmarks.datasets import load_quickdraw
        print(f"[adaptive] loading QuickDraw n_train={args.n_train}, n_test={args.n_test}, "
              f"seed={args.seed}…")
        train_imgs, test_imgs = load_quickdraw(
            args.n_train, args.n_test, seed=args.seed, img_size=cfg["img_size"]
        )
    elif args.dataset == "div2k_8q":
        from pdft_benchmarks.datasets import load_div2k
        print(f"[adaptive] loading DIV2K-8q n_train={args.n_train}, n_test={args.n_test}, "
              f"seed={args.seed}…")
        train_imgs, test_imgs = load_div2k(
            args.n_train, args.n_test, seed=args.seed, size=cfg["img_size"]
        )
    else:
        raise ValueError(f"unknown dataset: {args.dataset!r}")

    print(f"[adaptive] train={train_imgs.shape}, test={test_imgs.shape}")

    classical_b = cfg["classical_b"]
    trained_b   = cfg["trained_b"]
    cells_root  = Path(cfg["cells_root"])

    # --- Build candidate families ---

    # Family 1: block_dct — stateless, build directly
    block_dct_candidates = [
        (f"block_dct_{b}", BASELINE_FACTORIES[f"block_dct_{b}"](list(train_imgs)))
        for b in classical_b
    ]
    print(f"[adaptive] built block_dct family: {[n for n, _ in block_dct_candidates]}")

    # Family 2: block_bd_pca — dataset-fitted, needs train_imgs
    block_bd_pca_candidates = []
    for b in classical_b:
        key = f"block_bd_pca_{b}"
        fn = BASELINE_FACTORIES[key](list(train_imgs))
        block_bd_pca_candidates.append((key, fn))
    print(f"[adaptive] built block_bd_pca family: {[n for n, _ in block_bd_pca_candidates]}")

    # Family 3: real_rich — trained, load from cells
    real_rich_candidates = []
    for b in trained_b:
        json_path = cells_root / f"real_rich_{b}" / f"trained_real_rich_{b}.json"
        if not json_path.exists():
            print(f"[adaptive] SKIP real_rich_{b}: {json_path} not found")
            continue
        try:
            basis = load_trained_basis(json_path)
            fn = make_compress_fn(basis)
            real_rich_candidates.append((f"real_rich_{b}", fn))
            print(f"[adaptive] loaded real_rich_{b}")
        except Exception as e:
            print(f"[adaptive] SKIP real_rich_{b}: failed to load: {e}", file=sys.stderr)

    if not real_rich_candidates:
        print("[adaptive] WARNING: no real_rich trained cells found; "
              "real_rich family will be empty.", file=sys.stderr)

    # --- Evaluate each family ---
    families = [
        ("block_dct",    block_dct_candidates),
        ("block_bd_pca", block_bd_pca_candidates),
        ("real_rich",    real_rich_candidates),
    ]

    results: dict = {}
    for family_name, candidates in families:
        if not candidates:
            print(f"[adaptive] skipping empty family '{family_name}'")
            continue
        print(f"[adaptive] evaluating '{family_name}' "
              f"({len(candidates)} candidates, n_test={args.n_test}, ρ={KEEP_RATIO})…")
        results[family_name] = _evaluate_family(candidates, test_imgs, KEEP_RATIO)

    results["_meta"] = {
        "dataset": args.dataset,
        "keep_ratio": KEEP_RATIO,
        "granularity": "image",
        "seed": args.seed,
        "n_train": args.n_train,
    }

    # --- Save ---
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[adaptive] wrote {out_path}")

    # --- Summary ---
    print("\n--- Per-family summary (ρ={:.2f}) ---".format(KEEP_RATIO))
    for family_name, _ in families:
        if family_name not in results:
            continue
        r = results[family_name]
        fb = r["fixed_best"]
        ad = r["adaptive"]
        print(
            f"  {family_name:20s}  "
            f"fixed_best={fb['b']:>18s} {fb['psnr']:6.2f} dB  |  "
            f"adaptive={ad['psnr']:6.2f} dB  "
            f"gain={ad['gain_over_fixed_best_db']:+.3f} dB"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
