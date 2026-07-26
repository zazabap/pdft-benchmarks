#!/usr/bin/env python3
"""Re-evaluate the stored headline bases at top-k on the CURRENT data snapshot.

Evaluation only -- no training. Loads each committed trained basis (the
by_basis cells, plus the dataset-compression checkpoints for the real rich
variants) and scores mean test PSNR/SSIM with the exact pipeline protocol
(pdft.io.compress/recover: complex top-k counted once, keep = round(d*kr),
reconstruction = real(T^-1), clamped to [0, 1]) at an extended keep-ratio
grid that adds rho = 0.4 to the published ratios.

Motivation (see quickdraw_5q/checkpoints/GATE_NOTE.md): the QuickDraw .npy
snapshot on this machine differs from the one the original committed cells
were trained/evaluated on, so a paper table that adds new entries (the
rho = 0.4 column, the real rich row) must re-evaluate EVERY row on the
current snapshot to stay internally consistent. DIV2K is stable and doubles
as the protocol check: its re-evaluated values must reproduce the committed
cell metrics at the published ratios.

Writes results/structure/<experiment>/independent_reruns/topk_reeval.json
with per-basis {ratio: {mean_psnr, std_psnr, mean_ssim, ...}} plus the
drift vs each cell's committed metrics at the shared ratios.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RATIOS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.4]

DATASETS = {
    "quickdraw": {
        "loader": "quickdraw",
        "preset_ns": "quickdraw",
        "experiment": "quickdraw_pca_vs_block_dct",
        # by_basis cell name -> cell dir name (trained_<name>.json inside)
        "cells": ["qft", "entangled_qft", "rich_full", "tebd_u4", "dct4_ctl"],
        # the real rich variant: the dataset-compression contender checkpoint
        "extra": {"real_rich": "results/training/6_dataset_compression/"
                               "quickdraw_5q/checkpoints/trained_real_rich.json"},
        "baselines": ["block_dct_8", "block_fft_8"],
    },
    "div2k_8q": {
        "loader": "div2k",
        "preset_ns": "div2k_8q",
        "experiment": "div2k_8q_pca_vs_block_dct",
        "cells": ["qft", "entangled_qft", "rich_full", "tebd_u4", "mera_u4",
                  "dct4_ctl"],
        "extra": {"real_rich_8": "results/training/6_dataset_compression/"
                                 "div2k_8q/checkpoints/trained_real_rich_8.json"},
        "baselines": ["block_dct_8", "block_fft_8"],
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--ratios", default=",".join(str(r) for r in RATIOS))
    args = ap.parse_args()

    import numpy as np  # noqa: F401

    from pdft_benchmarks._loading import load_trained_basis
    from pdft_benchmarks.baselines import BASELINE_FACTORIES
    from pdft_benchmarks.datasets import load
    from pdft_benchmarks.evaluation import evaluate_baseline, evaluate_basis_shared
    from pdft_benchmarks.presets import get_preset

    cfg = DATASETS[args.dataset]
    ratios = [float(x) for x in args.ratios.split(",")]
    preset = get_preset(cfg["preset_ns"], "generalized")
    train_imgs, test_imgs = load(cfg["loader"], n_train=preset.n_train,
                                 n_test=preset.n_test, seed=preset.seed)

    by_basis = REPO_ROOT / "results/structure" / cfg["experiment"] / "by_basis"
    out_dir = REPO_ROOT / "results/structure" / cfg["experiment"] / "independent_reruns"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {"dataset": args.dataset, "ratios": ratios, "n_test": len(test_imgs),
              "protocol": "pdft.io.compress/recover via evaluate_basis_shared; "
                          "baselines via BASELINE_FACTORIES/evaluate_baseline",
              "by_basis": {}, "drift_vs_cell_dB": {}}

    sources = {name: by_basis / name / f"trained_{name}.json" for name in cfg["cells"]}
    sources.update({name: REPO_ROOT / rel for name, rel in cfg["extra"].items()})

    for name, path in sources.items():
        if not path.exists():
            print(f"[reeval] {name}: MISSING {path}", flush=True)
            continue
        basis = load_trained_basis(path)
        metrics, nan_counts = evaluate_basis_shared(basis, test_imgs, ratios)
        if any(nan_counts.values()):
            print(f"[reeval] WARN {name}: nan_counts={nan_counts}", flush=True)
        result["by_basis"][name] = metrics
        row = "  ".join(f"{metrics[str(r)]['mean_psnr']:6.2f}" for r in ratios)
        print(f"[reeval] {name:16s} {row}", flush=True)
        # drift vs the cell's committed metrics at shared ratios
        cell_metrics_path = by_basis / name / "metrics.json"
        if cell_metrics_path.exists():
            cell = json.loads(cell_metrics_path.read_text())
            committed = cell.get(name, {}).get("metrics", {})
            drift = {r: round(result["by_basis"][name][r]["mean_psnr"]
                              - committed[r]["mean_psnr"], 3)
                     for r in committed if r in result["by_basis"][name]}
            result["drift_vs_cell_dB"][name] = drift
            print(f"[reeval]   drift vs cell (dB): {drift}", flush=True)

    for base in cfg["baselines"]:
        fn = BASELINE_FACTORIES[base](train_imgs)
        metrics, _ = evaluate_baseline(fn, test_imgs, ratios)
        result["by_basis"][base] = metrics
        row = "  ".join(f"{metrics[str(r)]['mean_psnr']:6.2f}" for r in ratios)
        print(f"[reeval] {base:16s} {row}", flush=True)

    out = out_dir / "topk_reeval.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"[reeval] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
