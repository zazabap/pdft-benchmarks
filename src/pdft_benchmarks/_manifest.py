"""MANIFEST.json schema and builder/validator for results/published/.

The MANIFEST is the on-disk source of truth for which (dataset, basis)
cells are 'published' (vs. ablations, vs. archived). The builder reads
each cell directory and emits the MANIFEST; the validator checks that
on-disk state matches what MANIFEST claims.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0"

DATASETS: dict[str, dict[str, Any]] = {
    "div2k_8q":  {"m": 8,  "n": 8,  "image_size": [256, 256],
                  "n_train": 500, "n_test": 100},
    "div2k_10q": {"m": 10, "n": 10, "image_size": [1024, 1024],
                  "n_train": 500, "n_test": 50},
    "quickdraw": {"m": 5,  "n": 5,  "image_size": [32, 32],
                  "n_train": 500, "n_test": 100},
}

BASES: dict[str, dict[str, str]] = {
    "qft":           {"family": "circuit", "factory": "pdft.QFTBasis"},
    "entangled_qft": {"family": "circuit", "factory": "pdft.EntangledQFTBasis"},
    "tebd":          {"family": "circuit", "factory": "pdft.TEBDBasis"},
    "mera":          {"family": "circuit", "factory": "pdft.MERABasis",
                      "constraint": "m+n must be power of 2"},
    "blocked":       {"family": "block",
                      "factory": "pdft.BlockedBasis(inner=QFTBasis, m_inner=m/2, block_log=m/2)"},
    "rich":          {"family": "block",
                      "factory": "pdft.BlockedBasis(inner=RichBasis, m_inner=m/2, block_log=m/2)"},
    "real_rich":     {"family": "block",
                      "factory": "pdft.BlockedBasis(inner=RealRichBasis, m_inner=m/2, block_log=m/2)"},
}

CLASSICAL_BASELINES = ["fft", "dct", "block_fft_8", "block_dct_8"]

# (dataset, "mera") cells are SKIPPED whenever m+n is not a power of 2.
MERA_INCOMPATIBLE_DATASETS = {
    name for name, row in DATASETS.items()
    if not (((row["m"] + row["n"]) & (row["m"] + row["n"] - 1)) == 0
            and (row["m"] + row["n"]) > 0)
}


import math


KEEP_RATIOS_REPORTED = ("0.05", "0.1", "0.15", "0.2")


def summarize_metrics(cell_metrics: dict, *, basis_key: str) -> dict:
    """Build the `metrics_summary` block for one MANIFEST cell entry.

    Reports PSNR at each of the standard keep ratios + train_time_s.
    Missing keep ratios produce NaN (not an error: some ablations don't
    cover all keep ratios).
    """
    if basis_key not in cell_metrics:
        raise KeyError(f"basis {basis_key!r} not in metrics")
    block = cell_metrics[basis_key]
    summary: dict = {}
    metrics_by_kr = block.get("metrics", {})
    for kr in KEEP_RATIOS_REPORTED:
        if kr in metrics_by_kr:
            summary[f"psnr_at_keep_{kr}"] = float(metrics_by_kr[kr]["mean_psnr"])
        else:
            summary[f"psnr_at_keep_{kr}"] = math.nan
    summary["train_time_s"] = float(block.get("time", math.nan))
    return summary


import json
from datetime import datetime, timezone
from pathlib import Path


def build_manifest(
    published_root: Path,
    *,
    git_sha: str,
    pdft_version: str,
    generated_at: str | None = None,
) -> dict:
    """Walk results/published/<dataset>__<basis>/ and produce a full MANIFEST dict."""
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat()

    cells: list[dict] = []
    for dataset in DATASETS:
        for basis in BASES:
            cell_id = f"{dataset}__{basis}"
            cell_dir = published_root / cell_id
            if not cell_dir.is_dir():
                # Tolerate missing cells during incremental builds; the
                # validator is the strict checker.
                continue
            skipped_path = cell_dir / "SKIPPED.json"
            if skipped_path.is_file():
                payload = json.loads(skipped_path.read_text())
                cells.append({
                    "id": cell_id,
                    "dataset": dataset,
                    "basis": basis,
                    "status": "skipped",
                    "path": f"{cell_id}/",
                    "skip_reason": (
                        f"{payload['reason']}: m+n={payload['m']+payload['n']} "
                        f"is not a power of 2"
                    ),
                })
                continue
            metrics = json.loads((cell_dir / "metrics.json").read_text())
            config = json.loads((cell_dir / "config.json").read_text())
            cells.append({
                "id": cell_id,
                "dataset": dataset,
                "basis": basis,
                "status": "active",
                "path": f"{cell_id}/",
                "preset": config.get("preset", "unknown"),
                "config": {
                    "epochs":     config["epochs"],
                    "n_train":    config["n_train"],
                    "n_test":     config["n_test"],
                    "lr_peak":    config["lr_peak"],
                    "batch_size": config["batch_size"],
                    "seed":       config["seed"],
                },
                "metrics_summary": summarize_metrics(metrics, basis_key=basis),
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "git_sha": git_sha,
        "pdft_version": pdft_version,
        "datasets": DATASETS,
        "bases": BASES,
        "classical_baselines": CLASSICAL_BASELINES,
        "cells": cells,
    }


class ManifestValidationError(Exception):
    """Raised when on-disk results/published/ disagrees with MANIFEST.json."""


_REQUIRED_ACTIVE_FILES = (
    "metrics.json", "env.json", "config.json",
    "rate_distortion_mse.csv",
    "rate_distortion_psnr.csv",
    "rate_distortion_ssim.csv",
    "timing_summary.csv",
)


def _summary_close(a: float, b: float, *, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def validate_manifest(published_root: Path) -> None:
    """Validate `published_root/MANIFEST.json` against on-disk cell tree.

    Raises ManifestValidationError on any discrepancy. Returns None on success.
    """
    manifest_path = published_root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ManifestValidationError(f"MANIFEST.json not found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    for cell in manifest["cells"]:
        cell_dir = published_root / cell["path"].rstrip("/")
        if not cell_dir.is_dir():
            raise ManifestValidationError(
                f"cell directory missing: {cell_dir} (claimed by MANIFEST id={cell['id']})"
            )
        if cell["status"] == "skipped":
            files = sorted(p.name for p in cell_dir.iterdir())
            if files != ["SKIPPED.json"]:
                raise ManifestValidationError(
                    f"extra files in skipped cell {cell['id']}: {files}"
                )
            continue
        for fname in _REQUIRED_ACTIVE_FILES:
            if not (cell_dir / fname).is_file():
                raise ManifestValidationError(
                    f"missing required file in {cell['id']}: {fname}"
                )
        on_disk_metrics = json.loads((cell_dir / "metrics.json").read_text())
        recomputed = summarize_metrics(on_disk_metrics, basis_key=cell["basis"])
        for k, v in cell["metrics_summary"].items():
            if not _summary_close(float(v), float(recomputed[k])):
                raise ManifestValidationError(
                    f"metrics_summary mismatch in {cell['id']}.{k}: "
                    f"manifest={v} on-disk={recomputed[k]}"
                )
