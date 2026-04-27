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
