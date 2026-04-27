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
