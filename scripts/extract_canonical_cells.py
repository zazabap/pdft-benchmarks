#!/usr/bin/env python3
"""Populate results/published/<dataset>__<basis>/ from existing source runs.

The EXTRACTION_TABLE below is the canonical mapping of (cell_id) →
(source_run_dir, source_basis_key). The 9 freshly-trained cells (3 on
div2k_10q + 6 on quickdraw) are added by editing the table after those
runs land in results/_archive/ (or wherever the runner deposits them).

Usage:
    python scripts/extract_canonical_cells.py
    python scripts/extract_canonical_cells.py --results-root results --published-root results/published
    python scripts/extract_canonical_cells.py --only div2k_8q
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdft_benchmarks._extraction import extract_cell, write_skipped_cell

# (m, n) per dataset for synthesizing config.json.
DATASET_MN = {"div2k_8q": (8, 8), "div2k_10q": (10, 10), "quickdraw": (5, 5)}

# Source-run paths are relative to --results-root (default: "results/").
EXTRACTION_TABLE: list[dict] = [
    # ===== 8q DIV2K (already trained) =====
    {"cell_id": "div2k_8q__qft",
     "source_run": "_archive/div2k_8q_generalized_20260425-102013_gpu0",
     "source_basis_key": "qft", "dataset": "div2k_8q"},
    {"cell_id": "div2k_8q__mera",
     "source_run": "_archive/div2k_8q_generalized_20260425-102013_gpu0",
     "source_basis_key": "mera", "dataset": "div2k_8q"},
    {"cell_id": "div2k_8q__entangled_qft",
     "source_run": "_archive/div2k_8q_generalized_20260425-102013_gpu1",
     "source_basis_key": "entangled_qft", "dataset": "div2k_8q"},
    {"cell_id": "div2k_8q__tebd",
     "source_run": "_archive/div2k_8q_generalized_20260425-102013_gpu1",
     "source_basis_key": "tebd", "dataset": "div2k_8q"},
    {"cell_id": "div2k_8q__blocked",
     "source_run": "_archive/div2k_8q_blocked_generalized_20260426-085726",
     "source_basis_key": "blocked_qft", "dataset": "div2k_8q"},
    {"cell_id": "div2k_8q__rich",
     "source_run": "_archive/div2k_8q_blocked_rich_generalized_20260426-110840",
     "source_basis_key": "blocked_rich", "dataset": "div2k_8q"},
    {"cell_id": "div2k_8q__real_rich",
     "source_run": "_archive/div2k_8q_REAL_20260426-123029",
     "source_basis_key": "blocked_real", "dataset": "div2k_8q"},

    # ===== 10q DIV2K (3 already trained, 3 from new run) =====
    {"cell_id": "div2k_10q__qft",
     "source_run": "_archive/div2k_10q_generalized_20260426-055335_gpu1_bs2",
     "source_basis_key": "qft", "dataset": "div2k_10q"},
    {"cell_id": "div2k_10q__entangled_qft",
     "source_run": "_archive/div2k_10q_generalized_20260426-055335_gpu0_bs2",
     "source_basis_key": "entangled_qft", "dataset": "div2k_10q"},
    {"cell_id": "div2k_10q__tebd",
     "source_run": "_archive/div2k_10q_generalized_20260426-055335_gpu1_bs2",
     "source_basis_key": "tebd", "dataset": "div2k_10q"},
    # NEW (filled after running experiments/div2k_10q_block.py).
    # The runner writes to results/div2k_10q_<m>q_generalized_<ts>/;
    # after running it, move that dir into _archive/ and update the
    # source_run path here to the actual timestamp.
    {"cell_id": "div2k_10q__blocked",
     "source_run": "_archive/div2k_10q_blocked_generalized_NEW",
     "source_basis_key": "blocked_qft", "dataset": "div2k_10q"},
    {"cell_id": "div2k_10q__rich",
     "source_run": "_archive/div2k_10q_blocked_generalized_NEW",
     "source_basis_key": "blocked_rich", "dataset": "div2k_10q"},
    {"cell_id": "div2k_10q__real_rich",
     "source_run": "_archive/div2k_10q_blocked_generalized_NEW",
     "source_basis_key": "blocked_real", "dataset": "div2k_10q"},

    # ===== QuickDraw circuit bases (3 from the 2026-04-27-060341 run) =====
    {"cell_id": "quickdraw__qft",
     "source_run": "_archive/quickdraw_5q_generalized_20260427-060341",
     "source_basis_key": "qft", "dataset": "quickdraw"},
    {"cell_id": "quickdraw__entangled_qft",
     "source_run": "_archive/quickdraw_5q_generalized_20260427-060341",
     "source_basis_key": "entangled_qft", "dataset": "quickdraw"},
    {"cell_id": "quickdraw__tebd",
     "source_run": "_archive/quickdraw_5q_generalized_20260427-060341",
     "source_basis_key": "tebd", "dataset": "quickdraw"},
    # NOTE: quickdraw block bases (blocked, rich, real_rich) are SKIPPED at
    # m=n=5 — see SKIPPED_CELLS below for the reason.
]

SKIPPED_CELLS: list[dict] = [
    # MERA at m+n not a power of 2.
    {"cell_id": "div2k_10q__mera", "dataset": "div2k_10q", "basis": "mera",
     "reason": "incompatible_qubits",
     "constraint": "m+n must be a power of 2"},
    {"cell_id": "quickdraw__mera", "dataset": "quickdraw", "basis": "mera",
     "reason": "incompatible_qubits",
     "constraint": "m+n must be a power of 2"},
    # Block bases at odd outer m: the registry's _blocked helper does
    # `m // 2` for both inner_m and block_log_m, dropping a qubit at odd m.
    {"cell_id": "quickdraw__blocked", "dataset": "quickdraw", "basis": "blocked",
     "reason": "block_factory_odd_m_unsupported",
     "constraint": "outer m must be even (registry _blocked uses m//2 for both halves)"},
    {"cell_id": "quickdraw__rich", "dataset": "quickdraw", "basis": "rich",
     "reason": "block_factory_odd_m_unsupported",
     "constraint": "outer m must be even (registry _blocked uses m//2 for both halves)"},
    {"cell_id": "quickdraw__real_rich", "dataset": "quickdraw", "basis": "real_rich",
     "reason": "block_factory_odd_m_unsupported",
     "constraint": "outer m must be even (registry _blocked uses m//2 for both halves)"},
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", type=Path)
    parser.add_argument("--published-root", default="results/published", type=Path)
    parser.add_argument("--only",
                        help="comma-separated cell_id substring filter",
                        default="")
    args = parser.parse_args(argv)

    args.published_root.mkdir(parents=True, exist_ok=True)

    only = [s for s in args.only.split(",") if s]

    for entry in EXTRACTION_TABLE:
        if only and not any(s in entry["cell_id"] for s in only):
            continue
        ds = entry["dataset"]
        m, n = DATASET_MN[ds]
        src = args.results_root / entry["source_run"]
        dest = args.published_root / entry["cell_id"]
        if not src.is_dir():
            print(f"SKIP {entry['cell_id']} — source missing: {src}", file=sys.stderr)
            continue
        print(f"extract {entry['cell_id']}  ←  {src}")
        extract_cell(
            source_run=src, cell_dir=dest,
            source_basis_key=entry["source_basis_key"],
            m=m, n=n,
        )

    for entry in SKIPPED_CELLS:
        if only and not any(s in entry["cell_id"] for s in only):
            continue
        ds = entry["dataset"]
        m, n = DATASET_MN[ds]
        dest = args.published_root / entry["cell_id"]
        print(f"skipped {entry['cell_id']} ({entry['reason']})")
        write_skipped_cell(
            dest, m=m, n=n, basis=entry["basis"],
            reason=entry["reason"], constraint=entry["constraint"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
