"""Pure-Python extraction core: split multi-basis source runs into per-basis cells.

The extraction table (driven by scripts/extract_canonical_cells.py) is the
canonical mapping from *(cell_id, source_run, source_basis_key)* to a
single-basis flat cell directory under results/published/. Importing modules
drive filesystem layout from here; this module stays IO-light so it can be
tested without disk fixtures.
"""

from __future__ import annotations

# Source-run basis-name flavors → registry keys.
# Identity rows are spelled out for clarity.
SOURCE_TO_REGISTRY: dict[str, str] = {
    "qft": "qft",
    "entangled_qft": "entangled_qft",
    "tebd": "tebd",
    "mera": "mera",
    "blocked_qft": "blocked",
    "blocked_rich": "rich",
    "blocked_real": "real_rich",
}


def rename_basis_key(source_key: str) -> str:
    """Map a source-run basis name to the registry key used in published cells.

    Raises KeyError on unknown inputs (intentional: unknown keys mean the
    extraction table is out of date with the basis registry).
    """
    if source_key not in SOURCE_TO_REGISTRY:
        raise KeyError(f"unknown source basis key {source_key!r}")
    return SOURCE_TO_REGISTRY[source_key]


CLASSICAL_BASELINES: tuple[str, ...] = ("fft", "dct", "block_fft_8", "block_dct_8")


def filter_metrics_for_cell(
    source_metrics: dict, *, source_basis_key: str
) -> dict:
    """Return a single-basis metrics payload for one published cell.

    The returned dict has the renamed basis + the four classical baselines.
    Missing baselines are silently skipped (some older source runs only
    have fft/dct).
    """
    if source_basis_key not in source_metrics:
        raise KeyError(f"basis {source_basis_key!r} not in source metrics")
    dest_key = rename_basis_key(source_basis_key)
    out: dict = {dest_key: source_metrics[source_basis_key]}
    for baseline in CLASSICAL_BASELINES:
        if baseline in source_metrics:
            out[baseline] = source_metrics[baseline]
    return out
