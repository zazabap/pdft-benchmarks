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
