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


def build_config_json(env: dict, *, m: int, n: int, basis: str) -> dict:
    """Synthesize a flat config.json dict for one published cell.

    Pulls all training hyperparameters from env['preset_dataclass'] (the
    pipeline always writes this — see pipeline.py). Adds {m, n, basis}
    so the cell is self-describing without env.json.
    """
    if "preset_dataclass" not in env:
        raise KeyError("env missing required field 'preset_dataclass'")
    pd = env["preset_dataclass"]
    return {
        "m": m,
        "n": n,
        "basis": basis,
        "preset": env.get("preset", "unknown"),
        "epochs": pd["epochs"],
        "n_train": pd["n_train"],
        "n_test": pd["n_test"],
        "optimizer": pd["optimizer"],
        "batch_size": pd["batch_size"],
        "warmup_frac": pd["warmup_frac"],
        "lr_peak": pd["lr_peak"],
        "lr_final": pd["lr_final"],
        "max_grad_norm": pd["max_grad_norm"],
        "validation_split": pd["validation_split"],
        "early_stopping_patience": pd["early_stopping_patience"],
        "seed": pd["seed"],
        "keep_ratios": list(pd["keep_ratios"]),
    }
