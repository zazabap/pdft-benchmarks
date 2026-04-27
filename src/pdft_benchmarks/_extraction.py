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


import json
import shutil
from pathlib import Path


def extract_cell(
    *,
    source_run: Path,
    cell_dir: Path,
    source_basis_key: str,
    m: int,
    n: int,
) -> None:
    """Populate `cell_dir` with all canonical-cell files from `source_run`.

    Calls pdft_benchmarks._report.main(cell_dir) at the end to regenerate
    per-cell CSVs and plots from the filtered metrics.json.
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "loss_history").mkdir(exist_ok=True)

    src_metrics = json.loads((source_run / "metrics.json").read_text())
    cell_metrics = filter_metrics_for_cell(src_metrics, source_basis_key=source_basis_key)
    (cell_dir / "metrics.json").write_text(json.dumps(cell_metrics, indent=2))

    shutil.copyfile(source_run / "env.json", cell_dir / "env.json")
    src_env = json.loads((source_run / "env.json").read_text())

    dest_basis = rename_basis_key(source_basis_key)
    cfg = build_config_json(src_env, m=m, n=n, basis=dest_basis)
    (cell_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    src_trained = source_run / f"trained_{source_basis_key}.json"
    if src_trained.is_file():
        shutil.copyfile(src_trained, cell_dir / f"trained_{dest_basis}.json")

    src_loss = source_run / "loss_history" / f"{source_basis_key}_loss.json"
    if src_loss.is_file():
        shutil.copyfile(src_loss, cell_dir / "loss_history" / f"{dest_basis}_loss.json")

    src_log = source_run / "run.log"
    if src_log.is_file():
        shutil.copyfile(src_log, cell_dir / "run.log")

    from pdft_benchmarks._report import main as report_main
    report_main(cell_dir)


def write_skipped_cell(
    cell_dir: Path,
    *,
    m: int,
    n: int,
    basis: str,
    reason: str = "incompatible_qubits",
    constraint: str = "m+n must be a power of 2",
) -> None:
    """Write only SKIPPED.json for a cell that cannot be trained.

    Two reasons are supported by the schema:
      - "incompatible_qubits": MERA at non-power-of-2 m+n.
      - "block_factory_odd_m_unsupported": _blocked / rich / real_rich at odd outer m
        (the registry's `_blocked` helper drops a qubit when m is odd).
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "SKIPPED.json").write_text(json.dumps({
        "reason": reason,
        "m": m, "n": n,
        "basis": basis,
        "constraint": constraint,
    }, indent=2))
