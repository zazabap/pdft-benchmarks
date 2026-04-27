# Publishable Benchmark Results — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the existing `pdft-benchmarks` results into a publication-ready `results/published/` tree containing the canonical 7 bases × 3 datasets matrix (19 active cells + 2 SKIPPED cells), with paper-worthy ablations under `results/ablations/` and full provenance preserved in `results/_archive/`.

**Architecture:** A small set of pure-Python tooling scripts under `scripts/` (extraction, validation, README rendering) operate on the existing run-dir contract (`metrics.json`, `loss_history/`, `trained_*.json`, `env.json`, `run.log`). The 9 missing trainings (3 on DIV2K-10q + 6 on QuickDraw) are produced by two thin runner scripts under `experiments/`. The validator + manifest are the hard contract between disk state and what is "published"; everything else is convenience around those.

**Tech Stack:** Python 3.12, JAX (via `pdft==0.2.1`), NumPy, scipy, matplotlib (PDF backend), pytest. `pdft_benchmarks.run_experiment()` is the only training entrypoint; `pdft_benchmarks._report.main()` is the only CSV/plot regenerator. No new runtime deps.

**Spec:** `docs/superpowers/specs/2026-04-27-publishable-benchmark-results-design.md` (commit `e1ddc68`).

---

## File Structure

### Created

| Path | Responsibility |
|------|----------------|
| `experiments/div2k_10q_block.py` | Thin runner: trains the 3 block bases (`blocked`, `rich`, `real_rich`) at m=n=10, batch_size=2, lr=0.003. |
| `scripts/extract_canonical_cells.py` | Splits a multi-basis source run into per-basis cells under `results/published/<dataset>__<basis>/`. Drives the extraction table. |
| `scripts/validate_manifest.py` | CLI: validates `MANIFEST.json` against the on-disk cell tree. Exits non-zero on drift. |
| `scripts/render_published_readme.py` | CLI: regenerates the **headline numbers** table inside `results/published/README.md` from MANIFEST. Idempotent. |
| `scripts/run_canonical.sh` | Convenience: re-derive all canonical cells end-to-end. |
| `src/pdft_benchmarks/_extraction.py` | Pure-Python core of `extract_canonical_cells.py` (testable without filesystem coupling). |
| `src/pdft_benchmarks/_manifest.py` | Pure-Python core of MANIFEST schema, builders, and validation. |
| `tests/test_extraction.py` | Unit tests for `_extraction` module. |
| `tests/test_manifest.py` | Unit tests for `_manifest` module. |
| `tests/test_render_published_readme.py` | Unit tests for the README renderer. |
| `results/published/MANIFEST.json` | Generated; committed. |
| `results/published/README.md` | Generated; committed. |
| `results/_archive/README.md` | One paragraph: provenance role, not maintained. |
| `results/ablations/{rich_init,stacked_depth,batch_size,learned_vs_dct_block}/README.md` | One per ablation group. |

### Modified

| Path | Change |
|------|--------|
| `experiments/quickdraw.py` | Default preset `moderate` → `generalized`; bases `["qft", "entangled_qft", "tebd"]` → all 7; baselines extended with `block_fft_8`, `block_dct_8`. |
| `experiments/div2k_10q_circuit.py` | Add a comment header pointing to `div2k_10q_block.py` for the block bases. (No behavioral change.) |
| `pyproject.toml` | Optional: register `validate-manifest`, `extract-canonical-cells`, `render-published-readme` as console_scripts (low-priority; defer if it adds risk). |

### Moved (via `git mv`)

Many directories under `results/` are reorganized into `_archive/`, `ablations/`, or extracted into `published/`. The exact mapping is encoded as a Python data structure in `scripts/extract_canonical_cells.py` (Task 5) — the plan does not reproduce it manually here.

---

## Phase A — Tooling: extraction core + tests

The extraction logic transforms a multi-basis source run into one or more single-basis cells. It is the most error-prone piece because it has to rename basis keys (`blocked_qft` → `blocked`, etc.) and filter `metrics.json` correctly. We write it pure first, with tests.

### Task 1: `_extraction.py` — basis-key renaming

**Files:**
- Create: `src/pdft_benchmarks/_extraction.py`
- Test:   `tests/test_extraction.py`

The source-run flavor of basis names doesn't always match the registry key. The extraction table maps source key → registry key:

```python
SOURCE_TO_REGISTRY = {
    # Identity (source key already matches the registry key)
    "qft": "qft",
    "entangled_qft": "entangled_qft",
    "tebd": "tebd",
    "mera": "mera",
    # Renames
    "blocked_qft": "blocked",
    "blocked_rich": "rich",
    "blocked_real": "real_rich",
}
```

- [ ] **Step 1: Write the failing test**

Path: `tests/test_extraction.py`

```python
"""Unit tests for _extraction module."""

from __future__ import annotations

import pytest

from pdft_benchmarks._extraction import (
    SOURCE_TO_REGISTRY,
    rename_basis_key,
)


def test_identity_mapping_for_circuit_bases():
    assert rename_basis_key("qft") == "qft"
    assert rename_basis_key("entangled_qft") == "entangled_qft"
    assert rename_basis_key("tebd") == "tebd"
    assert rename_basis_key("mera") == "mera"


def test_block_bases_get_renamed():
    assert rename_basis_key("blocked_qft") == "blocked"
    assert rename_basis_key("blocked_rich") == "rich"
    assert rename_basis_key("blocked_real") == "real_rich"


def test_unknown_key_raises():
    with pytest.raises(KeyError, match="unknown source basis key"):
        rename_basis_key("nonsense_basis")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extraction.py::test_identity_mapping_for_circuit_bases -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'pdft_benchmarks._extraction'`

- [ ] **Step 3: Write minimal implementation**

Path: `src/pdft_benchmarks/_extraction.py`

```python
"""Pure-Python extraction core: split multi-basis source runs into per-basis cells.

The extraction table (TABLE) is the canonical mapping from
*(cell_id, source_run, source_basis_key)* to a single-basis flat cell directory
under results/published/. Importing modules drive filesystem layout from here;
this module stays IO-light so it can be tested without disk fixtures.
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

    Raises KeyError on unknown inputs (this is intentional: unknown keys mean
    the extraction table is out of date).
    """
    if source_key not in SOURCE_TO_REGISTRY:
        raise KeyError(f"unknown source basis key {source_key!r}")
    return SOURCE_TO_REGISTRY[source_key]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extraction.py -v --no-cov`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pdft_benchmarks/_extraction.py tests/test_extraction.py
git commit -m "feat(extraction): basis-key rename table + unit tests"
```

---

### Task 2: `_extraction.py` — `filter_metrics_for_cell`

**Files:**
- Modify: `src/pdft_benchmarks/_extraction.py`
- Modify: `tests/test_extraction.py`

`filter_metrics_for_cell(source_metrics, source_basis_key)` takes a multi-basis `metrics.json` payload and returns a new dict keyed by the registry-renamed basis + the 4 classical baselines.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extraction.py`:

```python
from pdft_benchmarks._extraction import filter_metrics_for_cell


def _fake_baseline_block():
    return {"metrics": {"0.05": {"mean_psnr": 20.0, "std_psnr": 0.0,
                                  "mean_mse": 0.01, "std_mse": 0.0,
                                  "mean_ssim": 0.5, "std_ssim": 0.0,
                                  "nan_count": 0}},
            "time": 0.1}


def _fake_basis_block(psnr=27.0):
    return {"metrics": {"0.05": {"mean_psnr": psnr, "std_psnr": 0.5,
                                  "mean_mse": 0.001, "std_mse": 0.0,
                                  "mean_ssim": 0.8, "std_ssim": 0.0,
                                  "nan_count": 0}},
            "time": 100.0,
            "_pdft_py": {"warmup_s": 5.0, "device": "cuda:0",
                         "epochs_completed": 60, "steps": 600,
                         "n_test": 50, "eval_failed_count": {"0.05": 0}}}


def test_filter_keeps_one_basis_plus_4_baselines():
    src = {
        "qft": _fake_basis_block(27.0),
        "entangled_qft": _fake_basis_block(28.0),
        "tebd": _fake_basis_block(29.0),
        "fft": _fake_baseline_block(),
        "dct": _fake_baseline_block(),
        "block_fft_8": _fake_baseline_block(),
        "block_dct_8": _fake_baseline_block(),
    }
    out = filter_metrics_for_cell(src, source_basis_key="qft")
    assert set(out) == {"qft", "fft", "dct", "block_fft_8", "block_dct_8"}
    # the other learned bases must be dropped
    assert "entangled_qft" not in out
    assert "tebd" not in out


def test_filter_renames_blocked_qft_to_blocked():
    src = {
        "blocked_qft": _fake_basis_block(),
        "fft": _fake_baseline_block(),
        "dct": _fake_baseline_block(),
        "block_fft_8": _fake_baseline_block(),
        "block_dct_8": _fake_baseline_block(),
    }
    out = filter_metrics_for_cell(src, source_basis_key="blocked_qft")
    assert "blocked" in out
    assert "blocked_qft" not in out


def test_filter_raises_when_source_key_missing():
    src = {"fft": _fake_baseline_block()}
    with pytest.raises(KeyError, match="not in source metrics"):
        filter_metrics_for_cell(src, source_basis_key="qft")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extraction.py -v --no-cov`
Expected: 3 new tests fail with `ImportError` for `filter_metrics_for_cell`.

- [ ] **Step 3: Write the implementation**

Append to `src/pdft_benchmarks/_extraction.py`:

```python
CLASSICAL_BASELINES: tuple[str, ...] = ("fft", "dct", "block_fft_8", "block_dct_8")


def filter_metrics_for_cell(
    source_metrics: dict, *, source_basis_key: str
) -> dict:
    """Return a single-basis metrics payload for one published cell.

    The returned dict has exactly five top-level keys: the renamed basis +
    the four classical baselines. Missing baselines are silently skipped
    (some older source runs only have fft/dct).
    """
    if source_basis_key not in source_metrics:
        raise KeyError(f"basis {source_basis_key!r} not in source metrics")
    dest_key = rename_basis_key(source_basis_key)
    out: dict = {dest_key: source_metrics[source_basis_key]}
    for baseline in CLASSICAL_BASELINES:
        if baseline in source_metrics:
            out[baseline] = source_metrics[baseline]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extraction.py -v --no-cov`
Expected: 6 passed total.

- [ ] **Step 5: Commit**

```bash
git add src/pdft_benchmarks/_extraction.py tests/test_extraction.py
git commit -m "feat(extraction): filter_metrics_for_cell"
```

---

### Task 3: `_extraction.py` — `build_config_json`

**Files:**
- Modify: `src/pdft_benchmarks/_extraction.py`
- Modify: `tests/test_extraction.py`

The published cell needs `config.json` with the frozen training config. We derive it from the source run's `env.json` (which has a `preset_dataclass` block with all the values).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extraction.py`:

```python
from pdft_benchmarks._extraction import build_config_json


def test_build_config_json_extracts_preset_fields():
    env = {
        "preset": "generalized",
        "preset_dataclass": {
            "epochs": 60,
            "n_train": 500,
            "n_test": 100,
            "optimizer": "adam",
            "batch_size": 8,
            "warmup_frac": 0.05,
            "lr_peak": 0.3,
            "lr_final": 0.0003,
            "max_grad_norm": 1.0,
            "validation_split": 0.15,
            "early_stopping_patience": 5,
            "seed": 0,
            "keep_ratios": [0.05, 0.1, 0.15, 0.2],
        },
    }
    cfg = build_config_json(env, m=8, n=8, basis="qft")
    assert cfg["m"] == 8
    assert cfg["n"] == 8
    assert cfg["basis"] == "qft"
    assert cfg["preset"] == "generalized"
    assert cfg["epochs"] == 60
    assert cfg["batch_size"] == 8
    assert cfg["lr_peak"] == 0.3
    assert cfg["seed"] == 0
    assert cfg["keep_ratios"] == [0.05, 0.1, 0.15, 0.2]


def test_build_config_json_raises_on_missing_preset_dataclass():
    with pytest.raises(KeyError, match="preset_dataclass"):
        build_config_json({}, m=8, n=8, basis="qft")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extraction.py::test_build_config_json_extracts_preset_fields -v --no-cov`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Append to `src/pdft_benchmarks/_extraction.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extraction.py -v --no-cov`
Expected: 8 passed total.

- [ ] **Step 5: Commit**

```bash
git add src/pdft_benchmarks/_extraction.py tests/test_extraction.py
git commit -m "feat(extraction): build_config_json"
```

---

### Task 4: `_extraction.py` — `extract_cell` filesystem driver

**Files:**
- Modify: `src/pdft_benchmarks/_extraction.py`
- Modify: `tests/test_extraction.py`

End-to-end driver that takes a source run dir + cell id + source basis key, populates the cell dir with all required files (Section "Per-cell contents" of the spec), then calls `pdft_benchmarks._report.main()` to regenerate per-cell CSVs and plots.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extraction.py`:

```python
import json
import shutil
from pathlib import Path

from pdft_benchmarks._extraction import extract_cell


def _make_fake_source(tmp_path: Path) -> Path:
    """Build a minimal valid source run dir with one basis (qft) + one
    baseline (fft), plus loss_history and trained_qft.json."""
    src = tmp_path / "src_run"
    (src / "loss_history").mkdir(parents=True)
    metrics = {
        "qft": _fake_basis_block(27.0),
        "fft": _fake_baseline_block(),
    }
    (src / "metrics.json").write_text(json.dumps(metrics))
    (src / "env.json").write_text(json.dumps({
        "preset": "generalized",
        "preset_dataclass": {
            "epochs": 60, "n_train": 500, "n_test": 100, "optimizer": "adam",
            "batch_size": 8, "warmup_frac": 0.05, "lr_peak": 0.3,
            "lr_final": 0.0003, "max_grad_norm": 1.0,
            "validation_split": 0.15, "early_stopping_patience": 5,
            "seed": 0, "keep_ratios": [0.05, 0.1, 0.15, 0.2],
        },
    }))
    (src / "trained_qft.json").write_text(json.dumps({"type": "QFTBasis", "m": 8, "n": 8, "tensors": []}))
    (src / "loss_history" / "qft_loss.json").write_text(json.dumps({"step_losses": [[1.0, 0.5]]}))
    (src / "run.log").write_text("fake log")
    return src


def test_extract_cell_writes_all_required_files(tmp_path):
    src = _make_fake_source(tmp_path)
    dest = tmp_path / "div2k_8q__qft"
    extract_cell(
        source_run=src,
        cell_dir=dest,
        source_basis_key="qft",
        m=8, n=8,
    )
    assert (dest / "metrics.json").is_file()
    assert (dest / "config.json").is_file()
    assert (dest / "env.json").is_file()
    assert (dest / "trained_qft.json").is_file()
    assert (dest / "loss_history" / "qft_loss.json").is_file()
    assert (dest / "run.log").is_file()


def test_extract_cell_renames_blocked_qft_to_blocked(tmp_path):
    src = tmp_path / "blocked_src"
    (src / "loss_history").mkdir(parents=True)
    metrics = {
        "blocked_qft": _fake_basis_block(28.0),
        "fft": _fake_baseline_block(),
    }
    (src / "metrics.json").write_text(json.dumps(metrics))
    (src / "env.json").write_text(json.dumps({
        "preset": "generalized",
        "preset_dataclass": {
            "epochs": 60, "n_train": 500, "n_test": 100, "optimizer": "adam",
            "batch_size": 8, "warmup_frac": 0.05, "lr_peak": 0.3,
            "lr_final": 0.0003, "max_grad_norm": 1.0,
            "validation_split": 0.15, "early_stopping_patience": 5,
            "seed": 0, "keep_ratios": [0.05, 0.1, 0.15, 0.2],
        },
    }))
    (src / "trained_blocked_qft.json").write_text("{}")
    (src / "loss_history" / "blocked_qft_loss.json").write_text("{}")

    dest = tmp_path / "div2k_8q__blocked"
    extract_cell(source_run=src, cell_dir=dest, source_basis_key="blocked_qft", m=8, n=8)
    # destination uses the renamed key
    assert (dest / "trained_blocked.json").is_file()
    assert (dest / "loss_history" / "blocked_loss.json").is_file()
    assert not (dest / "trained_blocked_qft.json").exists()
    out_metrics = json.loads((dest / "metrics.json").read_text())
    assert "blocked" in out_metrics
    assert "blocked_qft" not in out_metrics


def test_extract_cell_writes_skipped_json_for_skipped_basis(tmp_path):
    from pdft_benchmarks._extraction import write_skipped_cell

    dest = tmp_path / "div2k_10q__mera"
    write_skipped_cell(dest, m=10, n=10, basis="mera")
    payload = json.loads((dest / "SKIPPED.json").read_text())
    assert payload["reason"] == "incompatible_qubits"
    assert payload["m"] == 10
    assert payload["n"] == 10
    assert payload["basis"] == "mera"
    # nothing else should be written
    assert sorted(p.name for p in dest.iterdir()) == ["SKIPPED.json"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extraction.py -v --no-cov`
Expected: 3 new tests fail with `ImportError: cannot import name 'extract_cell'`.

- [ ] **Step 3: Write the implementation**

Append to `src/pdft_benchmarks/_extraction.py`:

```python
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

    # 1. Filtered metrics
    src_metrics = json.loads((source_run / "metrics.json").read_text())
    cell_metrics = filter_metrics_for_cell(src_metrics, source_basis_key=source_basis_key)
    (cell_dir / "metrics.json").write_text(json.dumps(cell_metrics, indent=2))

    # 2. env.json (verbatim)
    shutil.copyfile(source_run / "env.json", cell_dir / "env.json")
    src_env = json.loads((source_run / "env.json").read_text())

    # 3. config.json (synthesized)
    dest_basis = rename_basis_key(source_basis_key)
    cfg = build_config_json(src_env, m=m, n=n, basis=dest_basis)
    (cell_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # 4. trained_<basis>.json — copy + rename
    src_trained = source_run / f"trained_{source_basis_key}.json"
    if src_trained.is_file():
        shutil.copyfile(src_trained, cell_dir / f"trained_{dest_basis}.json")

    # 5. loss_history/<basis>_loss.json — copy + rename
    src_loss = source_run / "loss_history" / f"{source_basis_key}_loss.json"
    if src_loss.is_file():
        shutil.copyfile(src_loss, cell_dir / "loss_history" / f"{dest_basis}_loss.json")

    # 6. run.log — verbatim if present
    src_log = source_run / "run.log"
    if src_log.is_file():
        shutil.copyfile(src_log, cell_dir / "run.log")

    # 7. CSVs + plots — regenerate from cell-local metrics.json
    from pdft_benchmarks._report import main as report_main
    report_main(cell_dir)


def write_skipped_cell(cell_dir: Path, *, m: int, n: int, basis: str) -> None:
    """For (dataset, basis) pairs where the basis is incompatible with m+n,
    write only SKIPPED.json into the cell dir.
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "SKIPPED.json").write_text(json.dumps({
        "reason": "incompatible_qubits",
        "m": m, "n": n,
        "basis": basis,
        "constraint": "m+n must be a power of 2",
    }, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extraction.py -v --no-cov`
Expected: 11 passed total. (Note: `extract_cell` calls `_report.main` which will try to plot — if matplotlib backend complaints arise, set `MPLBACKEND=Agg` in `tests/conftest.py` or pytest env.)

If the plot step fails in tests because matplotlib has no display:
- Add `os.environ["MPLBACKEND"] = "Agg"` at the very top of `tests/conftest.py` if not already present.

- [ ] **Step 5: Commit**

```bash
git add src/pdft_benchmarks/_extraction.py tests/test_extraction.py
git commit -m "feat(extraction): extract_cell + write_skipped_cell driver"
```

---

## Phase B — Tooling: MANIFEST builder + validator

### Task 5: `_manifest.py` — schema constants + builder skeleton

**Files:**
- Create: `src/pdft_benchmarks/_manifest.py`
- Create: `tests/test_manifest.py`

The MANIFEST is the single source of truth for what's published. The builder reads each cell on disk and emits a fully-populated MANIFEST entry; the validator goes the other direction.

- [ ] **Step 1: Write the failing test**

Path: `tests/test_manifest.py`

```python
"""Unit tests for _manifest module."""

from __future__ import annotations

from pathlib import Path

from pdft_benchmarks._manifest import (
    SCHEMA_VERSION,
    DATASETS,
    BASES,
    CLASSICAL_BASELINES,
    MERA_INCOMPATIBLE_DATASETS,
)


def test_schema_version_is_string():
    assert SCHEMA_VERSION == "1.0"


def test_datasets_table_has_three_rows():
    assert set(DATASETS) == {"div2k_8q", "div2k_10q", "quickdraw"}
    for name, row in DATASETS.items():
        assert "m" in row and "n" in row
        assert "image_size" in row
        assert row["image_size"] == [2 ** row["m"], 2 ** row["n"]]


def test_bases_table_has_seven_keys():
    assert set(BASES) == {"qft", "entangled_qft", "tebd", "mera",
                          "blocked", "rich", "real_rich"}
    assert BASES["mera"]["constraint"] == "m+n must be power of 2"


def test_classical_baselines_constant():
    assert CLASSICAL_BASELINES == ["fft", "dct", "block_fft_8", "block_dct_8"]


def test_mera_incompatible_datasets():
    assert MERA_INCOMPATIBLE_DATASETS == {"div2k_10q", "quickdraw"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_manifest.py -v --no-cov`
Expected: 5 fails — `ModuleNotFoundError: No module named 'pdft_benchmarks._manifest'`.

- [ ] **Step 3: Write the implementation**

Path: `src/pdft_benchmarks/_manifest.py`

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_manifest.py -v --no-cov`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pdft_benchmarks/_manifest.py tests/test_manifest.py
git commit -m "feat(manifest): schema constants for results/published/"
```

---

### Task 6: `_manifest.py` — `summarize_metrics` for one cell

**Files:**
- Modify: `src/pdft_benchmarks/_manifest.py`
- Modify: `tests/test_manifest.py`

`summarize_metrics(cell_metrics, basis_key)` produces the `metrics_summary` block stored in each MANIFEST cell entry. It denormalizes the per-keep-ratio PSNR and adds train_time / num_parameters when available.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_manifest.py`:

```python
from pdft_benchmarks._manifest import summarize_metrics


def _make_metrics(psnrs: dict[str, float], time_s: float = 100.0):
    return {
        "qft": {
            "metrics": {
                kr: {"mean_mse": 0.0, "std_mse": 0.0,
                     "mean_psnr": p, "std_psnr": 0.0,
                     "mean_ssim": 0.5, "std_ssim": 0.0,
                     "nan_count": 0}
                for kr, p in psnrs.items()
            },
            "time": time_s,
            "_pdft_py": {"warmup_s": 5.0, "device": "cuda:0",
                         "epochs_completed": 60, "steps": 600,
                         "n_test": 50,
                         "eval_failed_count": {kr: 0 for kr in psnrs}},
        },
        "fft": {"metrics": {"0.05": {"mean_psnr": 20.0, "std_psnr": 0.0,
                                      "mean_mse": 0.0, "std_mse": 0.0,
                                      "mean_ssim": 0.0, "std_ssim": 0.0,
                                      "nan_count": 0}}, "time": 0.1},
    }


def test_summarize_extracts_psnr_per_keep_ratio():
    metrics = _make_metrics({"0.05": 24.5, "0.1": 27.0, "0.15": 29.0, "0.2": 30.5})
    summary = summarize_metrics(metrics, basis_key="qft")
    assert summary["psnr_at_keep_0.05"] == 24.5
    assert summary["psnr_at_keep_0.1"] == 27.0
    assert summary["psnr_at_keep_0.15"] == 29.0
    assert summary["psnr_at_keep_0.2"] == 30.5
    assert summary["train_time_s"] == 100.0


def test_summarize_returns_nan_for_missing_keep_ratio():
    import math
    metrics = _make_metrics({"0.05": 24.5})
    summary = summarize_metrics(metrics, basis_key="qft")
    assert math.isnan(summary["psnr_at_keep_0.1"])


def test_summarize_raises_if_basis_missing():
    import pytest
    metrics = {"fft": {"metrics": {}, "time": 0.0}}
    with pytest.raises(KeyError, match="basis 'qft' not in metrics"):
        summarize_metrics(metrics, basis_key="qft")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_manifest.py -v --no-cov`
Expected: 3 new fails with `ImportError`.

- [ ] **Step 3: Write the implementation**

Append to `src/pdft_benchmarks/_manifest.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_manifest.py -v --no-cov`
Expected: 8 passed total.

- [ ] **Step 5: Commit**

```bash
git add src/pdft_benchmarks/_manifest.py tests/test_manifest.py
git commit -m "feat(manifest): summarize_metrics per cell"
```

---

### Task 7: `_manifest.py` — `build_manifest` over a published tree

**Files:**
- Modify: `src/pdft_benchmarks/_manifest.py`
- Modify: `tests/test_manifest.py`

Walk `results/published/<dataset>__<basis>/` for each (dataset, basis) pair and produce the full MANIFEST. SKIPPED cells produce a status="skipped" entry; active cells produce status="active" with `metrics_summary`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_manifest.py`:

```python
import json
from datetime import datetime
from pathlib import Path

from pdft_benchmarks._manifest import build_manifest


def _write_active_cell(root: Path, dataset: str, basis: str, psnr_at_0_1: float):
    cell = root / f"{dataset}__{basis}"
    cell.mkdir(parents=True)
    (cell / "metrics.json").write_text(json.dumps({
        basis: {"metrics": {kr: {"mean_psnr": psnr_at_0_1 + i,
                                  "std_psnr": 0.0, "mean_mse": 0.0,
                                  "std_mse": 0.0, "mean_ssim": 0.5,
                                  "std_ssim": 0.0, "nan_count": 0}
                            for i, kr in enumerate(["0.05", "0.1", "0.15", "0.2"])},
                "time": 50.0,
                "_pdft_py": {"warmup_s": 5.0, "device": "cuda:0",
                             "epochs_completed": 60, "steps": 600,
                             "n_test": 50,
                             "eval_failed_count": {kr: 0 for kr in ["0.05", "0.1", "0.15", "0.2"]}}},
        "fft": {"metrics": {}, "time": 0.0},
    }))
    (cell / "config.json").write_text(json.dumps({"epochs": 60, "n_train": 500, "n_test": 50,
                                                   "lr_peak": 0.3, "batch_size": 8, "seed": 0,
                                                   "preset": "generalized"}))


def _write_skipped_cell(root: Path, dataset: str, basis: str, m: int, n: int):
    cell = root / f"{dataset}__{basis}"
    cell.mkdir(parents=True)
    (cell / "SKIPPED.json").write_text(json.dumps({
        "reason": "incompatible_qubits", "m": m, "n": n, "basis": basis,
        "constraint": "m+n must be a power of 2",
    }))


def test_build_manifest_produces_21_cells(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir()
    for ds in ("div2k_8q", "div2k_10q", "quickdraw"):
        for b in ("qft", "entangled_qft", "tebd", "blocked", "rich", "real_rich"):
            _write_active_cell(pub, ds, b, 27.0)
    _write_active_cell(pub, "div2k_8q", "mera", 27.0)
    _write_skipped_cell(pub, "div2k_10q", "mera", 10, 10)
    _write_skipped_cell(pub, "quickdraw", "mera", 5, 5)

    manifest = build_manifest(pub, git_sha="deadbeef", pdft_version="0.2.1")
    assert len(manifest["cells"]) == 21
    statuses = [c["status"] for c in manifest["cells"]]
    assert statuses.count("active") == 19
    assert statuses.count("skipped") == 2
    assert manifest["schema_version"] == "1.0"
    assert manifest["pdft_version"] == "0.2.1"
    assert manifest["git_sha"] == "deadbeef"


def test_build_manifest_includes_psnr_summary(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir()
    _write_active_cell(pub, "div2k_8q", "qft", 27.0)
    _write_active_cell(pub, "div2k_8q", "entangled_qft", 28.0)
    _write_active_cell(pub, "div2k_8q", "tebd", 28.0)
    _write_active_cell(pub, "div2k_8q", "mera", 28.0)
    _write_active_cell(pub, "div2k_8q", "blocked", 28.0)
    _write_active_cell(pub, "div2k_8q", "rich", 28.0)
    _write_active_cell(pub, "div2k_8q", "real_rich", 28.0)
    _write_skipped_cell(pub, "div2k_10q", "mera", 10, 10)
    for b in ("qft", "entangled_qft", "tebd", "blocked", "rich", "real_rich"):
        _write_active_cell(pub, "div2k_10q", b, 25.0)
    _write_skipped_cell(pub, "quickdraw", "mera", 5, 5)
    for b in ("qft", "entangled_qft", "tebd", "blocked", "rich", "real_rich"):
        _write_active_cell(pub, "quickdraw", b, 22.0)

    manifest = build_manifest(pub, git_sha="abc", pdft_version="0.2.1")
    qft_cell = next(c for c in manifest["cells"] if c["id"] == "div2k_8q__qft")
    assert qft_cell["metrics_summary"]["psnr_at_keep_0.1"] == 28.0  # 27.0 + i=1
    skipped = next(c for c in manifest["cells"] if c["id"] == "div2k_10q__mera")
    assert "metrics_summary" not in skipped
    assert skipped["skip_reason"].startswith("incompatible_qubits")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_manifest.py -v --no-cov`
Expected: 2 new fails with `ImportError: cannot import name 'build_manifest'`.

- [ ] **Step 3: Write the implementation**

Append to `src/pdft_benchmarks/_manifest.py`:

```python
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
                # validator (Task 8) is the strict checker.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_manifest.py -v --no-cov`
Expected: 10 passed total.

- [ ] **Step 5: Commit**

```bash
git add src/pdft_benchmarks/_manifest.py tests/test_manifest.py
git commit -m "feat(manifest): build_manifest from published tree"
```

---

### Task 8: `_manifest.py` — `validate_manifest`

**Files:**
- Modify: `src/pdft_benchmarks/_manifest.py`
- Modify: `tests/test_manifest.py`

Validate that on-disk state matches MANIFEST: every active cell has the required files, every skipped cell has only `SKIPPED.json`, every cell's `metrics_summary` matches its `metrics.json` to within float tolerance.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_manifest.py`:

```python
import pytest
from pdft_benchmarks._manifest import validate_manifest, ManifestValidationError


REQUIRED_ACTIVE_FILES = (
    "metrics.json", "env.json", "config.json",
    "rate_distortion_mse.csv",
    "rate_distortion_psnr.csv",
    "rate_distortion_ssim.csv",
    "timing_summary.csv",
)


def _populate_required_files(cell_dir: Path):
    """Top off a cell built by `_write_active_cell` with the rest of the
    files validate_manifest requires."""
    (cell_dir / "env.json").write_text("{}")
    for fname in ("rate_distortion_mse.csv", "rate_distortion_psnr.csv",
                  "rate_distortion_ssim.csv", "timing_summary.csv"):
        (cell_dir / fname).write_text("basis,keep_ratio,mean,std\n")
    (cell_dir / "loss_history").mkdir(exist_ok=True)


def test_validate_passes_on_well_formed_tree(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir()
    for ds in ("div2k_8q", "div2k_10q", "quickdraw"):
        for b in ("qft", "entangled_qft", "tebd", "blocked", "rich", "real_rich"):
            _write_active_cell(pub, ds, b, 27.0)
            _populate_required_files(pub / f"{ds}__{b}")
    _write_active_cell(pub, "div2k_8q", "mera", 27.0)
    _populate_required_files(pub / "div2k_8q__mera")
    _write_skipped_cell(pub, "div2k_10q", "mera", 10, 10)
    _write_skipped_cell(pub, "quickdraw", "mera", 5, 5)

    manifest = build_manifest(pub, git_sha="abc", pdft_version="0.2.1")
    (pub / "MANIFEST.json").write_text(json.dumps(manifest))

    # Must not raise
    validate_manifest(pub)


def test_validate_fails_when_active_cell_missing_required_file(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir()
    _write_active_cell(pub, "div2k_8q", "qft", 27.0)
    # Intentionally do NOT call _populate_required_files
    manifest = {
        "schema_version": "1.0",
        "datasets": DATASETS, "bases": BASES,
        "classical_baselines": CLASSICAL_BASELINES,
        "cells": [{"id": "div2k_8q__qft", "dataset": "div2k_8q",
                    "basis": "qft", "status": "active",
                    "path": "div2k_8q__qft/", "preset": "generalized",
                    "config": {"epochs": 60, "n_train": 500, "n_test": 100,
                               "lr_peak": 0.3, "batch_size": 8, "seed": 0},
                    "metrics_summary": {"psnr_at_keep_0.1": 28.0,
                                         "psnr_at_keep_0.05": 27.0,
                                         "psnr_at_keep_0.15": 29.0,
                                         "psnr_at_keep_0.2": 30.0,
                                         "train_time_s": 50.0}}],
        "git_sha": "abc", "pdft_version": "0.2.1", "generated_at": "x",
    }
    (pub / "MANIFEST.json").write_text(json.dumps(manifest))

    with pytest.raises(ManifestValidationError, match="missing required file"):
        validate_manifest(pub)


def test_validate_fails_on_skipped_cell_with_extra_files(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir()
    _write_skipped_cell(pub, "div2k_10q", "mera", 10, 10)
    # Sneak in an extra file
    (pub / "div2k_10q__mera" / "stowaway.json").write_text("{}")
    manifest = {
        "schema_version": "1.0",
        "datasets": DATASETS, "bases": BASES,
        "classical_baselines": CLASSICAL_BASELINES,
        "cells": [{"id": "div2k_10q__mera", "dataset": "div2k_10q",
                    "basis": "mera", "status": "skipped",
                    "path": "div2k_10q__mera/",
                    "skip_reason": "incompatible_qubits: m+n=20 is not a power of 2"}],
        "git_sha": "abc", "pdft_version": "0.2.1", "generated_at": "x",
    }
    (pub / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifestValidationError, match="extra files in skipped cell"):
        validate_manifest(pub)


def test_validate_fails_when_metrics_summary_disagrees(tmp_path):
    pub = tmp_path / "published"
    pub.mkdir()
    _write_active_cell(pub, "div2k_8q", "qft", 27.0)  # actual psnr_at_keep_0.1 = 28.0
    _populate_required_files(pub / "div2k_8q__qft")
    # Manifest claims 99.0 instead
    manifest = build_manifest(pub, git_sha="x", pdft_version="0.2.1")
    for c in manifest["cells"]:
        if c["id"] == "div2k_8q__qft":
            c["metrics_summary"]["psnr_at_keep_0.1"] = 99.0
    (pub / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifestValidationError, match="metrics_summary mismatch"):
        validate_manifest(pub)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_manifest.py -v --no-cov`
Expected: 4 new fails with `ImportError: cannot import name 'validate_manifest'` etc.

- [ ] **Step 3: Write the implementation**

Append to `src/pdft_benchmarks/_manifest.py`:

```python
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
        # active cell
        for fname in _REQUIRED_ACTIVE_FILES:
            if not (cell_dir / fname).is_file():
                raise ManifestValidationError(
                    f"missing required file in {cell['id']}: {fname}"
                )
        # cross-check metrics_summary
        on_disk_metrics = json.loads((cell_dir / "metrics.json").read_text())
        recomputed = summarize_metrics(on_disk_metrics, basis_key=cell["basis"])
        for k, v in cell["metrics_summary"].items():
            if not _summary_close(float(v), float(recomputed[k])):
                raise ManifestValidationError(
                    f"metrics_summary mismatch in {cell['id']}.{k}: "
                    f"manifest={v} on-disk={recomputed[k]}"
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_manifest.py -v --no-cov`
Expected: 14 passed total.

- [ ] **Step 5: Commit**

```bash
git add src/pdft_benchmarks/_manifest.py tests/test_manifest.py
git commit -m "feat(manifest): validate_manifest with disk cross-check"
```

---

## Phase C — Tooling: README renderer

### Task 9: `render_published_readme.py` — headline-table generator

**Files:**
- Create: `scripts/render_published_readme.py`
- Create: `tests/test_render_published_readme.py`

Read MANIFEST, write/update the **Headline numbers** table inside `results/published/README.md`. Idempotent: running twice yields the same file.

The README has a marker comment block:
```
<!-- BEGIN HEADLINE NUMBERS (auto-generated; do not edit) -->
<!-- END HEADLINE NUMBERS -->
```
The renderer rewrites everything between those markers.

- [ ] **Step 1: Write the failing test**

Path: `tests/test_render_published_readme.py`

```python
"""Tests for scripts/render_published_readme.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# Adopt the script via importlib since scripts/ isn't a package.
def _load_renderer():
    import importlib.util
    here = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "render_published_readme",
        here / "scripts" / "render_published_readme.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seed_readme(path: Path):
    path.write_text(
        "# Title\n\n"
        "## Headline numbers\n\n"
        "<!-- BEGIN HEADLINE NUMBERS (auto-generated; do not edit) -->\n"
        "(placeholder)\n"
        "<!-- END HEADLINE NUMBERS -->\n\n"
        "## Next section unchanged\n"
    )


def _seed_manifest(path: Path, psnrs):
    """psnrs: {(dataset, basis): psnr_at_keep_0.1}"""
    cells = []
    for (ds, b), p in psnrs.items():
        cells.append({"id": f"{ds}__{b}", "dataset": ds, "basis": b,
                       "status": "active", "path": f"{ds}__{b}/",
                       "metrics_summary": {"psnr_at_keep_0.1": p,
                                            "psnr_at_keep_0.05": 0.0,
                                            "psnr_at_keep_0.15": 0.0,
                                            "psnr_at_keep_0.2": 0.0,
                                            "train_time_s": 0.0}})
    cells.append({"id": "div2k_10q__mera", "dataset": "div2k_10q", "basis": "mera",
                   "status": "skipped", "path": "div2k_10q__mera/",
                   "skip_reason": "incompatible_qubits: m+n=20 is not a power of 2"})
    cells.append({"id": "quickdraw__mera", "dataset": "quickdraw", "basis": "mera",
                   "status": "skipped", "path": "quickdraw__mera/",
                   "skip_reason": "incompatible_qubits: m+n=10 is not a power of 2"})
    path.write_text(json.dumps({"cells": cells, "schema_version": "1.0",
                                  "datasets": {}, "bases": {}, "classical_baselines": [],
                                  "git_sha": "abc", "pdft_version": "0.2.1",
                                  "generated_at": "x"}))


def test_render_inserts_table_between_markers(tmp_path):
    rdr = _load_renderer()
    pub = tmp_path
    _seed_readme(pub / "README.md")
    _seed_manifest(pub / "MANIFEST.json", {
        ("div2k_8q", "qft"): 28.0,
        ("div2k_8q", "mera"): 29.0,
    })
    rdr.render(pub)
    text = (pub / "README.md").read_text()
    assert "(placeholder)" not in text
    assert "28.00" in text  # qft @ 0.1
    assert "29.00" in text  # mera @ 0.1
    # Markers preserved
    assert "<!-- BEGIN HEADLINE NUMBERS" in text
    assert "<!-- END HEADLINE NUMBERS -->" in text


def test_render_is_idempotent(tmp_path):
    rdr = _load_renderer()
    pub = tmp_path
    _seed_readme(pub / "README.md")
    _seed_manifest(pub / "MANIFEST.json", {("div2k_8q", "qft"): 28.0})
    rdr.render(pub)
    once = (pub / "README.md").read_text()
    rdr.render(pub)
    twice = (pub / "README.md").read_text()
    assert once == twice


def test_render_marks_skipped_as_dash(tmp_path):
    rdr = _load_renderer()
    pub = tmp_path
    _seed_readme(pub / "README.md")
    _seed_manifest(pub / "MANIFEST.json", {})  # no active cells
    rdr.render(pub)
    text = (pub / "README.md").read_text()
    # rows for div2k_10q and quickdraw should show "—" in the mera column
    assert "—" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_render_published_readme.py -v --no-cov`
Expected: 3 fails because `scripts/render_published_readme.py` doesn't exist yet.

- [ ] **Step 3: Write the implementation**

Path: `scripts/render_published_readme.py`

```python
#!/usr/bin/env python3
"""Regenerate the `## Headline numbers` table in results/published/README.md.

Reads results/published/MANIFEST.json. Idempotent: rewrites only the block
between `<!-- BEGIN HEADLINE NUMBERS ... -->` and `<!-- END HEADLINE NUMBERS -->`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BEGIN = "<!-- BEGIN HEADLINE NUMBERS (auto-generated; do not edit) -->"
END = "<!-- END HEADLINE NUMBERS -->"

DATASETS_ORDER = ("div2k_8q", "div2k_10q", "quickdraw")
BASES_ORDER = ("qft", "entangled_qft", "tebd", "mera",
               "blocked", "rich", "real_rich")


def _build_table(manifest: dict) -> str:
    by_id = {c["id"]: c for c in manifest["cells"]}
    header = ("| | " + " | ".join(BASES_ORDER) + " |\n"
              "|" + "|".join(["---"] * (len(BASES_ORDER) + 1)) + "|\n")
    rows = []
    for ds in DATASETS_ORDER:
        cells_text = []
        for b in BASES_ORDER:
            cell = by_id.get(f"{ds}__{b}")
            if cell is None or cell.get("status") == "skipped":
                cells_text.append("—")
            else:
                psnr = cell["metrics_summary"]["psnr_at_keep_0.1"]
                cells_text.append(f"{psnr:.2f}")
        rows.append(f"| **{ds}** | " + " | ".join(cells_text) + " |")
    return header + "\n".join(rows) + "\n"


def render(published_root: Path) -> None:
    manifest = json.loads((published_root / "MANIFEST.json").read_text())
    readme_path = published_root / "README.md"
    text = readme_path.read_text()
    if BEGIN not in text or END not in text:
        raise RuntimeError(
            f"README.md missing markers; expected:\n{BEGIN}\n{END}"
        )
    table = _build_table(manifest)
    pre, _, rest = text.partition(BEGIN)
    _, _, post = rest.partition(END)
    new_text = pre + BEGIN + "\n" + table + END + post
    readme_path.write_text(new_text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-root",
                        default="results/published",
                        type=Path)
    args = parser.parse_args(argv)
    render(args.published_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Make the script executable**

```bash
chmod +x scripts/render_published_readme.py
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_render_published_readme.py -v --no-cov`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/render_published_readme.py tests/test_render_published_readme.py
git commit -m "feat(scripts): render_published_readme — idempotent headline table generator"
```

---

## Phase D — CLI wrappers

### Task 10: `validate_manifest.py` CLI

**Files:**
- Create: `scripts/validate_manifest.py`

Thin CLI wrapping `pdft_benchmarks._manifest.validate_manifest`. Exits 0 on success, 1 on validation error, 2 on argparse error.

- [ ] **Step 1: Write the implementation**

Path: `scripts/validate_manifest.py`

```python
#!/usr/bin/env python3
"""Validate results/published/MANIFEST.json against on-disk cell tree.

Exits 0 on success, 1 on validation error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdft_benchmarks._manifest import validate_manifest, ManifestValidationError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-root",
                        default="results/published",
                        type=Path)
    args = parser.parse_args(argv)
    try:
        validate_manifest(args.published_root)
    except ManifestValidationError as e:
        print(f"validate_manifest: FAIL — {e}", file=sys.stderr)
        return 1
    print(f"validate_manifest: OK ({args.published_root})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make executable + smoke-run**

```bash
chmod +x scripts/validate_manifest.py
python scripts/validate_manifest.py --published-root /tmp/nonexistent_dir 2>&1 || echo "exit=$?"
```
Expected: prints `validate_manifest: FAIL — MANIFEST.json not found at ...`, exits 1.

- [ ] **Step 3: Commit**

```bash
git add scripts/validate_manifest.py
git commit -m "feat(scripts): validate_manifest CLI"
```

---

### Task 11: `extract_canonical_cells.py` CLI + extraction table

**Files:**
- Create: `scripts/extract_canonical_cells.py`

The script holds the **extraction table** (the canonical mapping from existing source runs to cell ids), and drives `_extraction.extract_cell` over each entry. It is the single place to update when source-run names change.

- [ ] **Step 1: Write the implementation**

Path: `scripts/extract_canonical_cells.py`

```python
#!/usr/bin/env python3
"""Populate results/published/<dataset>__<basis>/ from existing source runs.

The EXTRACTION_TABLE below is the canonical mapping of (cell_id) →
(source_run_dir, source_basis_key). The 9 freshly-trained cells (3 on
div2k_10q + 6 on quickdraw) are added by editing the table after those
runs land in results/_archive/ (or wherever the runner deposits them).

Usage:
    python scripts/extract_canonical_cells.py
    python scripts/extract_canonical_cells.py --results-root results --published-root results/published
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdft_benchmarks._extraction import extract_cell, write_skipped_cell

# (m, n) per dataset for synthesizing config.json
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
    # The runner writes to results/div2k_10q_blocked_generalized_<ts>/;
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

    # ===== QuickDraw (all 6 from new run) =====
    {"cell_id": "quickdraw__qft",
     "source_run": "_archive/quickdraw_generalized_NEW",
     "source_basis_key": "qft", "dataset": "quickdraw"},
    {"cell_id": "quickdraw__entangled_qft",
     "source_run": "_archive/quickdraw_generalized_NEW",
     "source_basis_key": "entangled_qft", "dataset": "quickdraw"},
    {"cell_id": "quickdraw__tebd",
     "source_run": "_archive/quickdraw_generalized_NEW",
     "source_basis_key": "tebd", "dataset": "quickdraw"},
    {"cell_id": "quickdraw__blocked",
     "source_run": "_archive/quickdraw_generalized_NEW",
     "source_basis_key": "blocked_qft", "dataset": "quickdraw"},
    {"cell_id": "quickdraw__rich",
     "source_run": "_archive/quickdraw_generalized_NEW",
     "source_basis_key": "blocked_rich", "dataset": "quickdraw"},
    {"cell_id": "quickdraw__real_rich",
     "source_run": "_archive/quickdraw_generalized_NEW",
     "source_basis_key": "blocked_real", "dataset": "quickdraw"},
]

SKIPPED_CELLS: list[dict] = [
    {"cell_id": "div2k_10q__mera", "dataset": "div2k_10q", "basis": "mera"},
    {"cell_id": "quickdraw__mera", "dataset": "quickdraw", "basis": "mera"},
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results", type=Path)
    parser.add_argument("--published-root", default="results/published", type=Path)
    parser.add_argument("--only", help="comma-separated cell_id substring filter", default="")
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
        print(f"skipped {entry['cell_id']}")
        write_skipped_cell(dest, m=m, n=n, basis=entry["basis"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/extract_canonical_cells.py
```

- [ ] **Step 3: Smoke-run with no source dirs (table loads, prints SKIP messages)**

```bash
python scripts/extract_canonical_cells.py --results-root /tmp/nonexistent --published-root /tmp/scratch_pub 2>&1 | head -3
```
Expected: prints "SKIP div2k_8q__qft — source missing: /tmp/nonexistent/_archive/..." for each row, exit 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/extract_canonical_cells.py
git commit -m "feat(scripts): extract_canonical_cells — driver + extraction table"
```

---

### Task 12: `experiments/div2k_10q_block.py` — runner for the 3 missing 10q block bases

**Files:**
- Create: `experiments/div2k_10q_block.py`

Sibling to `experiments/div2k_10q_circuit.py`. Trains `blocked`, `rich`, `real_rich` at m=n=10 with batch_size=2. Uses `dataclasses.replace` to override the preset's batch_size from the default 4 to 2 (matching the existing canonical 10q runs).

- [ ] **Step 1: Write the implementation**

Path: `experiments/div2k_10q_block.py`

```python
#!/usr/bin/env python3
"""DIV2K-10q (m=n=10, 1024×1024) — block bases.

Sibling of div2k_10q_circuit.py. Trains the three block bases registered
in pdft_benchmarks.bases.BASIS_FACTORIES — `blocked`, `rich`, `real_rich`
— under the `generalized` preset overridden to batch_size=2 (matches the
existing canonical 10q circuit runs).

MERA is not trained here (the circuit script doesn't either; m+n=20 is
not a power of 2).
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from pdft_benchmarks.pipeline import run_experiment
from pdft_benchmarks.presets import get_preset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base_preset = get_preset("div2k_10q", "generalized")
    # Match the existing canonical 10q runs: bs=2, lr_peak=0.003.
    preset = replace(base_preset, batch_size=2, lr_peak=0.003)

    res = run_experiment(
        dataset="div2k",
        m=10, n=10,
        bases=["blocked", "rich", "real_rich"],
        baselines=["fft", "dct", "block_fft_8", "block_dct_8"],
        preset=preset,
        output_dir=args.out,
        device=f"gpu:{args.gpu}" if args.gpu is not None else "auto",
    )
    print(f"\nDone. Results: {res.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-import**

```bash
python -c "import importlib.util, pathlib; \
  spec = importlib.util.spec_from_file_location('m', pathlib.Path('experiments/div2k_10q_block.py')); \
  mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); \
  print('OK', hasattr(mod, 'main'))"
```
Expected: prints `OK True`.

- [ ] **Step 3: Commit**

```bash
git add experiments/div2k_10q_block.py
git commit -m "feat(experiments): div2k_10q_block.py — train blocked/rich/real_rich at 10q"
```

---

### Task 13: `experiments/quickdraw.py` — expand to all 7 bases at `generalized`

**Files:**
- Modify: `experiments/quickdraw.py`

Default preset `moderate` → `generalized`. Add `mera`, `blocked`, `rich`, `real_rich` to the bases list (mera will skip silently). Add `block_fft_8`, `block_dct_8` to baselines.

- [ ] **Step 1: Replace the file body**

Show full new content of `experiments/quickdraw.py`:

```python
#!/usr/bin/env python3
"""QuickDraw benchmark (m=n=5, 32×32) — all 7 registered bases at `generalized`.

`mera` is silently skipped by run_experiment because m+n=10 is not a
power of 2 (the resulting cell will be marked SKIPPED in the published
tree by extract_canonical_cells.py).
"""

import argparse

from pdft_benchmarks.pipeline import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="generalized",
                        choices=["smoke", "moderate", "generalized"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    res = run_experiment(
        dataset="quickdraw",
        m=5, n=5,
        bases=["qft", "entangled_qft", "tebd", "mera",
               "blocked", "rich", "real_rich"],
        baselines=["fft", "dct", "block_fft_8", "block_dct_8"],
        preset=args.preset,
        output_dir=args.out,
        device=f"gpu:{args.gpu}" if args.gpu is not None else "auto",
    )
    print(f"\nDone. Results: {res.output_dir}")


if __name__ == "__main__":
    main()
```

Use the Edit tool with full-file replacement (or `replace_all` style) to overwrite the existing content. The full new content above replaces lines 1-end of `experiments/quickdraw.py`.

- [ ] **Step 2: Smoke-import to confirm syntax**

```bash
python -c "import importlib.util, pathlib; \
  spec = importlib.util.spec_from_file_location('m', pathlib.Path('experiments/quickdraw.py')); \
  mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); \
  print('OK', hasattr(mod, 'main'))"
```
Expected: prints `OK True`.

- [ ] **Step 3: Commit**

```bash
git add experiments/quickdraw.py
git commit -m "feat(experiments): quickdraw — generalized preset, all 7 bases, full baselines"
```

---

### Task 14: `scripts/run_canonical.sh` — convenience wrapper

**Files:**
- Create: `scripts/run_canonical.sh`

End-to-end: train missing cells, then extract, then validate. Two GPUs in parallel by default.

- [ ] **Step 1: Write the implementation**

Path: `scripts/run_canonical.sh`

```bash
#!/usr/bin/env bash
# Re-derive every canonical cell from a fresh checkout.
#
# Wall clock on 2x RTX 3090 (concurrent): ~3 hours.
# - DIV2K-10q block bases:  ~50 min/basis × 3 = 2.5 h on GPU 0
# - QuickDraw all 6 active: ~5 min/basis × 6  = 30 min on GPU 1
# - DIV2K-8q (already in archive) is not retrained — extracted only.
#
# Usage:
#     bash scripts/run_canonical.sh
#     GPU0=0 GPU1=1 bash scripts/run_canonical.sh
#
# Set ONLY_EXTRACT=1 to skip training and just (re-)build results/published/
# from whatever already exists in results/_archive/.
set -euo pipefail

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

if [ -z "${ONLY_EXTRACT:-}" ]; then
    echo "[run_canonical] starting div2k_10q_block on GPU ${GPU0} (background)…"
    python experiments/div2k_10q_block.py --gpu "${GPU0}" &
    pid_10q=$!

    echo "[run_canonical] starting quickdraw on GPU ${GPU1} (background)…"
    python experiments/quickdraw.py --gpu "${GPU1}" &
    pid_qd=$!

    wait "${pid_10q}" "${pid_qd}"
    echo "[run_canonical] both training jobs finished."
    echo "[run_canonical] NOTE: now move new results dirs into results/_archive/"
    echo "                and update EXTRACTION_TABLE in scripts/extract_canonical_cells.py"
    echo "                to point at them, then re-run with ONLY_EXTRACT=1."
    exit 0
fi

echo "[run_canonical] extracting cells…"
python scripts/extract_canonical_cells.py

echo "[run_canonical] (re)building MANIFEST.json + README.md…"
python scripts/render_published_readme.py

echo "[run_canonical] validating…"
python scripts/validate_manifest.py

echo "[run_canonical] OK"
```

- [ ] **Step 2: Make executable + lint with shellcheck if available**

```bash
chmod +x scripts/run_canonical.sh
command -v shellcheck >/dev/null && shellcheck scripts/run_canonical.sh || true
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_canonical.sh
git commit -m "feat(scripts): run_canonical.sh end-to-end wrapper"
```

---

## Phase E — User-driven training runs (PAUSE POINT)

These tasks require GPU time. The plan **pauses** here for the user to actually execute the runs. After they finish, the user resumes from Task 17.

### Task 15: User runs `experiments/div2k_10q_block.py` (manual, ~2.5 h on 1 GPU)

**Files:**
- Reads from existing code; produces a new run dir at `results/div2k_10q_<timestamp>/`.

- [ ] **Step 1 (USER): Run the script**

```bash
python experiments/div2k_10q_block.py --gpu 0
```

Expected: completes in ~2.5 h, produces `results/div2k_10q_generalized_<TIMESTAMP>/` containing a multi-basis `metrics.json` with keys `["blocked_qft", "blocked_rich", "blocked_real", "fft", "dct", "block_fft_8", "block_dct_8"]`.

- [ ] **Step 2 (USER): Capture the new dir name for Task 17**

Set an env var or note the path:
```bash
export NEW_10Q_DIR="$(ls -dt results/div2k_10q_generalized_* | head -1)"
echo "${NEW_10Q_DIR}"
```

(No commit yet — the dir is moved into `_archive/` in Task 17.)

---

### Task 16: User runs `experiments/quickdraw.py` (manual, ~30 min on 1 GPU)

- [ ] **Step 1 (USER): Run the script**

```bash
python experiments/quickdraw.py --gpu 1   # use a different GPU if running in parallel with Task 15
```

Expected: completes in ~30 min, produces `results/quickdraw_generalized_<TIMESTAMP>/` containing keys `["qft", "entangled_qft", "tebd", "blocked_qft", "blocked_rich", "blocked_real", "fft", "dct", "block_fft_8", "block_dct_8"]` (mera will be silently skipped, written as `{"skipped": "incompatible_qubits"}` in metrics.json).

- [ ] **Step 2 (USER): Capture the new dir name**

```bash
export NEW_QUICKDRAW_DIR="$(ls -dt results/quickdraw_generalized_* | head -1)"
echo "${NEW_QUICKDRAW_DIR}"
```

---

## Phase F — Curate the filesystem

### Task 17: Update extraction table to point at the new run dirs

**Files:**
- Modify: `scripts/extract_canonical_cells.py`

The placeholder paths `_archive/div2k_10q_blocked_generalized_NEW` and `_archive/quickdraw_generalized_NEW` are replaced with the actual timestamped dir names from Tasks 15-16.

- [ ] **Step 1: Look up the actual dir names**

```bash
ls -d results/div2k_10q_generalized_* results/quickdraw_generalized_* 2>/dev/null
```
Note the exact timestamps.

- [ ] **Step 2: Replace placeholder strings**

Use the Edit tool to replace:
- `_archive/div2k_10q_blocked_generalized_NEW` → `_archive/<actual-10q-dirname>` (replace_all=True for the 3 occurrences)
- `_archive/quickdraw_generalized_NEW` → `_archive/<actual-quickdraw-dirname>` (replace_all=True for the 6 occurrences)

- [ ] **Step 3: Commit**

```bash
git add scripts/extract_canonical_cells.py
git commit -m "chore(scripts): point extraction table at the freshly-trained runs"
```

---

### Task 18: Create the new directory layout

**Files:**
- New dirs: `results/published/`, `results/_archive/`, `results/ablations/{rich_init,stacked_depth,batch_size,learned_vs_dct_block}/`.

- [ ] **Step 1: Create dirs**

```bash
mkdir -p results/published \
         results/_archive \
         results/ablations/rich_init \
         results/ablations/stacked_depth \
         results/ablations/batch_size \
         results/ablations/learned_vs_dct_block
```

- [ ] **Step 2: Verify**

```bash
ls results/
ls results/ablations/
```
Expected: shows new top-level dirs alongside the existing timestamped run dirs.

(No commit — empty dirs aren't tracked by git anyway. Content arrives in subsequent tasks.)

---

### Task 19: Move source runs into `_archive/`

**Files:**
- `git mv` the existing timestamped run dirs into `results/_archive/`.

The source runs the extraction table references must be at the path the table claims. The simplest convention: move **all** existing `results/<not-the-new-dirs>/` into `results/_archive/`, then individual ablation moves come next.

- [ ] **Step 1: List candidates to archive**

```bash
ls -1 results/ | grep -vE '^(_archive|published|ablations)$' | grep -v '^run_.*\.log$'
```
Expected: prints the long list of timestamped dirs from `git status` at start of session (div2k_8q_*, div2k_10q_*, div2k_8q_blocked_*, etc.).

- [ ] **Step 2: Move each to _archive/ via git mv**

For every dir from Step 1 except the freshly-trained `div2k_10q_generalized_<NEW>` and `quickdraw_generalized_<NEW>` (we'll move those too — they go into `_archive/` like the others, since the published cells are extracted *copies*):

```bash
for d in results/div2k_8q_* results/div2k_10q_* results/quickdraw_* ; do
    [ -d "$d" ] || continue
    base=$(basename "$d")
    git mv "$d" "results/_archive/${base}"
done
git mv results/run_*.log results/_archive/ 2>/dev/null || true
```

- [ ] **Step 3: Verify**

```bash
ls results/_archive/ | head -10
ls results/ | head -10
```
Expected: timestamped dirs are now under `_archive/`; top-level `results/` only contains `_archive/`, `published/`, `ablations/`.

- [ ] **Step 4: Commit**

```bash
git status | head -20
git commit -m "chore(results): archive raw timestamped runs under _archive/"
```

---

### Task 20: Move ablations into `results/ablations/`

**Files:**
- `git mv` 4 specific groups out of `_archive/` into `results/ablations/<subdir>/`.

- [ ] **Step 1: rich_init group**

```bash
git mv results/_archive/div2k_8q_rich_DCTINIT_20260426-111726 results/ablations/rich_init/DCTINIT
git mv results/_archive/div2k_8q_rich_DENSE_20260426-113814 results/ablations/rich_init/DENSE
git mv results/_archive/div2k_8q_rich_DENSE_DCTINIT_20260426-113814 results/ablations/rich_init/DENSE_DCTINIT
git mv results/_archive/div2k_8q_rich_LONG_20260426-111726 results/ablations/rich_init/LONG
```

- [ ] **Step 2: stacked_depth group**

```bash
git mv results/_archive/div2k_8q_blocked_stacked_20260426-102601_K2 results/ablations/stacked_depth/K2
git mv results/_archive/div2k_8q_blocked_stacked_20260426-102601_K3 results/ablations/stacked_depth/K3_short
git mv results/_archive/div2k_8q_blocked_stacked_20260426-103547_K3 results/ablations/stacked_depth/K3_long
```

- [ ] **Step 3: batch_size group**

```bash
git mv results/_archive/div2k_8q_blocked_generalized_20260426-093846_bs4  results/ablations/batch_size/bs4
git mv results/_archive/div2k_8q_blocked_generalized_20260426-093846_bs16 results/ablations/batch_size/bs16
git mv results/_archive/div2k_8q_blocked_generalized_20260426-093846_bs32 results/ablations/batch_size/bs32
git mv results/_archive/div2k_8q_blocked_generalized_20260426-093846_bs64 results/ablations/batch_size/bs64
```

- [ ] **Step 4: learned_vs_dct_block group**

```bash
git mv results/_archive/div2k_8q_DCT_20260426-123029 results/ablations/learned_vs_dct_block/DCT
```

- [ ] **Step 5: Verify**

```bash
ls results/ablations/rich_init/ results/ablations/stacked_depth/ results/ablations/batch_size/ results/ablations/learned_vs_dct_block/
```
Expected: each subdir has its expected children.

- [ ] **Step 6: Commit**

```bash
git commit -m "chore(results): organize ablations under results/ablations/"
```

---

### Task 21: Write ablation README files

**Files:**
- Create: `results/ablations/rich_init/README.md`
- Create: `results/ablations/stacked_depth/README.md`
- Create: `results/ablations/batch_size/README.md`
- Create: `results/ablations/learned_vs_dct_block/README.md`
- Create: `results/_archive/README.md`

Each is a small, focused note: what varied, what stayed fixed, headline finding, control cell.

- [ ] **Step 1: Write `results/ablations/rich_init/README.md`**

```markdown
# Ablation: RichBasis initialization

**Question:** does the choice of RichBasis init scheme materially affect convergence at DIV2K-8q?

**Varied:** init scheme — `DCTINIT`, `DENSE`, `DENSE_DCTINIT`, `LONG`.

**Fixed:** dataset (DIV2K-8q, m=n=8), preset (`generalized`), all other hyperparameters.

**Control cell:** `results/published/div2k_8q__rich/` — the canonical RichBasis with the registry-default init.

**Subdirs:**
- `DCTINIT/` — init from a DCT-truncated basis.
- `DENSE/` — init dense random.
- `DENSE_DCTINIT/` — DCTINIT then densified.
- `LONG/` — same as the canonical run but trained for more epochs (sanity).

Compare `metrics.json` per subdir against the control to read off the headline numbers.
```

- [ ] **Step 2: Write `results/ablations/stacked_depth/README.md`**

```markdown
# Ablation: stacked-blocked depth (K)

**Question:** does deeper block stacking (K>1 within-block circuit repetitions) improve quality?

**Varied:** K — 2, 3 (short schedule), 3 (long schedule).

**Fixed:** dataset (DIV2K-8q, m=n=8), inner basis = `BlockedBasis` family.

**Control cell:** `results/published/div2k_8q__blocked/` — the canonical K=1.

**Subdirs:**
- `K2/` — K=2.
- `K3_short/` — K=3, short schedule.
- `K3_long/` — K=3, long schedule.
```

- [ ] **Step 3: Write `results/ablations/batch_size/README.md`**

```markdown
# Ablation: batch size sweep (DIV2K-8q blocked)

**Question:** how does batch size affect throughput vs accuracy on the canonical 8q blocked configuration?

**Varied:** batch_size ∈ {4, 16, 32, 64}.

**Fixed:** dataset (DIV2K-8q), basis = `blocked`, preset = `generalized`, all other hyperparameters.

**Control cell:** `results/published/div2k_8q__blocked/` — the canonical batch_size=8.

**Subdirs:** `bs4/`, `bs16/`, `bs32/`, `bs64/`.
```

- [ ] **Step 4: Write `results/ablations/learned_vs_dct_block/README.md`**

```markdown
# Ablation: BlockedBasis with DCT inner ≈ block_dct_8 baseline

**Question:** sanity check — when the inner of a BlockedBasis is fixed to a DCT, does the trained outcome reduce to the classical `block_dct_8` baseline?

**Fixed:** DIV2K-8q, generalized preset.

**Control:** the `block_dct_8` row inside `results/published/div2k_8q__blocked/metrics.json`.

**Subdir:** `DCT/` — full run with `blocked_dct` (BlockedBasis(inner=DCT)).
```

- [ ] **Step 5: Write `results/_archive/README.md`**

```markdown
# results/_archive/

Raw timestamped run directories preserved for provenance. These are the
**sources** that `results/published/` cells were extracted from, and the
originals of `results/ablations/`. Not part of the publication; not
maintained.

Refer to `results/published/MANIFEST.json` for canonical results, and to
the ablation `README.md` files for ablation context.
```

- [ ] **Step 6: Commit**

```bash
git add results/ablations/*/README.md results/_archive/README.md
git commit -m "docs(results): add ablation + archive README files"
```

---

### Task 22: Run the extractor — populate `results/published/`

**Files:**
- Generated: `results/published/<dataset>__<basis>/` × 21 (19 active + 2 SKIPPED).

- [ ] **Step 1: Run the extractor**

```bash
python scripts/extract_canonical_cells.py
```

Expected output: 19 lines of `extract <cell>  ←  <source>` and 2 lines of `skipped <cell>`. No `SKIP ... — source missing` messages — if any appear, the extraction table has a stale path; go back to Task 17.

- [ ] **Step 2: Verify cell contents**

```bash
ls results/published/ | wc -l   # expect 21
ls results/published/div2k_8q__qft/   # expect metrics.json, env.json, config.json, trained_qft.json, loss_history/, plots/, rate_distortion_*.csv, timing_summary.csv, run.log
cat results/published/div2k_10q__mera/SKIPPED.json
```

Expected: each active cell has the full file set; each SKIPPED cell has only `SKIPPED.json`.

- [ ] **Step 3: Commit**

```bash
git add results/published/
git commit -m "feat(results): populate results/published/ — 21 canonical cells"
```

---

### Task 23: Build initial MANIFEST.json

**Files:**
- Create: `results/published/MANIFEST.json`

Use a one-shot inline script invocation that calls `pdft_benchmarks._manifest.build_manifest`.

- [ ] **Step 1: Build and write the manifest**

```bash
python -c "
import json, subprocess
from pathlib import Path
import pdft
from pdft_benchmarks._manifest import build_manifest

git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
m = build_manifest(Path('results/published'),
                   git_sha=git_sha,
                   pdft_version=pdft.__version__)
Path('results/published/MANIFEST.json').write_text(json.dumps(m, indent=2))
print(f'wrote MANIFEST with {len(m[\"cells\"])} cells')
"
```

Expected output: `wrote MANIFEST with 21 cells`.

- [ ] **Step 2: Spot-check the file**

```bash
python -c "
import json
m = json.loads(open('results/published/MANIFEST.json').read())
statuses = [c['status'] for c in m['cells']]
print('active:', statuses.count('active'), 'skipped:', statuses.count('skipped'))
print('schema:', m['schema_version'], 'pdft:', m['pdft_version'])
"
```
Expected: `active: 19 skipped: 2` and `schema: 1.0 pdft: 0.2.1`.

- [ ] **Step 3: Commit**

```bash
git add results/published/MANIFEST.json
git commit -m "feat(results): MANIFEST.json — 19 active + 2 skipped cells"
```

---

### Task 24: Write `results/published/README.md` skeleton

**Files:**
- Create: `results/published/README.md`

The skeleton has all sections from Appendix A of the spec, with placeholder `{BEGIN HEADLINE NUMBERS}` markers between which Task 25's renderer will insert the headline-numbers table.

- [ ] **Step 1: Write the README**

Path: `results/published/README.md`

```markdown
# pdft Benchmarks — Published Results

Canonical results for the pdft 7-basis suite across three datasets, frozen for
publication. Each subdirectory is one *(dataset, basis)* cell; the matrix is
described by `MANIFEST.json` at this level.

## Citation

If you use these results, please cite:

> [paper / preprint citation TBD]

The artifact is also archived at: [Zenodo DOI TBD].

## The matrix at a glance

|                | qft | entangled_qft | tebd | mera     | blocked | rich | real_rich |
|----------------|-----|---------------|------|----------|---------|------|-----------|
| **div2k_8q**   | ✓   | ✓             | ✓    | ✓        | ✓       | ✓    | ✓         |
| **div2k_10q**  | ✓   | ✓             | ✓    | skipped¹ | ✓       | ✓    | ✓         |
| **quickdraw**  | ✓   | ✓             | ✓    | skipped¹ | ✓       | ✓    | ✓         |

¹ MERA requires `m+n` to be a power of 2. Both `div2k_10q` (m+n=20) and
`quickdraw` (m+n=10) violate this; cells contain only `SKIPPED.json`.

## Headline numbers (PSNR @ keep ratio 0.10, dB)

<!-- BEGIN HEADLINE NUMBERS (auto-generated; do not edit) -->
(populated by scripts/render_published_readme.py)
<!-- END HEADLINE NUMBERS -->

## What's in each cell

See `<dataset>__<basis>/`:

- `metrics.json` — bit-compatible with the upstream Julia schema; this
  basis + 4 classical baselines (`fft`, `dct`, `block_fft_8`, `block_dct_8`).
- `config.json` — frozen training config.
- `env.json` — git sha, JAX version, device, pdft version, dataset hash.
- `trained_<basis>.json` — all `n_train` trained bases.
- `loss_history/<basis>_loss.json` — list-of-lists, one row per image.
- `rate_distortion_{mse,psnr,ssim}.csv` — per keep-ratio reconstruction quality.
- `timing_summary.csv` — wall-clock per phase.
- `plots/*.pdf` — vector plots for this cell.
- `run.log` — captured stdout/stderr.

Skipped cells contain only `SKIPPED.json`.

## Reproducing

These results were generated with:

- `pdft` v0.2.1 (https://pypi.org/project/pdft/0.2.1/)
- This repo at git sha — see per-cell `env.json` for exact.
- DIV2K from the official train HR set; QuickDraw from the official 5-category subset.

Re-derive a single cell:

    pip install "pdft==0.2.1"
    pip install -e ".[bench,gpu]"
    python experiments/<dataset>_<group>.py --gpu 0

Re-derive all canonical cells (~3 hours on 1 GPU):

    bash scripts/run_canonical.sh

## Directory map

    results/
    ├── published/      ← the paper's results
    ├── ablations/      ← supplementary studies
    └── _archive/       ← raw timestamped runs (provenance)

## Versioning

- `MANIFEST.json` `schema_version`: 1.0
- These results are immutable. New runs go in *new* cell directories
  with bumped MANIFEST entries; old cells are not modified in place.

## Contact

Issues, corrections, or questions: open an issue at the repo URL listed in `pyproject.toml`.
```

- [ ] **Step 2: Commit**

```bash
git add results/published/README.md
git commit -m "docs(results): published/README.md skeleton"
```

---

### Task 25: Render the headline-numbers table

**Files:**
- Modify: `results/published/README.md` (between markers).

- [ ] **Step 1: Run the renderer**

```bash
python scripts/render_published_readme.py
```
Expected: exits 0, no output on stdout.

- [ ] **Step 2: Verify the table is in place**

```bash
grep -A 6 "BEGIN HEADLINE NUMBERS" results/published/README.md | head -10
```
Expected: shows a markdown table with three rows (`div2k_8q`, `div2k_10q`, `quickdraw`) and PSNR cells filled in.

- [ ] **Step 3: Commit**

```bash
git add results/published/README.md
git commit -m "docs(results): render headline-numbers table from MANIFEST"
```

---

### Task 26: Validate

**Files:**
- Read-only over `results/published/`.

- [ ] **Step 1: Run validator**

```bash
python scripts/validate_manifest.py
```
Expected: prints `validate_manifest: OK (results/published)`, exits 0.

If validation fails: read the error, identify the discrepancy, fix in place (re-run extractor for affected cell, or re-run MANIFEST builder), commit the fix.

- [ ] **Step 2: Run the existing test suite**

```bash
pytest tests/ --no-cov -x
```
Expected: all tests pass. The new tests added in Tasks 1-9 plus the existing ones.

- [ ] **Step 3: No commit needed if everything passed.**

---

## Phase G — Wrap-up

### Task 27: Update repo README

**Files:**
- Modify: `README.md` (top-level)

The current top-level README references `run_quickdraw.py` and `run_div2k_8q.py` which don't exist (they were renamed to `experiments/<...>.py` in commit `e2da58a`). Patch the README to point at `experiments/quickdraw.py` and `experiments/div2k_10q_block.py` and add a short "Published results" section.

- [ ] **Step 1: Add "Published results" section + fix command paths**

Add the following section under `## Run`:

```markdown
## Published results

Canonical results for the 7-basis × 3-dataset matrix live under
`results/published/`. See `results/published/README.md` and
`results/published/MANIFEST.json` for the full breakdown.

To re-derive everything from scratch:

    bash scripts/run_canonical.sh
```

Replace the broken commands in the existing `## Run` section with the actual `experiments/<...>.py` paths.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README — fix run commands, point at results/published/"
```

---

## Self-review

Run through this list against the spec:

**Spec coverage:**

- ✅ 7-basis registry → Tasks 1, 5 reference `BASIS_FACTORIES` keys.
- ✅ 3-dataset table (m,n,preset) → Tasks 5, 12, 13.
- ✅ Per-cell file contract → Task 4 + Task 8 (validator's `_REQUIRED_ACTIVE_FILES`).
- ✅ SKIPPED cell contract → Task 4 (`write_skipped_cell`) + validator (Task 8).
- ✅ MANIFEST.json schema → Tasks 5-7.
- ✅ Validator → Task 8.
- ✅ Ablations curation → Task 20.
- ✅ Source extraction table → Task 11.
- ✅ Run plan for missing cells (10q block + quickdraw) → Tasks 12, 13, 15, 16.
- ✅ Cleanup procedure → Tasks 18-20.
- ✅ README + headline-numbers renderer → Tasks 9, 24, 25.
- ✅ run_canonical.sh → Task 14.
- ✅ Reproducibility (pdft version pin, git sha in env.json) → already in pipeline; preserved verbatim by extractor.

**Placeholder scan:** Searched for "TBD", "TODO", "fill in" — only intentional `[paper / preprint citation TBD]` and `[Zenodo DOI TBD]` strings inside the README skeleton (Task 24), inherited from the spec. The `_NEW` strings in the extraction table (Task 11) are intentional placeholders, replaced by Task 17 with the actual run-dir names.

**Type consistency:**
- `extract_cell(...)` signature is the same in Task 4's tests, Task 11's CLI, and Task 14's `run_canonical.sh`.
- `validate_manifest(published_root)` is the same in Task 8's tests and Task 10's CLI.
- `summarize_metrics(metrics, basis_key=...)` consistent across Tasks 6, 7, 8.
- `rename_basis_key(source_key)` consistent across Tasks 1, 2, 4.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-27-publishable-benchmark-results.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Each Phase A-D task is self-contained TDD; Phases E-G are sequential with one user pause point (Tasks 15-16, GPU runs).

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Same pause point at Tasks 15-16.

Which approach?
