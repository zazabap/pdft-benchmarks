# Reproducing the paper

This branch (`main`) carries only the code and data behind the figures and
tables in the companion paper. The full research tree lives on `dev`.

**Every paper section has one `make` target**, `make 00` through `make 05`.
By default each renders from committed data (`results/`, and for the two
appendix sections, `data/`) with no retraining; every section also prints its
headline numbers to stdout as part of `verify()`, so you can spot-check the
match without opening the PDF:

```bash
make            # usage
make 02         # one section
make all        # every section, in order 00..05
```

Retraining is opt-in and needs a GPU + the datasets:

```bash
RETRAIN=1 make 02     # train/sweep -> cellify -> render, then the same output
```

## Environment

```bash
python -m venv .venv --system-site-packages
.venv/bin/pip install -e .          # pulls pdft from its git v0.2.3 tag
```

`pdft` is pinned to its git tag **`v0.2.3`**: the published PyPI `0.2.2` predates
`DCT4Basis` and the U(4) `tebd_u4`/`mera_u4` gates this repo needs, so an older
pdft fails at import or silently trains a different basis.

Datasets are not downloaded automatically; the loaders raise if absent.
DIV2K-HR (`0001.png`…`0800.png`) and the QuickDraw `numpy_bitmap` categories go
under the paths in `src/pdft_benchmarks/datasets/`. `make 00`, `01`, and `02`
need a dataset at render time (Fig 3 / Fig 5 compute directly from pixels);
without it, `01`/`02` still render Table 3 and print a skip note for the
image-derived panel. `03`, `04`, and `05` render entirely from committed JSON
under `results/` / `data/` — no dataset needed.

The six Quick Draw category files are **not** Google's full release: each is
exactly the first 3000 rows of the corresponding public `numpy_bitmap` file,
so the frozen split manifest's row indices resolve against these 3000-row
pools. Derive them with the snippet below (a range request fetches only the
needed ~2.3 MB per category), then verify against the committed checksums:

```bash
python - <<'EOF'
import pathlib, struct, urllib.request
import numpy as np

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
# Loader default; see README "Datasets" to point elsewhere.
DEST = pathlib.Path("/home/claude-user/ParametricDFT-Benchmarks.jl/data/quickdraw")
DEST.mkdir(parents=True, exist_ok=True)
for cat in ["airplane", "apple", "banana", "car", "cat", "clock"]:
    url = f"{BASE}/{cat}.npy"
    head = urllib.request.urlopen(
        urllib.request.Request(url, headers={"Range": "bytes=0-9"})).read()
    hlen = 10 + struct.unpack("<H", head[8:10])[0]  # .npy v1 total header size
    rng = f"bytes={hlen}-{hlen + 3000 * 784 - 1}"
    raw = urllib.request.urlopen(
        urllib.request.Request(url, headers={"Range": rng})).read()
    np.save(DEST / f"{cat}.npy",
            np.frombuffer(raw, np.uint8).reshape(3000, 784))
    print("wrote", DEST / f"{cat}.npy")
EOF
(cd /home/claude-user/ParametricDFT-Benchmarks.jl/data/quickdraw \
    && sha256sum -c "$OLDPWD"/results/structure/quickdraw_pool.sha256)
```

On mixed-GPU hosts the Makefile already exports `CUDA_DEVICE_ORDER=PCI_BUS_ID`
and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Figures and tables

The full map — each paper figure/table → its `make 0X` target → the section
script → the output file — is the table in the [README](README.md). Every
target renders from committed data with no retraining; `00`, `01`, and `02`
need the image datasets for their image-derived panels. Outputs are
**PDF + SVG, never PNG**.

The headline training budget behind the `RETRAIN=1` path is
**`--epochs 112 --no-early-stop`** (1008 steps). Its keep-ratio grid includes
ρ=0.01 so the headline table's ρ=0.01 column reproduces.

## Table 2 (the 500/100 evaluation with uncertainty)

The paper's Table 2 evaluates every method on **100** held-out images (the
historical 50 plus 50 disjoint extensions, identities preserved) at keep
ratios up to ρ=0.40. It is not a `make` target; it regenerates with:

```bash
JAX_PLATFORMS=cpu .venv/bin/python tools/reeval_table2_uncertainty.py
```

This rebuilds the frozen seed-42 split (`--split-only` re-derives it in
seconds and must reproduce the committed `split_sha256`), re-scores each
committed checkpoint / classical baseline per image, **validates the
first-50 means against the committed cell metrics** before assembling, and
writes:

- `results/structure/table2_500x100_uncertainty.json` — the committed
  provenance record: split manifest (SHA-256 per split), per-image PSNRs,
  mean ± SEM per cell, and paired-bootstrap 95% CIs;
- `results/structure/table2_500x100.tex` — the paper's Table 2, byte-for-byte.

The full run needs both datasets and a few CPU-hours; per-method parts are
cached under `results/structure/table2_500x100_parts/` (gitignored) so it is
resumable. `tools/reeval_topk_tables.py` is the companion protocol
cross-check: it re-evaluates the stored bases at the extended grid (adds
ρ=0.4) on the current data snapshot and reports drift against each committed
cell.

### Authored in the paper repository, not here

These have no benchmark provenance — hand-drawn typst diagrams or a composited
banner, built by the paper's own `make diagrams` / `make banner`:

| Paper artifact | Source (paper repo) |
|---|---|
| `banner-main-1x3.pdf` (`fig:banner`) | `figures/banner-main-1x3.typ` |
| `topology_gallery.pdf` (`fig:topology_circuits`) | `scripts/diagrams/topology_gallery.typ` |
| `cooley_tukey_to_qft.pdf` (`fig:cooley_tukey_to_qft`) | `scripts/diagrams/cooley_tukey_to_qft.typ` |
| `cooley_tukey_to_dct.pdf` (`fig:cooley_tukey_to_dct`) | `scripts/diagrams/cooley_tukey_to_dct.typ` |
| `qft_unfreeze_circuit.pdf` (`fig:app_circuit`) | `scripts/diagrams/qft_unfreeze_circuit.typ` |

`tab:circuits` (Table 1) and `tab:gate_relaxations` (Table 3) are authored
directly in the paper's `main.tex`. The training hyperparameters that a
previous revision tabulated in the paper are recorded per run in each
released cell's `env.json` (`epochs: 112`, `batch_size: 50`,
`lr_peak: 3e-3` → `lr_final: 3e-4`, `warmup_frac: 0.05`,
`max_grad_norm: 1.0`, `validation_split: 0.15`, `seed: 42`).

**Fig 3 (`ar1_histogram.pdf`, `fig:ar1_histogram`) moved here.** It used to be
paper-authored (`scripts/plot_ar1_histogram.py`, importing this repo's dataset
loaders but writing into the paper tree); it's now generated by `make 00`
(`experiments/00_dataset_dist.py`), writing
`results/dataset_dist/figures/ar1_histogram.{pdf,svg}`. The paper repo copies
the PDF from here rather than generating it.

## Verification

Independent classical-baseline reruns (no training) confirm the committed
`_baselines.json`:

```bash
python tools/independent_quickdraw_baselines.py --gpu 0 --seed 42 --n-train 500
python tools/independent_div2k_8q_baselines.py --gpu 0 --seed 42 --n-train 500
```

Test suite:

```bash
python -m pytest -q -m "not integration and not slow"
```
