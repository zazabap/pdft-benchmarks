# Reproducing the paper

This branch (`main`) carries only the code and data behind the figures and
tables in the companion paper. The full research tree lives on `dev`.

**Every artifact has one `make` target.** By default each renders from the
committed `results/` tree with no retraining; several also print their
headline numbers to stdout, so you can spot-check the match without opening
the PDF:

```bash
make            # usage
make fig4       # one artifact
make figures    # all figures
make tables     # all tables
make all        # everything
```

Retraining is opt-in and needs a GPU + the datasets:

```bash
RETRAIN=1 make fig4     # train -> cellify -> render, then the same output
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
under the paths in `src/pdft_benchmarks/datasets/`. Only `make fig5` needs a
dataset at render time; every other target renders from committed JSON.

On mixed-GPU hosts the Makefile already exports `CUDA_DEVICE_ORDER=PCI_BUS_ID`
and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Figures and tables

The full map — each paper figure/table → its `make` target → the generator
script → the output file — is the table in the [README](README.md). Every
target renders from the committed `results/` tree with no retraining; only
`make fig5` needs the image datasets. Outputs are **PDF + SVG, never PNG**.

The headline training budget behind the `RETRAIN=1` path is
**`--epochs 112 --no-early-stop`** (1008 steps). Its keep-ratio grid includes
ρ=0.01 so the headline table's ρ=0.01 column reproduces.

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
| `ar1_histogram.pdf` (`fig:ar1_histogram`) | paper's `scripts/plot_ar1_histogram.py` — imports this repo's dataset loaders but writes into the paper tree |

`tab:circuits`, `tab:hyperparams`, and `tab:gate_relaxations` are authored
directly in the paper's `main.tex`.

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
