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
.venv/bin/pip install -e .          # pulls pdft>=0.2.2 (DCT4Basis + U(4) gates)
```

`pdft>=0.2.2` is required: older releases lack `DCT4Basis` and the U(4)
`tebd_u4`/`mera_u4` parametrization and will fail at import or silently train a
different basis.

Datasets are not downloaded automatically; the loaders raise if absent.
DIV2K-HR (`0001.png`…`0800.png`) and the QuickDraw `numpy_bitmap` categories go
under the paths in `src/pdft_benchmarks/datasets/`. Only `make fig5` needs a
dataset at render time; every other target renders from committed JSON.

On mixed-GPU hosts the Makefile already exports `CUDA_DEVICE_ORDER=PCI_BUS_ID`
and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Figures and tables

Paper labels in parentheses. Outputs land under `results/…/figures/` or
`results/…/tables/` and are emitted as **PDF + SVG, never PNG**.

| Target | Paper artifact | Renders from |
|---|---|---|
| `make fig4` | topology loss curve (`fig:topology_loss`) | committed `loss_history/` |
| `make fig5` | freq/recon grids, both datasets (`fig:freqrecon_compact`) | committed `trained_*.json` **+ datasets** |
| `make fig6` | Quick Draw rate-distortion (`fig:rd_quickdraw`) | committed `rd_curves.json` |
| `make fig10` | QFT unfreeze dynamics (`fig:app_unfreeze_dynamics`) | committed |
| `make fig11` | seed robustness a+b (`fig:app_seed_robustness`) | committed |
| `make fig12-14` | DCT-IV exact disturbance (`fig:disturbance_{psnr,loss,recovery}`) | committed |
| `make table3` | headline PSNR, both datasets (`tab:div2k_repr`, `tab:quickdraw_repr`) | committed cells |
| `make table5` | per-ordering seed variance (`tab:app_seed_variance`) | committed |
| `make table6` | disturbance sweep (`tab:disturbance`) | committed |

The headline training budget behind the RETRAIN path is
**`--epochs 112 --no-early-stop`** (1008 steps). The keep-ratio grid includes
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
