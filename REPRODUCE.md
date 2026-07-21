# Reproducing the paper

This branch (`main`) carries only the code and data behind the figures and
tables in the companion paper. The full research tree — block-size sweeps,
QFT progressive/top-k studies, DCT-IV sweep training, profiling — lives on
`dev`.

Every figure below regenerates from the committed `results/` tree with no
retraining. The training commands are listed for provenance; you only need them
to rebuild a results cell from scratch.

## Environment

```bash
python -m venv .venv --system-site-packages
.venv/bin/pip install -e .
```

Datasets are not downloaded automatically; the loaders raise if absent.
DIV2K-HR (`0001.png`…`0800.png`) and the QuickDraw `numpy_bitmap` categories go
under the paths in `src/pdft_benchmarks/datasets/`.

All trained bases use the headline budget **`--epochs 112 --no-early-stop`**
(1008 steps; batch 50, n_train 500, val_split 0.15 → 9 steps/epoch). Runs are
cellified into `results/structure/<experiment>/by_basis/` via
`tools/cellify_run.py`. Figures are emitted as **PDF + SVG, never PNG**.

Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` alongside `CUDA_VISIBLE_DEVICES` — on mixed-GPU
hosts CUDA's default order is fastest-first, not PCI order, so `--gpu N` can
otherwise land on a different card than `nvidia-smi`'s GPU N.

## Figures

Paths in the first column are as referenced from the paper's `main.tex`.

| Paper artifact | Render command | Output in this repo |
|---|---|---|
| `structure/topology_loss_curve.pdf` (`fig:topology_loss`) | `python tools/paper/render_topology_loss.py` | `results/structure/div2k_8q_pca_vs_block_dct/figures/topology_loss_curve.{pdf,svg}` |
| `freqspace/quickdraw_all8_imgcat{,_freq}.pdf` (`fig:freqrecon_compact`) | `python tools/paper/render_freq_recon_grid.py --dataset quickdraw --image-indices 0` | `results/structure/quickdraw_pca_vs_block_dct/figures/freq_recon_grid_img0{,_freq}.{pdf,svg}` |
| `freqspace/div2k_all8_img390{,_freq}.pdf` (`fig:freqrecon_compact`) | `python tools/paper/render_freq_recon_grid.py --dataset div2k_8q --image-indices 11 --div2k-source-indices 390` | `results/structure/div2k_8q_pca_vs_block_dct/figures/freq_recon_grid_img390{,_freq}.{pdf,svg}` |
| `compression/rd_quickdraw.pdf` (`fig:rd_quickdraw`) | `python tools/paper/render_paper_compression_rd.py` | `results/training/6_dataset_compression/quickdraw_5q/figures/rd_quickdraw_paper.{pdf,svg}` |
| `direct_training/training_dynamics.pdf` (`fig:app_unfreeze_dynamics`) | `python tools/analysis/render_qft_unfreeze.py --combined --paper-style` | `results/training/2_direct_training/unfreeze/figures/paper/training_dynamics.pdf` |
| `direct_training/init_distribution.pdf` (`fig:app_seed_robustness_a`) | `python tools/analysis/render_init_distribution.py --base results/training/2_direct_training/random_seed/div2k_8q --from-json --paper-style` | `…/random_seed/div2k_8q/figures/paper/init_distribution.pdf` |
| `direct_training/seed_scatter_ratios.pdf` (`fig:app_seed_robustness_b`) | `python tools/analysis/render_seed_scatter_ratios.py --base results/training/2_direct_training/random_seed/div2k_8q --paper-style` | `…/random_seed/div2k_8q/figures/paper/seed_scatter_ratios.pdf` |
| `exact_disturbance/disturbance_psnr_vs_f.pdf` (`fig:disturbance_psnr`) | `python tools/analysis/render_disturbance_curve.py` | `results/training/4_exact_disturbance/figures/disturbance_psnr_vs_f.{pdf,svg}` |
| `exact_disturbance/disturbance_init_loss.pdf` (`fig:disturbance_loss`) | `python tools/analysis/render_disturbance_curve.py` | `results/training/4_exact_disturbance/figures/disturbance_init_loss.{pdf,svg}` |
| `exact_disturbance/disturbance_recovery.pdf` (`fig:disturbance_recovery`) | `python tools/analysis/render_disturbance_curve.py` | `results/training/4_exact_disturbance/figures/disturbance_recovery.{pdf,svg}` |

### Authored in the paper repository, not here

These have no benchmark provenance — they are hand-drawn typst diagrams or a
composited banner, built by the paper's own `make diagrams` / `make banner`.

| Paper artifact | Source |
|---|---|
| `figures/banner-main-1x3.pdf` (`fig:banner`) | `figures/banner-main-1x3.typ` + `figures/assets/*.png` |
| `topology_gallery.pdf` (`fig:topology_circuits`) | `scripts/diagrams/topology_gallery.typ` |
| `cooley_tukey_to_qft.pdf` (`fig:cooley_tukey_to_qft`) | `scripts/diagrams/cooley_tukey_to_qft.typ` |
| `cooley_tukey_to_dct.pdf` (`fig:cooley_tukey_to_dct`) | `scripts/diagrams/cooley_tukey_to_dct.typ` |
| `qft_unfreeze_circuit.pdf` (`fig:app_circuit`) | `scripts/diagrams/qft_unfreeze_circuit.typ` |
| `ar1_histogram.pdf` (`fig:ar1_histogram`) | paper's `scripts/plot_ar1_histogram.py` — imports `pdft_benchmarks.datasets` loaders from this repo, but writes into the paper tree |

## Tables

| Paper artifact | Render command | Output in this repo |
|---|---|---|
| `tab:div2k_repr` / `tab:quickdraw_repr` — mean test PSNR, both datasets | `python tools/paper/render_div2k_paper_table.py`<br>`python tools/paper/render_paper_table.py` | `results/structure/div2k_8q_pca_vs_block_dct/tables/published_8q_div2k.tex`<br>`results/structure/quickdraw_pca_vs_block_dct/tables/published_8q_quickdraw.tex` |
| `tab:app_seed_variance` — per-ordering seed robustness | `python tools/analysis/render_seed_variance_table.py --base results/training/2_direct_training/random_seed/div2k_8q` | `…/random_seed/div2k_8q/tables/seed_variance.tex` |
| `tab:disturbance` — disturbance sweep | `python tools/analysis/render_disturbance_curve.py` | `results/training/4_exact_disturbance/tables/disturbance_psnr.tex` |

The paper's numbers are hand-formatted into `main.tex`; the files above are the
numeric provenance, not a direct `\input`.

### Hand-written, no benchmark provenance

`tab:circuits` (four circuit variants), `tab:hyperparams` (benchmark setup), and
`tab:gate_relaxations` (gate-wise relaxations) are authored directly in
`main.tex`.

## Training commands

Only needed to rebuild a results cell from scratch.

| Results tree | Command |
|---|---|
| `results/structure/quickdraw_pca_vs_block_dct/` | `python experiments/paper/quickdraw_pca_vs_block_dct.py --gpu 0 --epochs 112 --no-early-stop` |
| `results/structure/div2k_8q_pca_vs_block_dct/` | `python experiments/paper/div2k_8q_pca_vs_block_dct.py --gpu 0 --bases qft,entangled_qft,tebd,mera --epochs 112 --no-early-stop`<br>`python experiments/paper/div2k_8q_pca_vs_block_dct.py --gpu 1 --bases blocked_8,rich_8,real_rich_8 --epochs 112 --no-early-stop` |
| `results/training/1_structure_inclusion/` | `python experiments/qft/qft_structure_inclusion.py` |
| `results/training/2_direct_training/random_seed/` | `python experiments/qft/qft_seed_sweep.py` |
| `results/training/2_direct_training/unfreeze/` | `python experiments/qft/qft_freeze_sweep.py` |
| `results/training/4_exact_disturbance/` | `python tools/run_dct4_disturbance_sweep.py` (drives `experiments/dct4/dct4_disturbance_sweep.py`) |
| `results/training/6_dataset_compression/` | `python experiments/misc/dataset_compression.py` |

The DIV2K driver isolates its GPU with `CUDA_VISIBLE_DEVICES` set **before** any
`pdft_benchmarks` import; QuickDraw uses JAX device selection instead.

## Retained but not currently cited

These renderers work against the committed tree and are kept because the paper
has drawn on them across revisions, but no figure in the current `main.tex`
includes their output:

| Renderer | Produces |
|---|---|
| `tools/paper/render_loss_curves.py --dataset {quickdraw,div2k_8q}` | `figures/loss_curves.{pdf,svg}` + the `loss_curve_{500,1000,2000}` budget snapshots |
| `tools/paper/render_ar1_examples.py` | `figures/ar1_examples.{pdf,svg}` — AR(1) patch examples, distinct from the paper's `ar1_histogram.pdf` |
| `tools/paper/render_pca_basis_visualization.py --dataset {…}` | `figures/pca_basis.{pdf,svg}` |
| `tools/analysis/render_seed_dynamics.py --base …` | `figures/paper/seed_training_dynamics.pdf` |

## Verification

Independent classical-baseline reruns (no training) confirm the committed
`_baselines.json` against a fresh computation:

```bash
python tools/independent_quickdraw_baselines.py --gpu 0 --seed 42 --n-train 500
python tools/independent_div2k_8q_baselines.py --gpu 0 --seed 42 --n-train 500
```

Test suite:

```bash
python -m pytest -q -m "not integration and not slow"
```
