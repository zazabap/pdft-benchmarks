# Reproduce the paper's figures and tables — one target per artifact.
#
# Default path: render from the committed results/ tree (no training).
#   make fig4                 # render one artifact
#   make figures tables       # all figures / all tables
#   make all                  # everything
#
# Retrain first (needs a GPU + the datasets), then render:
#   RETRAIN=1 make fig4
#
# figN / tableN  ->  paper \label  ->  renderer
#   fig4      fig:topology_loss                    tools/paper/render_topology_loss.py
#   fig5      fig:freqrecon_compact                tools/paper/render_freq_recon_grid.py (x2)
#   fig6      fig:rd_quickdraw                      tools/paper/render_paper_compression_rd.py
#   fig10     fig:app_unfreeze_dynamics             tools/analysis/render_qft_unfreeze.py
#   fig11     fig:app_seed_robustness{_a,_b}        render_init_distribution.py + render_seed_scatter_ratios.py
#   fig12-14  fig:disturbance_{psnr,loss,recovery}  tools/analysis/render_disturbance_curve.py
#   table3    tab:{div2k,quickdraw}_repr            render_div2k_paper_table.py + render_paper_table.py
#   table5    tab:app_seed_variance                 tools/analysis/render_seed_variance_table.py
#   table6    tab:disturbance                       tools/analysis/render_disturbance_curve.py
#
# ar1_histogram (fig 3) and the typst circuit diagrams are built in the PAPER
# repo, not here — see REPRODUCE.md ("Authored in the paper repository").
#
# All targets are .PHONY entry points; renderers own their outputs, so there is
# no file-dependency graph — do not use `make -j`.

PYTHON ?= .venv/bin/python
# For any target that imports JAX (render_freq_recon_grid, the RETRAIN training
# runs); harmless for the pure-JSON renderers. CUDA's default order is
# fastest-first, not PCI order, on mixed-GPU hosts.
export CUDA_DEVICE_ORDER := PCI_BUS_ID
export XLA_PYTHON_CLIENT_PREALLOCATE := false

# Headline training budget (see README / REPRODUCE.md); only used by RETRAIN.
EPOCHS      ?= 112
KEEP_RATIOS := 0.01,0.05,0.10,0.15,0.20

DIV2K_DIR := results/structure/div2k_8q_pca_vs_block_dct
QD_DIR    := results/structure/quickdraw_pca_vs_block_dct
SEED_BASE := results/training/2_direct_training/random_seed/div2k_8q
CAT_IMG   := experiments/paper/assets/quickdraw-cat.png

# Scratch dir for RETRAIN staging (driver output before cellify); cleaned per run.
REPRO_TMP ?= /tmp/repro

# RETRAIN basis lists = committed cells intersect each driver's ALL_BASES.
# (div2k excludes the legacy `rich` cell; on quickdraw the driver silently
# skips mera because m+n=10 is not a power of two.)
DIV2K_BASES := qft,entangled_qft,tebd,mera,blocked_8,rich_8,real_rich_8,dct4_ctl,tebd_u4,mera_u4,rich_full
QD_BASES    := qft,entangled_qft,tebd,mera,blocked,rich,real_rich,dct4_ctl,tebd_u4,rich_full,real_rich_full

# RETRAIN=1 turns each figure's train helper into a prerequisite.
retrain_div2k       := $(if $(filter 1,$(RETRAIN)),train-div2k,)
retrain_structure   := $(if $(filter 1,$(RETRAIN)),train-div2k train-quickdraw,)
retrain_compression := $(if $(filter 1,$(RETRAIN)),train-compression,)
retrain_unfreeze    := $(if $(filter 1,$(RETRAIN)),train-unfreeze,)
retrain_seed        := $(if $(filter 1,$(RETRAIN)),train-seed,)
retrain_dist        := $(if $(filter 1,$(RETRAIN)),train-disturbance,)

.PHONY: help all figures tables \
        fig4 fig5 fig6 fig10 fig11 fig12-14 table3 table5 table6 \
        train-div2k train-quickdraw train-compression \
        train-unfreeze train-seed train-disturbance

.DEFAULT_GOAL := help

help:
	@echo "Reproduce paper artifacts (default: render from committed results/):"
	@echo "  figures:  make fig4 fig5 fig6 fig10 fig11 fig12-14"
	@echo "  tables:   make table3 table5 table6"
	@echo "  all:      make figures | make tables | make all"
	@echo "  retrain:  RETRAIN=1 make <target>   (needs a GPU + the datasets)"
	@echo "  python:   override with 'make PYTHON=python3 fig4'"

all: figures tables
figures: fig4 fig5 fig6 fig10 fig11 fig12-14
tables:  table3 table5 table6

# ------------------------------------------------------------------ figures
fig4: $(retrain_div2k)
	$(PYTHON) tools/paper/render_topology_loss.py

fig5: $(retrain_structure)
	$(PYTHON) tools/paper/render_freq_recon_grid.py --dataset quickdraw \
	    --custom-images $(CAT_IMG):cat --keep-ratios $(KEEP_RATIOS)
	$(PYTHON) tools/paper/render_freq_recon_grid.py --dataset div2k_8q \
	    --image-indices 11 --div2k-source-indices 390 --keep-ratios $(KEEP_RATIOS)

fig6: $(retrain_compression)
	$(PYTHON) tools/paper/render_paper_compression_rd.py --dataset quickdraw_5q

fig10: $(retrain_unfreeze)
	$(PYTHON) tools/analysis/render_qft_unfreeze.py --combined --paper-style

fig11: $(retrain_seed)
	$(PYTHON) tools/analysis/render_init_distribution.py --base $(SEED_BASE) \
	    --from-json --paper-style
	$(PYTHON) tools/analysis/render_seed_scatter_ratios.py --base $(SEED_BASE) \
	    --paper-style

fig12-14: $(retrain_dist)
	$(PYTHON) tools/analysis/render_disturbance_curve.py

# ------------------------------------------------------------------- tables
table3: $(retrain_structure)
	$(PYTHON) tools/paper/render_div2k_paper_table.py
	$(PYTHON) tools/paper/render_paper_table.py

table5: $(retrain_seed)
	$(PYTHON) tools/analysis/render_seed_variance_table.py --base $(SEED_BASE)

# render_disturbance_curve.py writes the three disturbance figures AND the
# disturbance table in one pass, so table6 reuses fig12-14 rather than running
# the renderer a second time under `make all`.
table6: fig12-14

# ---------------------------------------------------------------- retrain helpers
# Single-process, full basis set on one GPU. The two-GPU split in README is a
# speed optimization, not required for correctness.
train-div2k:
	rm -rf $(REPRO_TMP)/div2k
	$(PYTHON) experiments/paper/div2k_8q_pca_vs_block_dct.py --gpu 0 \
	    --bases $(DIV2K_BASES) --epochs $(EPOCHS) --no-early-stop \
	    --keep-ratios $(KEEP_RATIOS) --out $(REPRO_TMP)/div2k
	$(PYTHON) tools/cellify_run.py --in $(REPRO_TMP)/div2k \
	    --out $(DIV2K_DIR)/by_basis --bases $(DIV2K_BASES)

train-quickdraw:
	rm -rf $(REPRO_TMP)/quickdraw
	$(PYTHON) experiments/paper/quickdraw_pca_vs_block_dct.py --gpu 0 \
	    --bases $(QD_BASES) --epochs $(EPOCHS) --no-early-stop \
	    --keep-ratios $(KEEP_RATIOS) --out $(REPRO_TMP)/quickdraw
	$(PYTHON) tools/cellify_run.py --in $(REPRO_TMP)/quickdraw \
	    --out $(QD_DIR)/by_basis --bases $(QD_BASES)

train-compression:
	$(PYTHON) experiments/misc/dataset_compression.py --gpu 0

train-unfreeze:
	$(PYTHON) experiments/qft/qft_freeze_sweep.py

train-seed:
	$(PYTHON) experiments/qft/qft_seed_sweep.py

train-disturbance:
	$(PYTHON) tools/run_dct4_disturbance_sweep.py
