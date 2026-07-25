# Reproduce the paper by section. Each target renders from committed data and
# verifies; add RETRAIN=1 to run the experiment standalone first (needs a GPU
# + the datasets).
PYTHON ?= .venv/bin/python
export CUDA_DEVICE_ORDER := PCI_BUS_ID
export XLA_PYTHON_CLIENT_PREALLOCATE := false
RETRAIN_FLAG := $(if $(filter 1,$(RETRAIN)),--retrain,)

.PHONY: help all 00 01 02 03 04 05
.DEFAULT_GOAL := help

help:
	@echo "Reproduce the paper by section (default: render + verify from committed data):"
	@echo "  make 00   dataset distribution   (Fig 3)"
	@echo "  make 01   Quick Draw bases        (Table 3 QD, Fig 5 QD)"
	@echo "  make 02   DIV2K bases             (Table 3 DIV2K, Fig 4, Fig 5 DIV2K)"
	@echo "  make 03   dataset compression     (Fig 6)"
	@echo "  make 04   QFT robustness          (Appendix C: Fig 10/11, Table 5)"
	@echo "  make 05   DCT-IV disturbance      (Appendix D: Fig 12-14, Table 6)"
	@echo "  make all                          every section"
	@echo "  RETRAIN=1 make 0X                 run the experiment standalone first"

all: 00 01 02 03 04 05
00: ; $(PYTHON) experiments/00_dataset_dist.py $(RETRAIN_FLAG)
01: ; $(PYTHON) experiments/01_bases_quickdraw.py $(RETRAIN_FLAG)
02: ; $(PYTHON) experiments/02_bases_div2k.py $(RETRAIN_FLAG)
03: ; $(PYTHON) experiments/03_dataset_compression.py $(RETRAIN_FLAG)
04: ; $(PYTHON) experiments/04_robustness_qft.py $(RETRAIN_FLAG)
05: ; $(PYTHON) experiments/05_robustness_dct_iv.py $(RETRAIN_FLAG)
