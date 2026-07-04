# Reproduction gate — QuickDraw retrain (2026-07-04)

Retrained `real_rich` + `rich` (seed 42, generalized preset, `--epochs 112
--no-early-stop`, 1008 steps — identical config to the committed cells per
their `env.json`/`_pdft_py`) to obtain checkpoints, since the committed
cells saved only metrics.

**Absolute PSNR did NOT reproduce** (retrained ~0.7–1.0 dB below the
committed `results/structure/quickdraw_pca_vs_block_dct` cells at every
keep ratio). Root cause: the QuickDraw `.npy` files on this machine are a
different dataset snapshot than the committed run used — even the
deterministic, training-free classical baselines shift on the same nominal
seed-42 split (`dct` −0.07…−0.16 dB, `fft` −0.18…−0.20 dB, `block_dct_8`
−0.50…−0.85 dB vs `_baselines.json`). DIV2K (fixed ETH zip) reproduced the
committed cells exactly (≤0.01 dB), confirming code + config are faithful.

**The scientific gap reproduces.** real_rich − block_dct_8, evaluated
within a single run on identical images:

| keep | committed gap | retrained gap |
|------|---------------|---------------|
| 0.05 | +1.60 dB      | +1.40 dB      |
| 0.10 | +3.10 dB      | +2.92 dB      |
| 0.15 | +4.58 dB      | +4.43 dB      |
| 0.20 | +6.04 dB      | +5.91 dB      |

The 6_dataset_compression experiment compares all contenders on the same
current data, so its conclusions are internally consistent. Absolute PSNR
values in this experiment are on the current data snapshot and are ~0.5–1
dB below the committed structure-tree tables; compare gaps, not absolutes.

`metrics.json` + `env.json` here are the retrained run's own outputs
(includes the same-run classical baselines used for the gap check above).
