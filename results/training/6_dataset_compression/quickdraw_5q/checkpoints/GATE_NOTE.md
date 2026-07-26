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

## DCT-IV contender added (2026-07-26)

`trained_dct4_ctl.json` is the committed
`results/structure/quickdraw_pca_vs_block_dct/by_basis/dct4_ctl` cell's
checkpoint (trained on the current data snapshot; its plain top-k metrics
reproduce here to within quantization), stored **realified**: the U(1)^4
sign gates sit ~1e-3 off the exact real point after training, and zeroing
the imaginary parts measured 0.000 dB PSNR change at every (keep_ratio,
bits) grid point on the test split while shrinking the stored basis file
from 65,307 to 23,698 bytes. The codec reads this exact file
(is_complex=False), so the counted basis bytes are what the decoder uses.
The rd_curves.json sweep was regenerated on CPU with all four contenders
(dct4_ctl, real_rich, rich, block_dct_8); the pre-existing contenders'
readings reproduced the prior committed sweep (35 dB crossing 369 vs 474
B/img, +7.0 dB at 40% of raw).

## Payload-only accounting (2026-07-26)

Per-point `total_bytes` / `bytes_per_image` / `ratio_vs_raw` in
rd_curves.json + headline_50pct.json now count the image payload (blob)
bytes ONLY; `basis_bytes` is still recorded per point for transparency
but excluded from the totals. Rationale: the comparison is dataset
compression per image, the basis is stored once per dataset
independently, and the analytic baselines' transform definitions were
never counted either. The committed JSONs were migrated in place
(derived fields recomputed from the stored `blob_bytes_total`; no
re-encode), and the sweep driver writes the same definition going
forward.
