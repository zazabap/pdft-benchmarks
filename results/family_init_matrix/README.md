# Family × init progressive block-size matrix

Cross-family study: the progressive block-size curriculum
(`experiments/qft_progressive.py`) run for every circuit family × both
initialisations, so families and init policies can be compared head-to-head on
the same axis.

- **Families:** `rich`, `qft`, `tebd`, `entangled_qft`, `mera`
- **Inits:** `identity` (gates dropped to the identity operator) and `random`
  (rich Haar U(2)/U(4); qft Haar-on-Hadamards + random controlled-phase angles;
  tebd/entangled_qft/mera native seeded random)
- **Stages:** `k=2..8` (block size `2^k`), except `mera` which is only defined
  at `k in {2,4,8}` (m must be a power of 2).

## Layout

```
<dataset>/<family>_<init>/_runs/stage_k<k>/   one cell per stage
                                              (env.json, metrics.json,
                                               trained_*.json, loss_history/)
<dataset>/<family>_<init>/manifest.json       per-sweep aggregate
<dataset>/report/                             dual-rate writeup + figures
```

As in the rest of `results/`, `trained_*.json` and `loss_history/` are
gitignored (kept on disk, not tracked); `env.json` / `metrics.json` /
`manifest.json` / figures are tracked.

## Status (2026-05-25)

- **div2k_8q** — COMPLETE. All 10 combos saved (rich/qft k=1..8, tebd/ent-qft
  k=2..8, mera k∈{2,4,8}; identity + random each). Headline: only `rich` clears
  the classical block-8 reference; the four QFT-derived families are bit-identical
  under identity init; identity init ≥ random for every family. Report under
  `div2k_8q/report/`. NB: `rich_identity` and `qft_identity` here duplicate the
  standalone `results/rich_progressive` / `results/qft_progressive` (single-init)
  experiments — the matrix reuses their identity data.
- **tuberlin_8q** — COMPLETE. All 10 combos saved. Sketches are block-sparse, so
  the story inverts vs DIV2K: PSNR *falls* with block size, and classical
  block-DCT-8 is near-lossless at the light rate (median ~94 dB @ rho=0.20) but
  drops to ~36 dB @ rho=0.05, where learned `rich` (identity) edges above it.
  Family ranking flips between rates (QFT-derived families lead at rho=0.20,
  `rich` at rho=0.05). Dual-rate report under `tuberlin_8q/report/`.

The report metric is mean test PSNR over 50 sketches at two rates:
`rho=0.20` (5x) and `rho=0.05` (20x). On these block-sparse sketches a classical
block-DCT-8 reference is near-lossless at the light rate (~94 dB median) but
drops to ~36 dB at 20x, where the learned circuits become competitive.
