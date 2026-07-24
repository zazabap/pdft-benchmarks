# tools/

Command-line utilities and figure renderers. Run from the repo root
(e.g. `python tools/paper/render_topology_loss.py …`), or via the top-level
`make` targets (see the repo README).

## Layout

- **`paper/`** — renderers for the paper's canonical figures and tables:
  topology loss (Fig. 4), frequency/reconstruction gallery (Fig. 5),
  rate-distortion (Fig. 6), and the per-dataset PSNR results tables.
- **`analysis/`** — exploratory and results-writeup renderers / diagnostics
  (seed variance, QFT block-structure & unfreeze, block emergence, lambda
  sweeps, disturbance curves, …). The shared matplotlib style
  `paper_style.py` lives here, next to its callers.
- **root** — dataset/pipeline utilities that are *not* figure renderers:
  `cellify_run.py`, `independent_*_baselines.py`, `validate_manifest.py`,
  `eval_seed_basis.py`, `run_seed_sweep.py`, `run_dct4_disturbance_sweep.py`.

## Figure convention

Renderers emit **PDF + SVG** (no PNG); no figure-level titles — captions live
in the writeups. See `analysis/paper_style.py` for the shared Wong-palette
style.
