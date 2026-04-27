# Ablation: RichBasis initialization

**Question:** does the choice of `RichBasis` init scheme materially affect convergence at DIV2K-8q?

**Varied:** init scheme — `DCTINIT`, `DENSE`, `DENSE_DCTINIT`, `LONG`.

**Fixed:** dataset (DIV2K-8q, m=n=8), preset (`generalized`), all other hyperparameters.

**Control cell:** `results/published/div2k_8q__rich/` — the canonical RichBasis with the registry-default init.

**Subdirs:**

- `DCTINIT/` — init from a DCT-truncated basis.
- `DENSE/` — init dense random.
- `DENSE_DCTINIT/` — DCTINIT then densified.
- `LONG/` — same as the canonical run but trained for more epochs (sanity).

To compare, read `metrics.json` per subdir against the control.
