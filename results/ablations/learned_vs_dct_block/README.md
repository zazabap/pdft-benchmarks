# Ablation: BlockedBasis with DCT inner ≈ block_dct_8 baseline

**Question:** sanity check — when the inner of a `BlockedBasis` is fixed to a DCT, does the trained outcome reduce to the classical `block_dct_8` baseline?

**Fixed:** DIV2K-8q, generalized preset.

**Control:** the `block_dct_8` row inside `results/published/div2k_8q__blocked/metrics.json`.

**Subdir:** `DCT/` — full run with `blocked_dct` (`BlockedBasis(inner=DCT)`).
