# Ablation: stacked-blocked depth (K)

**Question:** does deeper block stacking (K>1 within-block circuit repetitions) improve quality?

**Varied:** K — 2, 3 (short schedule), 3 (long schedule).

**Fixed:** dataset (DIV2K-8q, m=n=8), inner basis = `BlockedBasis` family.

**Control cell:** `results/published/div2k_8q__blocked/` — the canonical K=1.

**Subdirs:**

- `K2/` — K=2.
- `K3_short/` — K=3, short schedule.
- `K3_long/` — K=3, long schedule.
