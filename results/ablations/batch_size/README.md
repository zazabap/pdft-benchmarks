# Ablation: batch size sweep (DIV2K-8q blocked)

**Question:** how does batch size affect throughput vs accuracy on the canonical 8q blocked configuration?

**Varied:** `batch_size ∈ {4, 16, 32, 64}`.

**Fixed:** dataset (DIV2K-8q), basis = `blocked`, preset = `generalized`, all other hyperparameters.

**Control cell:** `results/published/div2k_8q__blocked/` — the canonical training run (preset default `batch_size`).

**Subdirs:** `bs4/`, `bs16/`, `bs32/`, `bs64/`.
