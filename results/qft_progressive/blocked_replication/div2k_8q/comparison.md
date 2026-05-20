# DIV2K-8q `blocked_8` reproduction via QFT + frozen_indices

The canonical `blocked_8` cell uses `BlockedBasis(QFTBasis(3, 3), 5, 5)` at
analytic init, trained 1008 steps on DIV2K-8q. The reproduction uses
`QFTBasis(8, 8)` at identity init with the 60 outer gates frozen at identity
via the new `pdft.train_basis_batched(..., frozen_indices=...)` parameter
(upstream pdft commit 31decf7, PR #20).

By operator equivalence — putting newly introduced gates at identity in
QFT(k+1) yields QFT(k) ⊗ I_2 on the inner (k+1)-qubit space — the two
basis configurations produce bit-exactly identical operators. Training
dynamics and final PSNR should match within numerical noise.

## Results

| Configuration | Family | Init | Frozen | PSNR @ ρ=0.20 | val MSE final |
|---|---|---|---|---|---|
| `blocked_8` (canonical) | `BlockedBasis(QFTBasis(3,3), 5, 5)` | analytic | implicit (block-tiling) | 32.261 | 129.3 |
| `qft_frozen_outer` (replication) | `QFTBasis(8, 8)` | identity | explicit 60 outer indices | 32.261 | 129.3 |
| Δ | — | — | — | +0.000 | 0.0 |

Both runs use the headline preset (1008 steps, batch 50, Adam, cosine LR
0.01 → 0.001, val_split 0.15, seed 42, k_train=6554).

## Reproduction PSNR @ each ρ

| ρ | `blocked_8` (canonical) | `qft_frozen_outer` (replication) | Δ (dB) |
|---|---|---|---|
| 0.05 | 25.185 | 25.185 | +0.000 |
| 0.10 | 28.086 | 28.086 | +0.000 |
| 0.15 | 30.297 | 30.297 | +0.000 |
| 0.20 | 32.261 | 32.261 | +0.000 |

## What this demonstrates

The operator-equivalence claim is empirically validated to 3 decimal places at
the headline 1008-step budget. The maximum observed PSNR delta across all four
rate points is +0.000 dB (sub-millidecibel noise floor), confirming that
`QFTBasis(8, 8)` with 60 outer gates frozen at identity is functionally
indistinguishable from `BlockedBasis(QFTBasis(3, 3), 5, 5)` under the
canonical training conditions. The bit-exact frozen-tensor sanity check also
passed: all 60 outer gate tensors remained precisely at identity after
1008 training steps.

The practical implication is that `frozen_indices` is a strictly-more-general
primitive than BlockedBasis for this kind of structural sparsity. The same
equivalence works for any subset of the canonical-order tensor list, not just
inner/outer partitions aligned to block boundaries. In particular, one can
freeze arbitrary subsets of a fully-connected QFT circuit to recover any
intermediate topology between the dense QFT(m, n) and a block-tiled
BlockedBasis(QFT(k, k), …) limit, opening the door to learned sparsity
patterns that do not correspond to any explicit BlockedBasis configuration.
