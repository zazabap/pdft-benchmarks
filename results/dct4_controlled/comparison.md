# DCT-IV: structured O(2) twiddle (frozen mirror) vs dense O(4)

DIV2K-8q, `m=n=8`, batch 50, K=6554 (ρ=0.10), seed 42, **198 optimizer steps**
(22 epochs), both variants trained back-to-back in one process on the **same
card** (`NVIDIA RTX 6000 Ada`). The structured variant uses
`DCT4Basis(parametrization="controlled")` (twiddle = single-angle `CRY` on O(2))
with the 112 mirror CNOTs **frozen** as fixed routing; the baseline is the stock
`DCT4Basis` (all 168 two-qubit gates trained on dense O(4)). Source:
`experiments/dct4_controlled_compare.py`; raw numbers `compare.json`.

| metric | dct4 (O(4)) | dct4 (controlled, O(2)) | ratio |
|---|---:|---:|---:|
| **trainable params** | 2872 | **408** | **7.0× fewer** |
| ms/step | 1657.8 | 2016.7 | 0.82× (slower) |
| final train loss | 141.1 | 145.8 | — |
| PSNR@0.05 | 25.513 | 25.395 | −0.12 dB |
| PSNR@0.10 | 27.872 | 27.749 | −0.12 dB |
| PSNR@0.15 | 29.688 | 29.551 | −0.14 dB |
| PSNR@0.20 | 31.280 | 31.128 | −0.15 dB |

## What the effect is

- **Parameter count: 7× leaner.** Avoiding O(4) drops the trainable set from 2872
  to 408 (56 single-angle O(2) twiddles + 46 singles/signs; the 112 mirror CNOTs
  are frozen). **Reconstruction is essentially unchanged** — within ~0.15 dB at
  every keep ratio after 198 steps. So the dense-O(4) freedom of the twiddle and
  mirror gates buys ≈nothing in quality here; the structured DCT-IV nearly
  matches it with **7× fewer free parameters**.
- **Speed: not improved — ~22% slower per step.** This is the important,
  non-obvious result.

## Why it is *not* faster (despite frozen mirror + cheaper gates)

1. **Freezing does not reduce the backward pass.** `train_basis_batched` computes
   `jax.value_and_grad` over **all** gate tensors every step; `frozen_indices`
   only zeroes the frozen gates' *update* before the optimizer step (and the
   optimizer is <1% of a step — see `results/profiling/`). The expensive
   gradient *computation* still flows through all 168 gates, so freezing the
   mirror saves essentially nothing in compute.
2. **The current `CRY` application isn't cheaper than the `U4` it replaces.** The
   stepped builder applies `CRY` as a 1-qubit `tensordot` on the target **plus a
   full-size `jnp.where` mask** over the 2¹⁶ state (× batch 50), and the backward
   differentiates through that mask. That offsets the cheaper contraction — and
   adds overhead — versus the single dense `U4` `tensordot`.
3. A portion of the gap is the controlled variant's larger first-step XLA compile
   amortized over only 198 steps.

## Takeaway

The structured (avoid-O(4)) DCT-IV is a **much leaner, equally accurate**
parametrization (7× fewer trainable params, ~equal PSNR), but it is **not faster
to train as implemented** — a real speedup needs the deferred work: (a) a gate
application that doesn't materialize a full-size masked intermediate (apply the
CNOT mirror as an index permutation and `CRY` structurally on the control=1 half
only), and (b) avoiding gradient computation through frozen gates. Those are the
genuine forward/backward-cost levers; the parametrization change alone moves
parameters, not wall-clock.

## Full 1008-step run (training dynamics)

Both variants trained to the headline 1008-step budget (parallel, separate Ada
cards; loss is card-independent). Trajectories in
`figures/training_dynamics.{pdf,svg}`, traces in `trace_dct4_*.json`.

| metric | dct4 O(4) | dct4 controlled (O(2)) |
|---|---:|---:|
| final train loss | 114.07 | 122.70 |
| validation-loss plateau | ~102 (still creeping down) | ~110 (flat from ~epoch 40) |
| PSNR@0.10 | 29.28 dB | 28.85 dB (−0.43) |
| wall-clock (1008 steps, Ada) | 26.9 min (1.60 s/step) | 31.3 min (1.87 s/step) |

At full convergence the gap is a bit wider than the 198-step snapshot (−0.43 vs
−0.12 dB PSNR): the controlled basis (408 params) **plateaus** at val ≈110 around
epoch 40, while the O(4) basis (2872 params) keeps inching down to ≈102. So the
dense-O(4) capacity *does* buy a little at convergence — but the structured basis
still lands within ~0.43 dB using 7× fewer parameters. It remains ~17% slower per
step (same reasons as above).

Per-step cost decomposition for the O(4) DCT-IV (forward/backward/optimizer split,
why it is ~7× the QFT) is in `results/profiling/`.
