# DCT-IV DIV2K-8q training profile

Canonical `pdft.DCT4Basis(8,8)` (PR #22), batch 50, K=6554 (rho=0.1), 214 gates in 3 manifold groups, **dtype `complex128` (FP64)** on `NVIDIA RTX A6000`. Medians over 20 warm reps (`block_until_ready`); heavy graphs isolated per-process to avoid OOM.

## Per-step attribution

| phase | time | % of step |
|---|---:|---:|
| forward (loss eval) | 1422.8 ms | 41.4% |
| backward (value_and_grad) | 2042.2 ms | 59.4% |
| optimizer (Riemannian Adam) | -28.2 ms | -0.8% |
| **warm step (total)** | **3436.7 ms** | 100% |

**Answer: the NON-optimizer part (forward+backward through the circuit) dominates.** The optimizer (project/retract/transport on the gate manifolds) is only -0.8% of each step; forward+backward through the circuit is 100.8%. The optimizer cost is also batch-independent (it touches the tiny 2x2/4x4 gate tensors, not the 50 images), so at the real batch its share is small.

### Inside the forward

| sub-phase | time |
|---|---:|
| forward transform (apply 214 gates to 2^16 state) | 703.8 ms |
| top-k truncate (full sort over 65536 coeffs) | 6.2 ms |
| inverse transform (reconstruct) | 710.3 ms |

## Compile + end-to-end

- First-step XLA compile: **5.5 s** (one-time).
- Warm step: **3436.7 ms** x 1008 steps = 57.7 min.
- Validation: 2135.4 ms/epoch x 112 = 4.0 min.
- **Extrapolated total: 61.8 min** (cross-check vs the real 1008-step run on the other card).

## Why it is slow

1. **FP64 (`complex128`).** The whole pipeline is complex128. On RTX 6000 Ada / A6000 (consumer / pro-viz silicon) FP64 runs at ~1/32-1/64 of FP32, so every contraction and the gradient run far below peak. Phase-independent tax; the single biggest lever.
2. **Two full circuit contractions per image** (forward + inverse) over a 2^16=65536 state through 214 gates, vmapped over 50 images, then differentiated.
3. **A full top-k sort** over 65536 coefficients per image (and its derivative).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=<free-gpu> CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=/workspaces/parametric-dft-paper/pdft-dct4main/src \
  /workspaces/pdft-benchmarks/.venv/bin/python results/profiling/dct4_div2k_8q/profile_script.py
```

Raw numbers: `profile.json`. Device trace (TensorBoard/Perfetto): `trace/` (captured=True).
