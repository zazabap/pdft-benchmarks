# QFT DIV2K-8q training profile

Canonical `pdft.QFTBasis(8,8)`, batch 50, K=6554 (rho=0.1), 72 gates in 2 manifold groups, **dtype `complex128` (FP64)** on `NVIDIA RTX A6000`. Medians over 20 warm reps (`block_until_ready`); heavy graphs isolated per-process to avoid OOM.

## Per-step attribution

| phase | time | % of step |
|---|---:|---:|
| forward (loss eval) | 201.4 ms | 40.3% |
| backward (value_and_grad) | 297.7 ms | 59.6% |
| optimizer (Riemannian Adam) | 0.5 ms | 0.1% |
| **warm step (total)** | **499.6 ms** | 100% |

**Answer: the NON-optimizer part (forward+backward through the circuit) dominates.** The optimizer (project/retract/transport on the gate manifolds) is 0.1% of each step; forward+backward through the circuit is 99.9%.

### Inside the forward

| sub-phase | time |
|---|---:|
| forward transform (apply 72 gates to 2^16 state) | 98.2 ms |
| top-k truncate (full sort over 65536 coeffs) | 5.7 ms |
| inverse transform (reconstruct) | 98.8 ms |

## Compile + end-to-end

- First-step XLA compile: **22.3 s** (one-time).
- Warm step: **499.6 ms** x 1008 steps = 8.4 min.
- Validation: 297.7 ms/epoch x 112 = 0.6 min.
- **Extrapolated total: 9.3 min.**

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=<free-gpu> CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=/workspaces/parametric-dft-paper/pdft-dct4main/src \
  /workspaces/pdft-benchmarks/.venv/bin/python results/profiling/qft_div2k_8q/profile_script.py --basis qft
```

Raw numbers: `profile.json`. Device trace: `trace/` (captured=True).
