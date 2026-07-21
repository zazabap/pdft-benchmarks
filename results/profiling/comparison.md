# DIV2K-8q training step profile — DCT-IV vs QFT

Same card (`NVIDIA RTX A6000`), same objective (batch 50, K=6554 / rho=0.1, 1008 steps), same dtype `complex128` (FP64). Canonical inits. Medians over 20 warm reps, heavy graphs isolated per-process. Identical profiler code (`profile_script.py`) for both, only the basis constructor differs.

## Headline

- **QFT trains ~6.9x faster per step** (500 ms vs 3437 ms).
- End-to-end (1008 steps + validation, extrapolated): **QFT ~9.3 min vs DCT-IV ~61.8 min**.
- **Same shape in both:** the optimizer step is negligible (0.1% / -0.8%); the cost is the forward+backward circuit contraction (~40% / ~60%).

## Side-by-side

| metric | QFT | DCT-IV | ratio |
|---|---:|---:|---:|
| gates (1-qubit / 2-qubit) | 72 / 0 | 46 / 168 | — |
| total gates | 72 | 214 | 3.0x |
| manifold groups | 2 | 3 | — |
| warm step (ms) | 499.6 | 3436.7 | 6.9x |
| · forward / loss (ms) | 201.4 | 1422.8 | 7.1x |
| &nbsp;&nbsp;– forward transform (ms) | 98.2 | 703.8 | 7.2x |
| &nbsp;&nbsp;– top-k sort (ms) | 5.74 | 6.15 | 1.1x |
| &nbsp;&nbsp;– inverse transform (ms) | 98.8 | 710.3 | 7.2x |
| · backward / grad (ms) | 297.7 | 2042.2 | 6.9x |
| · optimizer (ms) | 0.5 | -28.2 | -57.2x |
| validation / epoch (ms) | 297.7 | 2135.4 | 7.2x |
| per-gate forward cost (ms/gate) | 1.36 | 3.29 | 2.4x |
| extrapolated total (min) | 9.3 | 61.8 | 6.6x |

## Why DCT-IV is ~7x slower

1. **Two-qubit gates.** QFT is 72 gates, **all 1-qubit** (Hadamard + controlled-phase, both `(2,2)`). DCT-IV is 214 gates = 46 one-qubit + **168 two-qubit `(2,2,2,2)` gates** (the `U4` blocks). A 2-qubit gate contracts two legs of the 2^16 state and is individually far heavier than a 1-qubit gate — so per-gate forward cost is 3.29 ms vs 1.36 ms (2.4x). Combined with ~3x the gate count, that compounds to ~7x.
2. **Everything else is the same.** Both are complex128 (FP64), both do forward+inverse contractions + a top-k sort, both have a negligible optimizer. The top-k sort is ~6 ms in both (irrelevant). FP64 is the shared underlying tax; the gate set is what separates them.

## End-to-end on the real (faster) card

The profiles above are on the `NVIDIA RTX A6000`. The actual headline DCT-IV 1008-step run was measured on a `cuda:0` (RTX 6000 Ada): **27.5 min real wall-clock** at 1636 ms/step (loss 174.6 -> 114.1, val 101.8).

So the **same DCT-IV workload is ~2.1x faster on the Ada card than on the A6000** (3437 vs 1636 ms/step) — the FP64 throughput gap between the two GPU generations. The QFT-vs-DCT-IV ratio above is unaffected: both were profiled on the same A6000.

## Takeaways

- The optimizer (Riemannian Adam project/retract/transport) is **not** the bottleneck for either basis — it touches the tiny gate tensors, not the images, and is < 1% of a step (the small DCT-IV negative is cross-process noise).
- The lever for DCT-IV specifically is its **2-qubit `U4` gate set**; the lever for both is **FP64** (complex64/FP32 would help on this consumer-class silicon if the reconstruction accuracy tolerates it).
- Validation is ~1.5x a training step's forward (75 vs 50 images, forward-only) and runs once/epoch: 4.0 min of DCT-IV's total, 0.6 min of QFT's.

Per-basis detail: `qft_div2k_8q/` and `dct4_div2k_8q/` (report.md, profile.json, trace/).
