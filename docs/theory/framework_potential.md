# Could pdft be a new framework for image-compression optimality?

**Status:** discussion document, in support of paper drafting.
**Date:** 2026-04-30

This document records four directions in which the pdft tensor-network /
quantum-circuit basis class could yield genuine theoretical contributions
to image-compression theory beyond the empirical rate-distortion results
in `results/published/`. None of these directions are required for a
solid empirical paper; all are required for a paper claiming a *new
optimality framework*.

The classical compression literature offers no closed-form optimality
theorem for natural images: KLT requires Gaussian sources (Karhunen
1947, Loève 1948); DCT requires asymptotic AR(1) (Ahmed–Natarajan–Rao
1974, Jain 1979); l1-minimization requires exact sparsity (Candès–Tao
2006); manifold-based optimality requires a known manifold. Natural
images satisfy *none* of these. Recent learned-compression work
(Ballé–Laparra–Simoncelli 2017, Theis–Shi–Cunningham–Huszár 2017,
Yang–Mandt 2023) sidesteps the issue by minimizing the empirical
rate-distortion loss in a parametric family — without claiming
closed-form optimality. Our method sits in that paradigm by default.

The four directions below describe how to elevate the contribution
*beyond* "another parametric family that works empirically."

---

## Direction 1: Compressibility ↔ entanglement entropy

A natural image, reshaped as a tensor state, has a Schmidt decomposition
across any bipartition of its degrees of freedom. The Schmidt rank
across cut C equals the minimum bond dimension D such that a
tensor-network state with bond dim D can represent the image exactly
along C. The associated von Neumann entropy S(C) = -∑ λ_i² log λ_i²
captures the *information* across the cut.

In condensed-matter physics, ground states of local 1D Hamiltonians
satisfy an *area law*: S(C) ≤ const, independent of system size. This
is what makes MPS / DMRG efficient. Natural images have empirical
area-law-like behavior (low entanglement across local cuts) — already
observed in image-tensor-network work (Stoudenmire–Schwab 2016).

**Potential theorem:** Let *p(x)* be a distribution over n×n images
and let *S(p, L)* be the expected entanglement entropy across an L-cut
under *p*. Then there exists a tensor-network basis Φ with bond
dimension D = exp(S(p, L)) that achieves rate-distortion within ε of
the optimum on samples from *p*, where ε → 0 as L → n.

This connects rate-distortion *directly* to entanglement entropy — a
quantity well-studied in physics but new to compression theory.

**What it would require:**
- A formal proof of the bound (likely via standard MPS approximation
  arguments, e.g., Verstraete–Cirac–Latorre 2008).
- Numerical measurement of S(p, L) for natural-image distributions
  at various L (8, 16, 32, ...). This is a finite-block-size
  analogue of the area law.
- Empirical validation: compare measured S(p, 8) to the bond
  dimension required by a fitted MPS or MERA basis to hit a target
  PSNR.

**Estimated effort:** 3–6 months of additional theoretical work.
**Impact if successful:** publishable as a standalone paper; opens
a new sub-field.

**Key references:**
- Schumacher, B., "Quantum coding," *Phys. Rev. A* 51:2738 (1995)
  — classical analogue: Shannon coding to von Neumann entropy bound.
- Verstraete, F., Cirac, J. I., Latorre, J. I., "Matrix product
  density operators: Renormalization fixed points and boundary
  theories," *Annals Phys.* 322:1452 (2008).
- Stoudenmire, E. M., Schwab, D. J., "Supervised learning with
  tensor networks," *NeurIPS* 2016.

---

## Direction 2: Tractability of optimization within the TN family

KLT solves "min E[||x − Px||²] over all rank-k linear projections P."
The solution requires SVD of an N²×N² covariance matrix — exponential
in image side. Within the TN family, parameter count is *polynomial*
(MPS: O(Nχ²); MERA: O(N log N · χ⁴) for bond dimension χ), and the
optimization decomposes into local updates.

**Question:** Is the rate-distortion-optimal TN basis efficiently
computable (poly time)?

The answer is *yes* for several restricted regimes — DMRG-style
sweep optimization converges to local optima in poly time, and recent
SDP relaxations (Glasser et al. 2020, "Probabilistic graphical models
and tensor networks") give global guarantees for specific structures.

**Potential theorem:** For the parametric class C of TN bases with
fixed structure σ and bond dim D, the rate-distortion objective is
*efficiently approximable* (within ε) in time poly(N, 1/ε).

**Comparison to KLT:** KLT itself is poly time (it's just SVD), so
this is *not* a tractability win against KLT. The novelty is in
proving the same tractability for a strictly *larger* family —
specifically, the family that *also* optimizes over the rule
(top-k-magnitude vs rank-k), which KLT cannot do.

**What it would require:**
- An algorithm (DMRG-like) with provable global convergence under
  reasonable assumptions.
- Or: an SDP relaxation with provable approximation ratio.
- Empirical validation: show convergence on the existing benchmark.

**Estimated effort:** 2–4 months.
**Impact:** medium; complementary to direction 1.

---

## Direction 3: Multi-scale rate-distortion via MERA

MERA (Multi-scale Entanglement Renormalization Ansatz) has explicit
hierarchical structure: coarse-graining layers + disentanglers
between scales. Wavelet theory has classical multi-scale RD analysis
(Donoho–Johnstone 1994), but those bounds rely on signal-class
assumptions (Besov-space membership, piecewise smoothness).

**Potential contribution:** Show MERA captures a *strictly larger*
signal class than wavelets with the same RD scaling, OR achieves a
*strictly faster* RD scaling on the same class.

The renormalization-group-flow interpretation of MERA (Vidal 2007)
provides a different theoretical handle than wavelet theory. The
RG-fixed-point structure suggests MERA can capture conformally
invariant signal statistics that wavelets cannot.

**What it would require:**
- A theorem of the form: "for signal class X, MERA achieves
  ||x − x̂||² = O(k^−α) with α strictly greater than the optimal
  wavelet rate."
- Or: a counterexample-class showing MERA strictly dominates wavelets.
- Numerical comparison on natural-image RD curves (already partially
  in our benchmark — MERA cells, where applicable).

**Estimated effort:** 1–2 months for the theorem; 1 month for
expanded numerics.
**Impact:** high if the theorem holds; small impact if only
empirical.

---

## Direction 4: Quantum-classical compression duality

Schumacher 1995 proved quantum compression achieves the von Neumann
entropy bound for quantum data. Shannon 1948 proved classical
compression achieves the Shannon entropy bound for classical data.
These are different bounds for different settings.

Our basis class is a *classical realization of quantum-circuit
transforms*: classical (data and operations live in ℝᴺ, not Hilbert
space) but structurally quantum-inspired.

**Potential framing:** The pdft basis family is a "quantum-inspired
classical compression family" — classical (so Shannon-bounded), but
inheriting expressivity properties from quantum-complexity theory
(QFTBasis, MERA, etc.). This *exposes* a new frontier: bounds on
classical RD via quantum-circuit-complexity arguments.

**What it would require:**
- Conceptual framing in the introduction.
- A discussion section drawing the bridge.
- Potentially: a single-source comparison showing pdft RD performance
  matches a known quantum-compression bound (e.g., for a Gaussian
  ensemble where quantum compression is well-understood).

**Estimated effort:** weeks (no new technical work; it's a
positioning argument).
**Impact:** low-to-medium technical, high pedagogical /
visibility — frames the work as theoretically significant.

---

## What is *defensible right now* (without theory additions)

If we do not add any of the four directions above, the strongest
honest claim is:

> *"We introduce a structured-parametric basis family — drawn from
> tensor-network and quantum-circuit constructions — and demonstrate
> empirically that end-to-end rate-distortion optimization within
> this family outperforms both fixed (DCT) and dataset-fitted linear
> (KLT) baselines, particularly when source statistics deviate from
> AR(1). The family has polynomial parameter count, retains the
> linearity of transform coding, and admits efficient optimization.
> Establishing closed-form rate-distortion optimality within this
> family — connecting compression rate to entanglement entropy or
> renormalization-group fixed points — is left for future work."*

This is publishable. It is *not* a new optimality framework. It is
a new *parametric family* with empirical evidence of utility.

---

## Recommendation

For paper-drafting, choose between:

1. **Empirical paper** (current state of the work). Strongest claim is
   "new structured-parametric family + empirical RD evidence."
   Timeline: ready now.

2. **New-framework paper** (requires direction 1 or 2). Strongest
   claim is "compressibility ↔ entanglement entropy, with
   constructive transforms achieving the bound."
   Timeline: 3–6 months of additional theoretical work.

Pursue (1) first as a workshop/short paper if timing matters; pursue
(2) for a journal venue (IEEE Trans. IT, Nature Communications) once
the theoretical content lands.

A pre-registration of (2) — committing to which direction in this
document and outlining the planned theoretical contribution — would
strengthen any (1) submission by signalling the broader research
program.
