# Paper-revision recommendations

**Target paper:** *Parametric Quantum Circuits as Sparse Image Bases*
([zazabap/parametric-dft-paper](https://github.com/zazabap/parametric-dft-paper))

**Date:** 2026-04-30

This document records two decisions about the paper that came out of a
benchmarking review session — *which bases to feature* and *what the
main contribution should be* — and gives drop-in replacement text for
both. The full empirical evidence is in `results/published/` of this
repo (pdft-benchmarks).

---

## 1. Which bases to feature in the headline table

### Empirical evidence (PSNR at 20% retention, from `results/published/`)

| basis | QuickDraw | DIV2K-8q | DIV2K-10q |
|---|---:|---:|---:|
| Block-DCT 8×8 (reference) | 26.63 | **34.01** | 38.79 |
| Blocked QFT | 30.06 | 32.26 | — |
| Blocked RichBasis | **32.60** | 33.65 | — |
| Blocked RealRichBasis | 32.37 | 33.70 | — |
| QFT (basic) | 24.35 | 30.91 | 35.35 |
| Entangled-QFT | 24.35 | 31.23 | 35.67 |
| TEBD | 24.01 | 31.28 | 35.67 |
| MERA | — | 31.28 | — |

### Three observations supporting the cut

1. **TEBD ≈ MERA at every DIV2K-8q cell** (25.09 / 27.56 / 29.52 / 31.28
   identical to two decimal places). Empirically indistinguishable in
   this regime; keeping both as separate paper rows costs table space
   without adding story.
2. **Blocked-QFT is strictly dominated** by both Rich-Basis variants on
   every cell. No reason to feature it.
3. **Blocked RichBasis vs Blocked RealRichBasis** is a 0.05–0.20 dB
   difference. RichBasis wins on QuickDraw (32.60 vs 32.37); RealRichBasis
   wins on DIV2K (33.70 vs 33.65). The two are essentially the same model
   (real vs complex parameterization).

### Recommended featured set (3 learned bases)

| role | basis | rationale |
|---|---|---|
| **Foundational** | **QFT (basic)** | The Cooley–Tukey-derived endpoint of the parametric family; carries the gate-level mechanistic story. |
| **Topology pivot** | **Entangled-QFT** | Marginally beats QFT on DIV2K via cross-dim coupling — minimal but credible support for the topology insight. |
| **Headline learned basis** | **Blocked-RealRichBasis** | Closes the gap to BlockDCT on DIV2K (−0.3 dB) and beats it on QuickDraw by +5.7 dB. |

### Move to appendix

- **TEBD** (or fold into a single row "TEBD/MERA" — the data treats them as equivalent).
- **MERA** (only meaningful on DIV2K-8q since `m+n` must be a power of 2).
- **Blocked-QFT**.
- **Blocked-RichBasis** (subsumed by RealRichBasis in the headline; keep in appendix as the QuickDraw winner).

This cuts the main rate-distortion table from 10 rows to 6 (3 classical
+ 3 learned). All paper claims are preserved; the table becomes
substantially easier to read.

---

## 2. Recommended main conclusion / contribution

### Why the current contribution list is sub-optimal

The paper's stated contributions are:

> (i) DFT/DCT not optimal — learned circuit beats DCT at every retention rate.
> (ii) Mechanistic reading — block locality emerges in trained QFT.
> (iii) Topology ranking — Entangled-QFT > QFT > MERA > TEBD.
> (iv) Open-source library.

Two issues:

- **(i) is a generic learned-codec result.** Any decent neural codec
  beats full-image DCT; the user's method is one example, not the
  decisive evidence. The framing puts the paper in competition with
  Ballé et al. 2017 and similar without a clear win.
- **(iii) is empirically weak.** The published numbers show
  TEBD ≈ MERA ≈ Entangled-QFT at most DIV2K-8q cells — the
  "stable ranking" claim isn't really supported.

### What's actually the strongest finding in the data

The hidden empirical signal is the **data-dependence of the gain over BlockDCT**:

| dataset (kr=20%) | source regime | Block-DCT vs best learned | gap |
|---|---|---:|---:|
| DIV2K-8q (256×256) | AR(1)-like natural images | 34.01 vs 33.70 | **−0.31 dB** (DCT wins) |
| QuickDraw (32×32) | sparse strokes (non-AR(1)) | 26.63 vs 32.60 | **+5.97 dB** (learned wins) |

This pattern is **theoretically predicted** by the Ahmed–Jain
DCT-converges-to-KLT-as-ρ→1 result, which holds for AR(1) sources and
fails for non-AR(1) sources. Your QuickDraw number is the empirical
demonstration of where this asymptotic equivalence breaks down.

### Recommended restructured contributions (ordered by strength)

> 1. **Framework.** A continuous parametric family of unitary image
>    bases, derived from the Cooley–Tukey FFT by relaxing each gate
>    into a free element of U(2) or U(1)⁴, that includes FFT and
>    BlockDCT as special points and admits gradient-based search via
>    Riemannian optimization on the unitary manifold combined with a
>    straight-through estimator for top-k truncation.
>
> 2. **Empirical finding: gain over BlockDCT is predicted by source
>    regime.** On DIV2K-style natural images (AR(1)-like, where DCT is
>    near-optimal by Ahmed–Jain asymptotic equivalence), learning closes
>    the gap to BlockDCT to within 0.3 dB but does not exceed it. On
>    QuickDraw line drawings (non-AR(1)), the same family beats BlockDCT
>    by 5.9 dB at 20% retention. **This validates the framework where
>    existing theory predicts DCT should fall short.**
>
> 3. **Mechanistic interpretation.** The trained basic-variant QFT
>    spontaneously freezes specific qubits during training, reducing the
>    effective basis to an intermediate 16-pixel block size — JPEG's
>    block-locality inductive bias emerging without being designed in.
>
> 4. **Open-source release.** ParametricDFT.jl combines tensor-network
>    contraction, Riemannian optimization, and differentiable top-k
>    truncation in a single pipeline.

The "topology ranking" claim becomes a §4.2 minor technical observation
rather than a headline contribution — the empirics don't support a
clean ranking on DIV2K, so demoting it strengthens the paper's
epistemic posture.

### Suggested replacement abstract (22% shorter)

> Image codecs from JPEG to HEVC project images onto fixed unitary
> bases — the DFT, the DCT, and their block-wise variants. Closed-form
> optimality results justify these choices only under restrictive
> source assumptions (Gaussian for KLT, asymptotic AR(1) for DCT) that
> natural images only approximately satisfy. We construct a continuous
> parametric family of unitary image bases by relaxing the Hadamard
> and controlled-phase gates of the Cooley–Tukey FFT into free
> elements of U(2) and U(1)⁴, and search this family by Riemannian
> optimization on the unitary manifold combined with a straight-
> through estimator for top-k coefficient selection. **The gain from
> learning over fixed BlockDCT is predicted by how far the source
> departs from AR(1):** on DIV2K natural images, where DCT is
> near-optimal, learning closes the gap to within 0.3 dB; on
> QuickDraw line drawings, where the AR(1) assumption fails, the
> same family beats BlockDCT by 5.9 dB at 20% retention. Inspecting
> the trained gates of the basic-variant QFT shows that training
> spontaneously freezes specific qubits, reducing the effective
> basis to an intermediate 16-pixel block size — JPEG's
> block-locality inductive bias emerging without being designed in.
> The framework is released as the open-source Julia library
> ParametricDFT.jl.

---

## 3. References to add

If the restructured contribution #2 lands, add to `references.bib`:

- **Ahmed, N., Natarajan, T., Rao, K. R.** (1974), "Discrete Cosine
  Transform," *IEEE Trans. Computers* C-23(1):90–93. (DCT origin +
  KLT-asymptotic-equivalence claim.)
- **Jain, A. K.** (1979), "A sinusoidal family of unitary transforms,"
  *IEEE Trans. PAMI* 1(4):356–365. (Formal proof of DCT → KLT for
  AR(1).)
- **Effros, M., Feng, H., Zeger, K.** (2004), "Suboptimality of the
  Karhunen–Loève transform for transform coding," *IEEE Trans. IT*
  50(8):1605–1619. (Provides the formal "KLT not always optimal"
  framing that supports your finding #2.)
- **Goyal, V. K.** (2001), "Theoretical foundations of transform
  coding," *IEEE Signal Processing Magazine* 18(5):9–21.
  (Survey-level treatment, useful for the introduction.)

The full BibTeX is in `pdft-benchmarks/docs/theory/refs.bib`.

---

## Note on rank-truncation variants

The benchmark code includes `pca_rank`, `block_pca_8_rank`, `dct_rank`,
`block_dct_8_rank` — the textbook eigenvalue-rank-truncation /
zigzag-position-truncation rules. These were added to confirm that
KLT's L2-optimality theorem holds under its native rule (Block-PCA-rank
beats Block-DCT-rank by 0.10 dB on DIV2K-8q at kr=0.20 — small but
consistent across all cells, confirming the theorem).

**These variants should NOT appear in the paper.** Reason: rank-rule
loses ~7 dB to magnitude-rule on every basis, purely from
uniform-vs-adaptive per-block bit allocation (see Berger 1971,
*Rate Distortion Theory*, Ch. 5: water-filling). That 7 dB gap is
about the *rule*, not the *basis* — both DCT-rank (26.30 dB) and
PCA-rank (26.40 dB) lose by approximately equal amounts. Featuring
them in the paper would invite readers to compare rank-rule numbers
to magnitude-rule numbers, which is not a basis-quality comparison
and would muddy the data-regime story.

The variants stay in `src/pdft_benchmarks/pca.py` and
`src/pdft_benchmarks/baselines.py` as **internal documentation that we
verified the KLT theorem holds correctly within our framework** —
defensive against the natural reviewer concern "did you apply KLT in
its theoretically-optimal regime?" Yes; we did; the gap to DCT under
that regime is 0.10 dB.

## Open question (for future work)

If the paper wants to claim a **new framework for compression
optimality**, the strongest path is:

> *Compressibility ↔ entanglement entropy.* Tensor-network theory has
> a constructive bound: a state with entanglement entropy ≤ S across
> a cut is representable by a tensor network with bond dimension
> exp(S). This bound is well-known in physics but absent from
> compression theory. A formal "rate-distortion ≤ entanglement
> entropy" theorem, plus measurement of the entropy of natural-image
> distributions, would give a genuinely new framework.

Estimated effort: 3–6 months of additional theoretical work. See
`pdft-benchmarks/docs/theory/framework_potential.md` for the full
discussion of this and three alternative theoretical directions.
