---
title: "Restructure main contribution: lead with data-regime finding (DCT-AR(1) gap), demote topology ranking"
labels: paper, discussion
target_repo: zazabap/parametric-dft-paper
---

# Restructure main contribution: lead with data-regime finding

## Why the current framing is sub-optimal

Current paper claims four contributions:

> (i) DFT/DCT not optimal — learned circuit beats DCT at every retention rate.
> (ii) Mechanistic reading — block locality emerges in trained QFT.
> (iii) Topology ranking — Entangled-QFT > QFT > MERA > TEBD.
> (iv) Open-source library `ParametricDFT.jl`.

Two issues:

1. **Contribution (i) is a generic learned-codec result.** Any decent
   neural codec (Ballé et al. 2017 and successors) beats full-image
   DCT at every keep ratio. The current framing puts the paper in
   competition with neural compression *without* a clear distinguishing
   claim — and Ballé et al. are doing the same thing with much more
   parametric capacity.

2. **Contribution (iii) is empirically weak.** Looking at the
   published numbers in `tables/published_8q_quickdraw.tex`:

   | basic variant | DIV2K-8q PSNR @ kr=20% |
   |---|---:|
   | QFT | 30.91 |
   | Entangled-QFT | 31.23 |
   | TEBD | 31.28 |
   | MERA | 31.28 |

   TEBD ≈ MERA ≈ Entangled-QFT to within 0.05 dB. The "stable ranking"
   claim that puts Entangled-QFT first is not really supported by the
   data. Demoting (iii) from a headline contribution to a §4.2
   technical observation strengthens the paper's epistemic posture.

## What's actually the strongest finding in the data

The hidden empirical signal is the **data-dependence of the gain over
BlockDCT**:

| dataset (kr=20%) | source regime | Block-DCT vs best learned | gap |
|---|---|---:|---:|
| DIV2K-8q (256×256) | AR(1)-like natural images | 34.01 vs 33.70 | **−0.31 dB** (DCT wins) |
| QuickDraw (32×32)  | sparse strokes (non-AR(1)) | 26.63 vs 32.60 | **+5.97 dB** (learned wins) |

This pattern is **theoretically predicted** by the Ahmed–Jain
DCT-→-KLT-as-ρ→1 result (Ahmed-Natarajan-Rao 1974, Jain 1979): DCT is
near-optimal on AR(1)-like sources and strictly suboptimal off-AR(1).
The QuickDraw result is the empirical demonstration of where this
asymptotic equivalence breaks down.

This finding **distinguishes the paper from generic neural codecs**:
not just "we beat DCT" (everyone beats DCT) but "**we beat DCT
exactly where theory predicts DCT should fail, and tie DCT exactly
where theory predicts DCT should succeed.**" That's a much sharper
empirical contribution.

## Proposed restructured contributions

> 1. **Framework.** A continuous parametric family of unitary image
>    bases, derived from the Cooley–Tukey FFT by relaxing each gate
>    into a free element of U(2) or U(1)⁴. The family includes FFT
>    and BlockDCT as special points and admits gradient-based search
>    via Riemannian optimization on the unitary manifold combined
>    with a straight-through estimator for top-k truncation.
>
> 2. **Empirical finding: gain over BlockDCT is predicted by source
>    regime.** On DIV2K natural images (AR(1)-like, where DCT is
>    near-optimal by the Ahmed–Jain asymptotic equivalence), learning
>    closes the gap to BlockDCT to within 0.3 dB but does not exceed
>    it. On QuickDraw line drawings (non-AR(1)), the same family
>    beats BlockDCT by 5.9 dB at 20% retention. **This validates the
>    framework precisely where existing theory predicts DCT should
>    fall short.**
>
> 3. **Mechanistic interpretation.** The trained basic-variant QFT
>    spontaneously freezes specific qubits during training, reducing
>    the effective basis to an intermediate 16-pixel block size —
>    JPEG's block-locality inductive bias emerging without being
>    designed in.
>
> 4. **Open-source release.** ParametricDFT.jl combines tensor-network
>    contraction, Riemannian optimization, and differentiable top-k
>    truncation in a single pipeline.

The current contribution (iii) (topology ranking) becomes a §4.2 minor
technical observation rather than a headline claim.

## Suggested replacement abstract

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
> `ParametricDFT.jl`.

## References to add to `references.bib`

- Ahmed, N., Natarajan, T., Rao, K. R. (1974), "Discrete Cosine
  Transform," *IEEE Trans. Computers* C-23(1):90–93.
- Jain, A. K. (1979), "A sinusoidal family of unitary transforms,"
  *IEEE Trans. PAMI* 1(4):356–365.
- Effros, M., Feng, H., Zeger, K. (2004), "Suboptimality of the
  Karhunen–Loève transform for transform coding," *IEEE Trans. IT*
  50(8):1605–1619.
- Goyal, V. K. (2001), "Theoretical foundations of transform coding,"
  *IEEE Signal Processing Magazine* 18(5):9–21.

Full BibTeX is at
[`pdft-benchmarks/docs/theory/refs.bib`](https://github.com/zazabap/pdft-benchmarks/blob/main/docs/theory/refs.bib).

## Companion table change

Recommended trim of the headline rate-distortion table from 10 rows
to 6 (3 classical + 3 learned). Drop-in LaTeX is at
[`pdft-benchmarks/tables/published_8q_quickdraw_v2.tex`](https://github.com/zazabap/pdft-benchmarks/blob/main/tables/published_8q_quickdraw_v2.tex).

Bases moved to appendix:
- TEBD ≈ MERA on DIV2K-8q at every cell (24.01 dB on QD); fold into a
  single "TEBD/MERA" row in the appendix.
- Blocked-QFT — strictly dominated by both Rich-Basis variants.
- Blocked-RichBasis — QuickDraw winner (32.60 dB), but the difference
  vs Blocked-RealRichBasis is < 0.25 dB; subsumed by RealRichBasis in
  the headline.

Featured set in headline:
- **QFT** (basic) — foundational, carries the gate-level story.
- **Entangled-QFT** (basic) — topology pivot showing cross-dim coupling
  helps on DIV2K.
- **Blocked-RealRichBasis** — headline learned basis, +2.85 dB over
  full-image DCT, −0.3 dB vs BlockDCT.

---

To submit this issue:

```bash
cd /path/to/pdft-benchmarks
gh issue create \
  --repo zazabap/parametric-dft-paper \
  --title "Restructure main contribution: lead with data-regime finding (DCT-AR(1) gap), demote topology ranking" \
  --label paper,discussion \
  --body-file docs/paper-issue-draft.md
```

Or paste the body manually at
https://github.com/zazabap/parametric-dft-paper/issues/new
