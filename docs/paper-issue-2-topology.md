# Move §4.2 topology comparison details to Appendix B; shorten body to one paragraph

Companion to #2 (main-contribution restructuring). This issue addresses
the related question: where should the topology-comparison content live
once contribution (iii) is no longer a headline claim?

## Why a partial move (not a full appendix punt)

Three forces argue against fully removing topology content from the body:

1. **The headline table will feature Entangled-QFT** (per the trimmed
   6-row table in
   [`pdft-benchmarks/tables/published_8q_quickdraw_v2.tex`](https://github.com/zazabap/pdft-benchmarks/blob/main/tables/published_8q_quickdraw_v2.tex)).
   Without a body justification, that choice looks arbitrary.
2. **§3 (construction of the four topologies) is load-bearing for the
   continuous-family claim.** "The family includes FFT and BlockDCT
   as endpoints" requires showing what's between those endpoints. §3
   is exactly that.
3. **Entangled-QFT genuinely does add value.** ~0.3 dB on DIV2K-8q at
   20% retention vs. plain QFT. Real but small — the appropriate
   response is honest acknowledgment, not full removal.

The published numbers don't support the current "stable ranking"
claim, though:

| basic variant | DIV2K-8q PSNR @ kr=20% |
|---|---:|
| QFT | 30.91 |
| Entangled-QFT | 31.23 |
| TEBD | 31.28 |
| MERA | 31.28 |

TEBD ≈ MERA ≈ Entangled-QFT to within 0.05 dB. The current paper text
asserts "Entangled QFT > QFT > MERA > TEBD" which the data does not
support cleanly.

## Recommended structural change

| current location | action | rationale |
|---|---|---|
| §3 (topology construction) | **keep as-is** | needed for "continuous family" framing |
| §4.2 (full topology comparison subsection) | **shorten to one body paragraph** | empirically weak ranking; small effect size |
| (new) Appendix B | **add full per-(dataset, kr) comparison** | preserves data for interested readers |
| Contribution list | **drop (iii) "topology ranking"** | promotes weakly-supported claim to headline status |

## Suggested replacement paragraph for §4.2

Drop the existing subsection content and replace with something like:

> *Within the basic-variant family we trained four topologies:
> separable QFT, Entangled-QFT (with cross-dimensional coupling),
> TEBD-style ring connectivity, and MERA-inspired hierarchy. On
> DIV2K-8q at 20% retention, Entangled-QFT (31.23 dB) marginally
> outperformed QFT (30.91 dB), with TEBD and MERA performing within
> 0.05 dB of Entangled-QFT. The choice of topology among reasonable
> variants thus contributes ≤0.4 dB — an order of magnitude smaller
> than the gain from block-wrapping (§4.4). Full per-(dataset,
> retention) numbers are deferred to Appendix B.*

Four sentences. The body now (a) acknowledges the variants exist,
(b) honestly states the small effect size, (c) routes the curious
reader to the appendix, (d) sets up the much-larger block-wrapping
effect as the actual story.

## Suggested Appendix B layout

```
Appendix B  Topology variation: full numerical comparison

  B.1  Per-(dataset, retention) PSNR / SSIM table for QFT,
       Entangled-QFT, TEBD, MERA on DIV2K-8q and Quick Draw
       (port from the current main-paper §4.2 table).

  B.2  Per-(dataset, retention) PSNR / SSIM table for the three
       block-wrapped variants (Blocked-QFT, Blocked-RichBasis,
       Blocked-RealRichBasis), since Blocked-QFT and Blocked-
       RichBasis also drop out of the trimmed headline table.

  B.3  Brief discussion: why the differences are small (≤0.4 dB
       within basic, ≤0.25 dB within block-wrapped). Likely
       explanation: all four basic topologies parameterize roughly
       the same space at this depth; the bottleneck is block
       structure, not topology fine-tuning within a fixed structure.

  B.4  (Optional) MERA at QuickDraw is undefined (m+n=10 not a
       power of 2); document the constraint and the resulting
       N/A entries.
```

## Summary of cuts

The combined effect of #2 (this issue's parent) + the changes above:

- **Headline table**: 10 rows → 6 (3 classical + 3 learned).
- **Headline contributions**: 4 items → 4 items (replace
  contribution-iii topology claim with the data-regime finding).
- **§4.2 in body**: full subsection → 1 paragraph + "see Appendix B."
- **New Appendix B**: full topology + block-variant tables for
  readers who want the details.

The paper retains all empirical content; only the framing changes.
The framing change brings the paper's epistemic posture in line with
what the data actually supports.

---

Companion files in `pdft-benchmarks`:
- [`docs/paper-recommendations.md`](https://github.com/zazabap/pdft-benchmarks/blob/main/docs/paper-recommendations.md)
  — full memo covering both issues.
- [`tables/published_8q_quickdraw_v2.tex`](https://github.com/zazabap/pdft-benchmarks/blob/main/tables/published_8q_quickdraw_v2.tex)
  — drop-in 6-row replacement table.
