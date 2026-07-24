#set page(paper: "us-letter", margin: (x: 0.9in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set block(above: 0.9em, below: 0.4em)
#show raw: set text(size: 8.5pt)

#let ss = json("seed_sweep.json")
#let ref = json("reference/classical_dct4.json")
#let f1(x) = str(calc.round(x, digits: 1))
#let f2(x) = str(calc.round(x, digits: 2))
#let agg(r) = ss.agg.at(r)
#let nseed = agg("0.2").n
#let epochs = ss.epochs
#let fft20 = ref.block_fft_8.psnr.at("0.2")
#let dct20 = ref.block_dct_8.psnr.at("0.2")
#let canon20 = ref.canonical_dct4.psnr.at("0.2")

#align(center)[
  #text(size: 15pt, weight: "bold")[Seed robustness of the controlled (O(2)-twiddle) DCT-IV basis]
  #v(2pt)
  #text(size: 10.5pt)[#nseed random real-orthogonal initialisations, normal
  top-20% training (#epochs steps) on the full 500-image DIV2K-8q train pool]
  #v(3pt)
  #text(size: 9pt, fill: rgb("#555"))[Generated #datetime.today().display("[year]-[month]-[day]")]
]

= Setup

The *controlled* DCT-IV parametrization (`DCT4Basis(8, 8,
parametrization: "controlled")`) trains the affine twiddle as a single-angle
`CRY` gate on *O(2)* and keeps the mirror Q/R CNOTs as fixed `CX` routing —
*408 trainable parameters vs 2872* for the dense-O(4) form (7#sym.times fewer),
and #sym.tilde 2#sym.times faster per optimizer step. As in the o4 seed study,
each of #nseed seeds $s$ reseeds *everything trainable* (a fresh Haar
real-orthogonal init via `dct4_random_controlled_basis(8, 8, s)`, the training
RNG `train_basis_batched(seed = s)`), but here trained on the *full 500-image*
pool with mini-batches of 50 (not a 50-image subsample), one cosine-LR top-20%
run of #epochs steps. The held-out *test set is held fixed* (canonical seed 42,
50 images). Test PSNR is scored at four keep ratios $rho$.

= Endpoint variance

#figure(
  stack(spacing: 7pt,
    grid(columns: (1.3fr, 0.9fr, 1fr), column-gutter: 6pt, align: horizon,
      image("figures/seed_variance_band.svg", width: 100%),
      image("figures/seed_variance_scatter.svg", width: 100%),
      image("figures/seed_variance_hist.svg", width: 100%))),
  caption: [Test PSNR across #nseed seeds. *(a)* mean $plus.minus sigma$ band
  (+ min#sym.dash.en max whiskers) vs $rho$, against the untrained exact DCT-IV
  (dashdot), block-FFT 8#sym.times#8 (dashed) and block-DCT 8#sym.times#8
  (dotted). *(b)* per-seed scatter at $rho{=}.20$ with the mean$plus.minus sigma$
  bar. *(c)* the endpoint distribution at $rho{=}.20$ (histogram + fitted
  normal); references marked.])

#align(center)[#table(
  columns: 6, align: (left, right, right, right, right, right),
  stroke: 0.4pt + luma(180), inset: (x: 6pt, y: 3pt),
  table.header([*basis*], [$rho{=}.01$], [$rho{=}.05$], [$rho{=}.10$],
               [$rho{=}.20$], [*min\@.20*]),
  [controlled DCT-IV],
  text(weight: "bold")[#f2(agg("0.01").mean) #sym.plus.minus #f2(agg("0.01").std)],
  text(weight: "bold")[#f2(agg("0.05").mean) #sym.plus.minus #f2(agg("0.05").std)],
  text(weight: "bold")[#f2(agg("0.1").mean) #sym.plus.minus #f2(agg("0.1").std)],
  text(weight: "bold")[#f2(agg("0.2").mean) #sym.plus.minus #f2(agg("0.2").std)],
  [#f2(agg("0.2").min)],
  table.hline(stroke: 0.6pt),
  [DCT-IV (untrained)], [#f2(ref.canonical_dct4.psnr.at("0.01"))],
  [#f2(ref.canonical_dct4.psnr.at("0.05"))], [#f2(ref.canonical_dct4.psnr.at("0.1"))],
  [#f2(canon20)], [#text(fill: luma(150))[—]],
  [block-FFT 8#sym.times#8], [#f2(ref.block_fft_8.psnr.at("0.01"))],
  [#f2(ref.block_fft_8.psnr.at("0.05"))], [#f2(ref.block_fft_8.psnr.at("0.1"))],
  [#f2(fft20)], [#text(fill: luma(150))[—]],
  [block-DCT 8#sym.times#8], [#f2(ref.block_dct_8.psnr.at("0.01"))],
  [#f2(ref.block_dct_8.psnr.at("0.05"))], [#f2(ref.block_dct_8.psnr.at("0.1"))],
  [#f2(dct20)], [#text(fill: luma(150))[—]],
)]
#align(center, text(8pt, fill: luma(90))[mean $plus.minus sigma$ test PSNR (dB),
  $n = #nseed$ random-init seeds; *min\@.20* is the single worst seed at
  $rho{=}.20$.])

= Reading

#let n_below_canon = ss.per_seed.values().filter(v => v.at("0.2") < canon20).len()
#let n_below_fft = ss.per_seed.values().filter(v => v.at("0.2") < fft20).len()

From #nseed Haar real-orthogonal inits, #epochs steps of top-20% training on the
*full 500-image* pool concentrate the endpoint into a narrow band: at
$rho{=}.20$ the per-seed standard deviation is #f2(agg("0.2").std) dB
(mean #f2(agg("0.2").mean), min #f2(agg("0.2").min), max #f2(agg("0.2").max)).
Against the untrained exact DCT-IV (#f2(canon20) dB @ $rho{=}.20$),
#(nseed - n_below_canon) of #nseed trained seeds match-or-exceed it; against
block-FFT 8#sym.times#8 (#f2(fft20) dB), #(nseed - n_below_fft) clear it.

Two things this study establishes. *(i)* From a random start the controlled
DCT-IV plateaus *just below* the fixed transform — reaching the 31#sym.dash.en#33
dB of exact-init training requires the *canonical exact-DCT-IV init*, not more
data (moving the pool from 50 to 500 images added only #sym.tilde 0.04 dB @
$rho{=}.20$; the lever is initialisation, not data). *(ii)* Despite fixing the
mirror routing and using *7#sym.times fewer* trainable parameters, the controlled
basis still *beats the dense-O(4) form from random init* at every $rho$
(#sym.plus 0.4#sym.dash.en 1 dB) with lower variance — the fixed `CX` mirror is a
correct structural prior that leaves less of the DCT-IV to rediscover.

#figure(
  image("figures/seed_dynamics.svg", width: 78%),
  caption: [Per-seed training dynamics: every seed's top-$k$ MSE descent (one
  curve per seed) on a shared y-range. The Haar real-orthogonal starts descend
  from genuinely different, far-from-trained losses and converge into the narrow
  endpoint band above.])
