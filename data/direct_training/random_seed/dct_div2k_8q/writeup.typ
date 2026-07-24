#set page(paper: "us-letter", margin: (x: 0.9in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set block(above: 0.9em, below: 0.4em)
#show raw: set text(size: 8.5pt)

#let ss = json("seed_sweep.json")
#let ref = json("reference/classical_dct4.json")
#let f1(x) = str(calc.round(x, digits: 1))
#let f2(x) = str(calc.round(x, digits: 2))
#let rhos = ("0.01", "0.05", "0.1", "0.2")
#let agg(r) = ss.agg.at(r)
#let nseed = agg("0.2").n
#let epochs = ss.epochs
#let fft20 = ref.block_fft_8.psnr.at("0.2")
#let dct20 = ref.block_dct_8.psnr.at("0.2")
#let canon20 = ref.canonical_dct4.psnr.at("0.2")

// L0 spread of the random inits (initial top-k MSE per seed, own batch).
#let l0vals = ss.init_loss_per_seed.values()
#let l0min = calc.min(..l0vals)
#let l0max = calc.max(..l0vals)
#let l0mean = l0vals.sum() / l0vals.len()

#align(center)[
  #text(size: 15pt, weight: "bold")[Seed robustness of the learnable DCT-IV basis]
  #v(2pt)
  #text(size: 10.5pt)[#nseed random real-orthogonal initialisations, normal
  top-20% training (#epochs steps), DIV2K-8q]
  #v(3pt)
  #text(size: 9pt, fill: rgb("#555"))[Generated #datetime.today().display("[year]-[month]-[day]")]
]

= Question

Does normal training of the learnable DCT-IV basis depend on the random draw?
`DCT4Basis(8,8)` is the real-orthogonal, ancilla-free analogue of the QFT: at
its *canonical* init the forward operator is the exact bit-reversed orthonormal
DCT-IV (#f2(canon20) dB @ $rho{=}.20$ on the held-out test set). Following the
QFT seed study's init policy, we do *not* start there: each of #nseed seeds $s$
reseeds *everything trainable* — a fresh *Haar real-orthogonal* gate init
(`dct4_random_basis(8,8,s)`: SO(2)/SO(4) on the gates, a random sign on the
controlled-phase gate, so the operator stays real-orthogonal), the 50-image
training-batch subsample (from the fixed 500-image pool), and the training RNG —
while the held-out *test set is held fixed* (canonical seed 42). Unlike the QFT
writeup the operator is trained with *normal* batched training
(`pdft.train_basis_batched`, one cosine-LR top-20% run of #epochs steps), not
the gate-unfreeze sweep. Test PSNR is scored at four keep ratios $rho$.

= The random initialisations are genuinely different

#figure(
  grid(columns: (1fr, 1.15fr), column-gutter: 6pt, align: horizon,
    image("figures/init_distribution_L0.svg", width: 100%),
    image("figures/init_distribution_pca.svg", width: 100%)),
  caption: [*(a)* the untrained random operator's initial top-$k$ MSE loss
  $L_0$ per seed — it spans #f1(l0min)#sym.dash.en#f1(l0max)
  (mean #f1(l0mean)), so the seeds start from genuinely different,
  far-from-trained points. *(b)* a 2-D PCA of the #nseed init parameter
  vectors; the inits are spread across many near-orthogonal directions rather
  than clustered.])

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

// Inline table: trained DCT-IV mean +/- sigma + min@.20, with reference rows.
#align(center)[#table(
  columns: 6, align: (left, right, right, right, right, right),
  stroke: 0.4pt + luma(180), inset: (x: 6pt, y: 3pt),
  table.header([*basis*], [$rho{=}.01$], [$rho{=}.05$], [$rho{=}.10$],
               [$rho{=}.20$], [*min\@.20*]),
  [trained DCT-IV],
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

#let n_below_fft = ss.per_seed.values().filter(v => v.at("0.2") < fft20).len()
#let n_below_canon = ss.per_seed.values().filter(v => v.at("0.2") < canon20).len()

Despite starting from #nseed genuinely different Haar real-orthogonal inits
(far from trained — $L_0$ mean #f1(l0mean)), #epochs steps of normal top-20%
training concentrate the endpoint into a narrow band: at $rho{=}.20$ the
per-seed standard deviation is #f2(agg("0.2").std) dB
(mean #f2(agg("0.2").mean), min #f2(agg("0.2").min), max #f2(agg("0.2").max)).
So the random draw causes only small turbulence in the endpoint.

Against the untrained exact DCT-IV (#f2(canon20) dB @ $rho{=}.20$),
#(nseed - n_below_canon) of #nseed trained seeds match-or-exceed it; against
block-FFT 8#sym.times#8 (#f2(fft20) dB), #(nseed - n_below_fft) of #nseed
clear it. Block-DCT 8#sym.times#8 (#f2(dct20) dB) remains the strongest
classical transform here.

The per-seed endpoints'
#if agg("0.2").shapiro_p != none [ Shapiro#sym.dash.en Wilk normality $p =
#f2(agg("0.2").shapiro_p * 1000) times 10^(-3)$ ] else [ normality ]
is reported in `seed_sweep.json`; the fitted normal in the endpoint panel is a
visual summary of location and spread.

#figure(
  image("figures/seed_dynamics.svg", width: 78%),
  caption: [Per-seed training dynamics: every seed's top-$k$ MSE descent (one
  curve per seed) on a shared y-range. The Haar real-orthogonal starts descend
  from genuinely different, far-from-trained losses and converge into the
  narrow endpoint band above.])
