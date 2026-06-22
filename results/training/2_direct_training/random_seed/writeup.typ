#set page(paper: "us-letter", margin: (x: 0.9in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set block(above: 0.9em, below: 0.4em)
#show raw: set text(size: 8.5pt)

#let ss = json("div2k_8q/seed_sweep.json")
#let cl = json("reference/classical_div2k.json")
#let idist = json("div2k_8q/reference/init_distribution.json")
#let f1(x) = str(calc.round(x, digits: 1))
#let f2(x) = str(calc.round(x, digits: 2))
#let orderings = ("bg", "lr", "rl")
#let olab = (bg: "block-growth", lr: "left" + sym.arrow.r + "right",
             rl: "right" + sym.arrow.r + "left")
#let rhos = ("0.05", "0.1", "0.15", "0.2")
#let agg(o, r) = ss.per_ordering.at(o).agg.at(r)
#let nseed = agg("bg", "0.2").n
#let fft20 = cl.block_fft_8.psnr.at("0.2")

#align(center)[
  #text(size: 15pt, weight: "bold")[Seed robustness of the gate-unfreeze QFT]
  #v(2pt)
  #text(size: 10.5pt)[#nseed random-initialisation seeds #sym.times three
  unfreeze orderings, top-20% training objective, DIV2K-8q]
  #v(3pt)
  #text(size: 9pt, fill: rgb("#555"))[Generated #datetime.today().display("[year]-[month]-[day]")]
]

= Question

Does the gate-unfreeze QFT endpoint depend on the random draw? We retrain the
`QFTBasis(8,8)` operator from *Haar-random* init under the #olab.bg (`bg`),
#olab.lr (`lr`) and #olab.rl (`rl`) unfreeze
orderings, for #nseed seeds each. Every seed $s$ reseeds *everything trainable* —
the Haar gate init, the 50-image training-batch subsample (from the fixed
500-image pool), and the training RNG — while the held-out *test set is held
fixed* (canonical seed 42) so the endpoint-PSNR spread reflects the trained
operator, not the test draw. Training minimises the top-*20%*-coefficient MSE
(the headline elsewhere is top-10%); test PSNR is scored at four keep ratios
$rho$.

= The random initialisations are genuinely different

#figure(
  grid(columns: (1fr, 1.15fr), column-gutter: 6pt, align: horizon,
    image("div2k_8q/figures/init_distribution_L0.svg", width: 100%),
    image("div2k_8q/figures/init_distribution_pca.svg", width: 100%)),
  caption: [*(a)* the untrained random operator's initial top-$k$ MSE loss
  $L_0$ on a common fixed batch, per seed — it spans
  #f1(idist.L0_stats.min)#sym.dash.en#f1(idist.L0_stats.max)
  (mean #f1(idist.L0_stats.mean), $sigma = #f1(idist.L0_stats.std)$), so the
  seeds start from genuinely different points. *(b)* a 2-D PCA of the
  #nseed init parameter vectors; the top two components explain only
  #f1(idist.pca.explained_var_ratio.at(0)*100)% + #f1(idist.pca.explained_var_ratio.at(1)*100)%
  of the variance, i.e. the inits are spread across many near-orthogonal
  directions rather than clustered.])

= Endpoint variance

#figure(
  stack(spacing: 7pt,
    grid(columns: (1fr, 1fr), column-gutter: 6pt, align: horizon,
      image("div2k_8q/figures/seed_variance_band.svg", width: 100%),
      image("div2k_8q/figures/seed_variance_hist.svg", width: 100%)),
    image("div2k_8q/figures/seed_scatter_ratios.svg", width: 100%)),
  caption: [Per-ordering test PSNR across #nseed seeds. *Top:* *(a)* mean
  $plus.minus sigma$ band (+ min#sym.dash.en max whiskers) vs $rho$, against
  block-FFT 8#sym.times#8 (dashed) and block-DCT 8#sym.times#8 (dotted); and
  *(c)* the endpoint distribution at $rho{=}.20$ (histogram + fitted normal).
  *Bottom:* *(b)* per-seed scatter at four keep ratios
  ($rho = 0.01, 0.05, 0.10, 0.20$) — the trained bases beat block-FFT at every
  rate and exceed both classical references by #sym.tilde 6 dB at $rho{=}.01$,
  while block-DCT overtakes them for $rho >= .05$.])

// Inline table: per-ordering mean +/- sigma, plus the worst single seed @ .20.
#let best = (r) => calc.max(..orderings.map(o => agg(o, r).mean))
#let cell(o, r) = {
  let a = agg(o, r)
  let s = f2(a.mean) + sym.plus.minus + f2(a.std)
  if calc.abs(a.mean - best(r)) < 1e-9 { text(weight: "bold")[#s] } else [#s]
}
#align(center)[#table(
  columns: 6, align: (left, right, right, right, right, right),
  stroke: 0.4pt + luma(180), inset: (x: 6pt, y: 3pt),
  table.header([*ordering*], [$rho{=}.05$], [$rho{=}.10$], [$rho{=}.15$],
               [$rho{=}.20$], [*min\@.20*]),
  ..orderings.map(o => (
    [#olab.at(o)], cell(o, "0.05"), cell(o, "0.1"), cell(o, "0.15"),
    cell(o, "0.2"), [#f2(agg(o, "0.2").min)],
  )).flatten(),
  table.hline(stroke: 0.6pt),
  [block-FFT 8#sym.times#8], [#f2(cl.block_fft_8.psnr.at("0.05"))],
  [#f2(cl.block_fft_8.psnr.at("0.1"))], [#f2(cl.block_fft_8.psnr.at("0.15"))],
  [#f2(fft20)], [#text(fill: luma(150))[—]],
  [block-DCT 8#sym.times#8], [#f2(cl.block_dct_8.psnr.at("0.05"))],
  [#f2(cl.block_dct_8.psnr.at("0.1"))], [#f2(cl.block_dct_8.psnr.at("0.15"))],
  [#f2(cl.block_dct_8.psnr.at("0.2"))], [#text(fill: luma(150))[—]],
)]
#align(center, text(8pt, fill: luma(90))[mean $plus.minus sigma$ test PSNR (dB),
  $n = #nseed$ random-init seeds per ordering; *min\@.20* is the single worst
  seed at $rho{=}.20$.])

= Reading

#let n_below(o) = ss.per_ordering.at(o).per_seed.values().filter(v => v.at("0.2") < fft20).len()
#let n_above_all = nseed * 3 - n_below("bg") - n_below("lr") - n_below("rl")

Despite starting from #nseed genuinely different random inits (previous
section), each ordering's endpoint sits in a narrow band: at $rho{=}.20$ the
per-seed standard deviation is
#orderings.map(o => olab.at(o) + " " + f2(agg(o, "0.2").std)).join(", ") dB. The
*release order*, not the random draw, is the dominant factor: #olab.bg and
#olab.lr are tight and sit highest, while #olab.rl
is #sym.tilde 2#sym.times wider and lowest — the dynamics figure below explains
why (rl starts from a far worse loss and its seeds stay spread until the final
unfreeze stages).

Against the classical block-FFT 8#sym.times#8 reference (#f2(fft20) dB
\@$rho{=}.20$): #olab.bg and #olab.lr clear it for *every*
seed (worst seed #f2(agg("bg", "0.2").min) / #f2(agg("lr", "0.2").min) dB;
margins +#f2(agg("bg", "0.2").min - fft20) / +#f2(agg("lr", "0.2").min - fft20)
dB). #olab.rl clears it for #(nseed - n_below("rl")) of #nseed
seeds; #n_below("rl") low-tail seeds dip below by at most
#f2(fft20 - agg("rl", "0.2").min) dB. Overall #n_above_all of #(nseed * 3) runs
(#f1(n_above_all / (nseed * 3) * 100)%) clear block-FFT. So the gate-unfreeze
QFT is *essentially always* above block-FFT — unconditionally for bg/lr, and for
#f1(100 * (nseed - n_below("rl")) / nseed)% of the highest-variance rl ordering
— while block-DCT 8#sym.times#8 (#f2(cl.block_dct_8.psnr.at("0.2")) dB) remains
the strongest classical transform here, ahead of the QFT family.

The per-seed endpoints are *not* Gaussian: a Shapiro#sym.dash.en Wilk test
rejects normality for all three orderings ($p < 10^(-7)$), because seeds settle
into a few discrete attractor basins of the very-flat top-$k$ MSE valley rather
than scattering smoothly. The fitted normal in the endpoint panel is a visual
summary of location and spread, not a claim of Gaussianity.

#figure(
  grid(columns: 3, column-gutter: 4pt, align: horizon,
    image("div2k_8q/figures/seed_dynamics_bg.svg", width: 100%),
    image("div2k_8q/figures/seed_dynamics_lr.svg", width: 100%),
    image("div2k_8q/figures/seed_dynamics_rl.svg", width: 100%)),
  caption: [Per-seed training dynamics: every seed's top-$k$ MSE staircase (one
  per unfreeze stage), one panel per ordering, on a shared y-range. Different
  Haar starts descend along different paths: #olab.bg *(a)* collapses fast and
  stays tight, whereas #olab.rl *(c)* starts far higher and its seeds remain
  spread until the final stages — visually accounting for the endpoint variances
  above. #olab.lr *(b)* is intermediate.])

#let blk = json("div2k_8q/block_structure.json")
#let frozen_per_dim = (16 - blk.n_mix_row - blk.n_mix_col) / 2
#let cp_active = calc.round(blk.cp_active_frac * 56)

= Emergent block structure of the trained QFT

We close by reading the *canonical published* trained QFT
(`QFTBasis(8,8)`, seed #blk.seed) directly, making the paper's central
structural claim concrete (sec5.3): the trained full-image QFT factors itself
into a block code with no block prior. Of its 16 Hadamard-role gates (8 per
dimension), #frozen_per_dim per dimension collapse to non-mixing *Pauli-Z/X*
gates (the off-diagonal/diagonal modulus drops to $approx 0.3%$) — classical
block indices — while #cp_active of the 56 controlled-phase gates stay active, so
the intra-block transform stays rich. The #blk.n_mix_row still-mixing row qubits
and #blk.n_mix_col column qubits leave an effective
$#blk.block_row times #blk.block_col$-pixel block.

#figure(
  image("div2k_8q/figures/block_gate_collapse.svg", width: 100%),
  caption: [Hadamard-role gates of the trained QFT, one cell per qubit and
  dimension, shaded by the mixing score $2|a||b|$ (1 = Hadamard, 0 = frozen) and
  labelled by type. Four of the eight Hadamards per dimension keep mixing
  frequencies; the other four freeze to Pauli-Z (or X) and act as classical block
  indices.])

The factorization shows directly in the dense operator and its spectrum.

#figure(
  grid(columns: (1.05fr, 1fr), column-gutter: 8pt, align: horizon,
    image("div2k_8q/figures/block_operator_heatmap.svg", width: 100%),
    image("div2k_8q/figures/block_leakage_sweep.svg", width: 100%)),
  caption: [*Left:* $log|W|$ of the trained QFT's 1-D operator factor,
  block-diagonal at the emergent 16-pixel block (off-block energy below $0.01%$).
  *Right:* off-block energy vs candidate block size — the knee at $b=16$ (dotted)
  shows the operator is block-structured at that specific scale and not below; the
  untrained closed-form QFT (not shown) stays high at every scale.])

The same factorization is visible directly in frequency space.

#figure(
  image("div2k_8q/figures/block_freq_spectrum.svg", width: 100%),
  caption: [Mean test-set power spectrum (peak-normalized
  $log_10 overline(|F|^2)$, averaged over the 50 DIV2K test images) of the
  untrained *global QFT* (left) and the *trained QFT* (right). The global
  transform packs energy into a single low-frequency lobe; the trained transform
  tiles into a $16 times 16$ grid of repeated sub-spectra — the block-periodic
  signature of a block transform, the same structure the $8 times 8$ block
  references produce in Fig. 4 of the main text.])
