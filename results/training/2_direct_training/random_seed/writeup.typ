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
`QFTBasis(8,8)` operator from *Haar-random* init under the *block-growth* (`bg`),
*left*#sym.arrow.r*right* (`lr`) and *right*#sym.arrow.r*left* (`rl`) unfreeze
orderings, for #nseed seeds each. Every seed $s$ reseeds *everything trainable* —
the Haar gate init, the 50-image training-batch subsample (from the fixed
500-image pool), and the training RNG — while the held-out *test set is held
fixed* (canonical seed 42) so the endpoint-PSNR spread reflects the trained
operator, not the test draw. Training minimises the top-*20%*-coefficient MSE
(the headline elsewhere is top-10%); test PSNR is scored at four keep ratios
$rho$.

= The random initialisations are genuinely different

#figure(image("div2k_8q/figures/init_distribution.svg", width: 92%),
  caption: [*Left:* the untrained random operator's initial top-$k$ MSE loss
  $L_0$ on a common fixed batch, per seed — it spans
  #f1(idist.L0_stats.min)#sym.dash.en#f1(idist.L0_stats.max)
  (mean #f1(idist.L0_stats.mean), $sigma = #f1(idist.L0_stats.std)$), so the
  seeds start from genuinely different points. *Right:* a 2-D PCA of the
  #nseed init parameter vectors; the top two components explain only
  #f1(idist.pca.explained_var_ratio.at(0)*100)% + #f1(idist.pca.explained_var_ratio.at(1)*100)%
  of the variance, i.e. the inits are spread across many near-orthogonal
  directions rather than clustered.])

= Endpoint variance

#figure(image("div2k_8q/figures/seed_variance.svg", width: 100%),
  caption: [Per-ordering test PSNR across #nseed seeds. *Left:* mean
  $plus.minus sigma$ band (+ min#sym.dash.en max whiskers) vs $rho$, against the
  block-FFT 8#sym.times#8 baseline (dashed) and block-DCT 8#sym.times#8 (dotted).
  *Middle:* per-seed scatter at $rho{=}.20$. *Right:* the endpoint distribution
  (histogram + fitted normal).])

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

Despite starting from #nseed genuinely different random inits (previous
section), every ordering's endpoint sits in a tight band: at $rho{=}.20$ the
per-seed standard deviation is
#orderings.map(o => olab.at(o) + " " + f2(agg(o, "0.2").std)).join(", ") dB.
Crucially the *worst single seed of all* still clears the classical block-FFT
8#sym.times#8 reference (#f2(fft20) dB \@$rho{=}.20$): the minimum is
#orderings.map(o => f2(agg(o, "0.2").min)).join(" / ") dB for bg / lr / rl. So
*neither the random initialisation nor the gate-release order moves the endpoint
much*, and the learned QFT is *always* above block-FFT — though block-DCT
8#sym.times#8 (#f2(cl.block_dct_8.psnr.at("0.2")) dB) remains the strongest
classical transform here, ahead of the QFT family.

#figure(image("div2k_8q/figures/seed_training_dynamics.svg", width: 100%),
  caption: [Per-seed training dynamics: every seed's top-$k$ MSE staircase (one
  per unfreeze stage), one panel per ordering. Different Haar starts descend
  along slightly different paths yet converge to the same attractor.])
