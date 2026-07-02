#set page(paper: "us-letter", margin: (x: 0.9in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set block(above: 0.9em, below: 0.4em)
#show raw: set text(size: 8.5pt)

#let man = json("div2k_8q/manifest.json")
#let cls = json("reference/classical_dct4.json")
#let adx = json("reference/adam_exact_baseline.json")
#let adr = json("reference/adam_random_seed_sweep.json")
#let f1(x) = str(calc.round(x, digits: 1))
#let f2(x) = str(calc.round(x, digits: 2))
#let f4(x) = str(calc.round(x, digits: 4))
#let rhos = ("0.05", "0.1", "0.2")
#let runkeys = ("exact/fwd", "exact/rev", "random/fwd", "random/rev")

#align(center)[
  #text(size: 15pt, weight: "bold")[Environment-sweep training of the DCT-IV operator]
  #v(2pt)
  #text(size: 10.5pt)[DMRG-style closed-form gate updates — no Adam, no learning
  rate — on the controlled O(2)-twiddle DCT-IV, DIV2K-8q: two inits
  $times$ two sweep orders]
  #v(3pt)
  #text(size: 9pt, fill: rgb("#555"))[Generated #datetime.today().display("[year]-[month]-[day]")]
]

#block(fill: rgb("#eef4fb"), stroke: 0.5pt + rgb("#9db8dd"), inset: 8pt,
       radius: 3pt, width: 100%)[
*What a sweep update is.* For a circuit in which gate $G_i$ enters linearly,
the derivative of the loss w.r.t. that gate *is its environment* $E_i$ — the
network contracted with everything except $G_i$. To first order,
$Delta L prop angle.l E_i, Delta G angle.r$, and the orthogonal matrix
minimizing the linearized loss is closed-form: with SVD $E_i = U Sigma V^top$,
$G_i^* = -U V^top$ (for the $Delta$-sign phase gate,
$phi^* = pi + arg E_(11)$). We visit the #man.runs.at("exact/fwd").n_visits
gate slots one at a time (Gauss-Seidel — each visit re-derives $E_i$ with all
previous updates in place), jump to $G_i^*$ behind a backtracking acceptance
check ($t in {1, 1/2, 1/4, 1/8}$ along $"polar"((1-t) G_i + t G_i^*)$, accept
only on strict loss decrease), and repeat full sweeps until a plateau. The
fixed-batch loss is therefore *monotone non-increasing by construction*, with
*no learning rate, momentum, or schedule* — the antithesis of the Riemannian
Adam used everywhere else in this project, which consumes the same
environments as small simultaneous scheduled steps.
]

= Question

Riemannian Adam reaches $approx$#f1(adx.psnr.at("0.2")) dB \@$rho{=}.20$ from
the exact DCT-IV init and $approx$#f1(adr.psnr_mean.at("0.2")) dB from random
inits (1008 steps). Does hyperparameter-free coordinate-wise optimization —
the classic tensor-network sweep — land in the same place? From both a
*random* init and the *exact* DCT-IV init, and does the *visit order*
(`fwd` = circuit emission order, `rev` = reversed) matter?

= Setup

`DCT4Basis(8, 8, parametrization: "controlled")` (pdft PR \#24, commit
`5365a5a`): 214 gates / 2200 real entries — O(2) singles and CRY twiddle
leaves, O(4) mirror gates, $Delta$-sign phase gates. Loss: top-#(str(calc.round(man.topk_ratio * 100)))%
reconstruction MSE ($k = #man.k_train$) on a *fixed* batch of the first
#man.batch_n images of the canonical seed-42 DIV2K-8q train pool — fixed, so
every run is deterministic. Test PSNR on the canonical 50-image test set.
Random init: haar-SO per gate (seed #man.runs.at("random/fwd").init_seed,
shared across orders), the same init family as the seed sweep. Sweeps stop at
relative per-sweep improvement $< #man.rel_tol$ or zero accepted visits
(cap #man.max_sweeps).

= Sweep dynamics

#figure(image("figures/sweep_dynamics.svg", width: 100%),
  caption: [Fixed-batch top-10% loss vs *cumulative gate visit* (top: exact
  init, bottom: random init; blue = `fwd`, vermilion = `rev`; light rules =
  sweep boundaries). Labels mark the largest single-visit drops (gate kind
  [qubits]); endpoints are annotated with test PSNR\@$rho{=}.20$. Every
  accepted visit decreases the loss by construction.])

#figure(image("figures/sweep_convergence.svg", width: 100%),
  caption: [Per-sweep endpoint loss, test PSNR\@$rho{=}.20$, and accepted-visit
  fraction. The accepted fraction decaying toward zero is the sweep "drying
  up" — the plateau criterion.])

= Reconstruction PSNR (DIV2K)

#let bests = (:)
#for rk in rhos { bests.insert(rk, calc.max(..runkeys.map(k => man.runs.at(k).psnr_final.at(rk)))) }
#align(center)[#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right),
  stroke: 0.4pt + luma(180), inset: (x: 6pt, y: 3pt),
  table.header([*configuration*], [$rho{=}.05$], [$rho{=}.10$], [$rho{=}.20$],
               [*train MSE*], [*sweeps*]),
  ..runkeys.map(k => {
    let r = man.runs.at(k)
    (raw("sweep · " + k),
     ..rhos.map(rk => {
       let v = r.psnr_final.at(rk)
       if calc.abs(v - bests.at(rk)) < 1e-4 {
         text(fill: rgb("#0072B2"), weight: "bold")[#f1(v)]
       } else [#f1(v)]
     }),
     [#f4(r.final_loss)], [#str(r.n_sweeps)])
  }).flatten(),
  table.hline(stroke: 0.8pt),
  [Adam · exact init (1008 steps)],
  ..rhos.map(rk => [#f1(adx.psnr.at(rk))]), [#f4(adx.final_loss)],
  [#text(fill: luma(150))[—]],
  [Adam · random init (mean of #adr.n_seeds seeds)#super[$dagger$]],
  ..rhos.map(rk => [#f1(adr.psnr_mean.at(rk)) $plus.minus$ #f1(adr.psnr_std.at(rk))]),
  [#text(fill: luma(150))[—]], [#text(fill: luma(150))[—]],
  table.hline(stroke: 0.5pt),
  [classical · #cls.canonical_dct4.label],
  ..rhos.map(rk => [#f1(cls.canonical_dct4.psnr.at(rk))]),
  [#text(fill: luma(150))[—]], [#text(fill: luma(150))[—]],
  [classical · #cls.block_dct_8.label],
  ..rhos.map(rk => [#f1(cls.block_dct_8.psnr.at(rk))]),
  [#text(fill: luma(150))[—]], [#text(fill: luma(150))[—]],
  [classical · #cls.block_fft_8.label],
  ..rhos.map(rk => [#f1(cls.block_fft_8.psnr.at(rk))]),
  [#text(fill: luma(150))[—]], [#text(fill: luma(150))[—]],
)]
#align(center, text(8pt, fill: luma(90))[*train MSE* = final fixed-batch
top-10% loss (sweep runs; Adam-exact shown on its own 500-image-pool loss —
different batch, indicative only). #super[$dagger$]the random-init Adam
reference trained at top-20% (its sweep predates the top-10% convention); the
rate-matched probe in the unfreeze study bounds the effect of the top-$k$
choice at $lt.eq 0.07$ dB.])

= Reading

// FILL AFTER RESULTS: 2-3 paragraphs. Address, with the actual numbers:
// (i) does the sweep from the exact init hold/improve the classical DCT-IV
// and match Adam-exact? (ii) how close does the random-init sweep get to the
// Adam-random band — same basin or short of it, and does fwd vs rev matter?
// (iii) which gates carry the largest drops (from the dynamics labels), and
// how fast does the accepted fraction dry up?
