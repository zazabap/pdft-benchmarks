#set page(paper: "us-letter", margin: (x: 0.9in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set block(above: 0.9em, below: 0.4em)
#show raw: set text(size: 8.5pt)

#let man = json("div2k_8q/manifest.json")
#let cls = json("reference/classical_dct4.json")
#let adx = json("reference/adam_exact_baseline.json")
#let adr = json("reference/adam_random_seed_sweep.json")
#let f1(x) = {
  let s = str(calc.round(x, digits: 1))
  if s.contains(".") { s } else { s + ".0" }
}
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
$Delta L prop chevron.l E_i, Delta G chevron.r$, and the orthogonal matrix
minimizing the linearized loss is closed-form: with SVD $E_i = U Sigma V^top$,
$G_i^* = -U V^top$ (for the $Delta$-sign phase gate,
$phi^* = pi + arg E_(11)$). We visit the 214
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
leaves, O(4) mirror gates, $Delta$-sign phase gates (@fig-order). Loss: top-#(str(calc.round(man.topk_ratio * 100)))%
reconstruction MSE ($k = #man.k_train$) on a *fixed* batch of the first
#man.batch_n images of the canonical seed-42 DIV2K-8q train pool — fixed, so
every run is deterministic. Test PSNR on the canonical 50-image test set.
Random init: haar-SO per gate (seed #man.runs.at("random/fwd").init_seed,
shared across orders), the same init family as the seed sweep. Sweeps stop at
relative per-sweep improvement $< #man.rel_tol$ or zero accepted visits; all
four runs stopped on this plateau criterion (the exact runs within
#str(man.runs.at("exact/rev").n_sweeps) sweeps, the random runs — extended after
an initial 20-sweep pass — within #str(man.runs.at("random/rev").n_sweeps)),
none at a sweep cap.

#figure(image("reference/dct4_sweep_ordering.svg", width: 100%),
  caption: [The *2-D* DCT-IV$(3, 3)$ — two independent DCT-IV$(3)$ blocks, one per
  image axis (*axis 0* = $q_1..q_3$, rows; *axis 1* = $q_4..q_6$, columns) — a
  small stand-in for the DIV2K DCT-IV$(8, 8)$ (214 gates, axis 1 = $q_9..q_(16)$).
  The four trainable gate kinds are the labels that appear at the drops in
  @fig-dynamics: *H* branch Hadamard (1q), *U4* mirror-CNOT (2q), *CRY*
  single-angle twiddle (2q), *CP* $Delta$-sign controlled-phase (2q).
  Superscripts give the *sweep visit order* (the basis' tensor-slot order): a
  `fwd` sweep visits gates $1 arrow.r N$, `rev` visits $N arrow.r 1$, and — unlike
  the one-gate-per-stage unfreeze study — the whole sweep repeats with *every*
  gate revisited on *every* pass (Gauss–Seidel) until the accepted fraction
  falls to $approx 0$.])
  <fig-order>

= Sweep dynamics

#figure(image("figures/sweep_dynamics.svg", width: 100%),
  caption: [Fixed-batch top-10% loss vs *cumulative gate visit* (top: exact
  init, bottom: random init; blue = `fwd`, vermilion = `rev`; light rules =
  sweep boundaries). Labels mark the largest single-visit drops (gate kind
  [qubits]); endpoints are annotated with test PSNR\@$rho{=}.20$. Every
  accepted visit decreases the loss by construction.])
  <fig-dynamics>

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

*From the exact init the sweep barely moves — and only the mirror gates move.*
Both orders converge (#str(man.runs.at("exact/fwd").n_sweeps) and
#str(man.runs.at("exact/rev").n_sweeps) sweeps; the accepted fraction falls to
$approx 0$ in the right panel of the convergence figure) to
$approx#f1(man.runs.at("exact/fwd").psnr_final.at("0.2")) "dB"$ \@$rho{=}.20$,
just #f2(man.runs.at("exact/fwd").psnr_final.at("0.2") - man.runs.at("exact/fwd").psnr_untrained.at("0.2")) dB
above the analytic DCT-IV (#f2(man.runs.at("exact/fwd").psnr_untrained.at("0.2")) dB)
and essentially *schedule-independent* — `fwd` and `rev` finish within
#f2(calc.abs(man.runs.at("exact/fwd").psnr_final.at("0.2") - man.runs.at("exact/rev").psnr_final.at("0.2"))) dB
of each other and overlap in the dynamics staircase. Every one of the twenty
largest single-visit drops is an *O(4) mirror-CNOT* (`U4`) gate, led by
`U4[8,7]`: from the analytic init all the improvement the sweep can find lives
in the nominally "fixed-routing" mirror block, while the single-qubit
Hadamards and CRY twiddles — already optimal in the exact DCT-IV — never carry
a large drop. Yet this converged endpoint sits *#f1(adx.psnr.at("0.2") - man.runs.at("exact/fwd").psnr_final.at("0.2")) dB below Adam*
from the same init (#f1(adx.psnr.at("0.2")) dB), and still
#f1(adx.psnr.at("0.1") - man.runs.at("exact/fwd").psnr_final.at("0.1")) dB below
at the *matched* top-10% rate $rho{=}.10$: the closed-form local optimum is a
genuinely shallower basin than the one scheduled Adam reaches, not a rate
artifact.

*From the random init the sweep descends fast, then converges — well short of
Adam.* The loss plunges two orders of magnitude in the first few sweeps — the
largest drops now spread across *all* gate kinds (Hadamards, CRY twiddles,
mirror CNOTs, and the $Delta$-sign phase), led by `H[8]`, `CRY[5,8]`, `CRY[6,8]`
— and test PSNR climbs from #f1(man.runs.at("random/fwd").psnr_untrained.at("0.2"))
to $approx#f1(man.runs.at("random/rev").psnr_final.at("0.2"))$ dB \@$rho{=}.20$,
where it *flattens*: the accepted fraction decays toward $0$ and test PSNR
plateaus even as the training loss keeps inching down — mild overfitting of the
*fixed* 50-image batch, exactly as in the exact case. `fwd` converges
(#str(man.runs.at("random/fwd").n_sweeps) sweeps) at
#f2(man.runs.at("random/fwd").psnr_final.at("0.2")) dB and `rev` settles at
#f2(man.runs.at("random/rev").psnr_final.at("0.2")) dB — both $approx 1.7$–$2$ dB
below Adam from random init (#f1(adr.psnr_mean.at("0.2")) $plus.minus$
#f1(adr.psnr_std.at("0.2")) dB). This is a *converged* gap, not the
stopping artifact it would have been at the original 20-sweep cap (the random
curves reach $approx 0$ accepted-fraction in the convergence figure, not the
$0.2$ plateau of a truncated run). Order matters more here than for the exact
init (`rev` #f2(man.runs.at("random/rev").psnr_final.at("0.2")) $>$ `fwd`
#f2(man.runs.at("random/fwd").psnr_final.at("0.2")) dB).

*Same conclusion from both ends.* Run to its plateau, the environment sweep
*converges below* Riemannian Adam from either init — $approx 2.3$ dB short from
exact, $approx 2$ dB from random — even though it is monotone, hyperparameter-free,
and robust to initialization and order. Two candidate causes; one is ruled out.
The top-$k$ rate does *not* explain the gap: the exact-init deficit is
#f1(adx.psnr.at("0.1") - man.runs.at("exact/fwd").psnr_final.at("0.1")) dB even
at the *matched* $rho{=}.10$. The *fixed #(man.batch_n)-image batch*, however, is
a genuine confound — on both inits the training loss keeps falling while test
PSNR plateaus (the deterministic sweep overfits its batch), whereas the Adam
references mini-batched over the full 500-image pool. So part of the deficit is
regularization rather than optimizer quality, and cleanly separating the two
needs a larger-batch sweep (deferred). What is unambiguous is that greedy,
one-gate-at-a-time closed-form solves land in *shallower basins* than
simultaneous, momentum-assisted, schedule-annealed updates. For this family the
sweep is a reliable way to *polish or sanity-check* an operator; every learned
DCT-IV here, sweep and Adam alike, still sits below the classical *block-DCT
8×8* (#f1(cls.block_dct_8.psnr.at("0.2")) dB \@$rho{=}.20$), the strongest
transform in the table.
