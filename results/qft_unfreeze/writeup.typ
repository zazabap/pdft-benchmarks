#set page(paper: "us-letter", margin: (x: 0.9in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set block(above: 0.9em, below: 0.4em)
#show raw: set text(size: 8.5pt)

#let inits = ("identity", "random")
#let orders = (("bg", "block-growth"), ("lr", "left" + sym.arrow.r + "right"),
               ("rl", "right" + sym.arrow.r + "left"))
#let man(ds, init) = json(ds + "/" + init + "/manifest.json")
#let qp = json("reference/qft_progressive_div2k_8q.json")
#let pastmax = json("reference/past_max_psnr.json")
#let f1(x) = str(calc.round(x, digits: 1))

#align(center)[
  #text(size: 15pt, weight: "bold")[Progressive gate-unfreezing of the QFT operator]
  #v(2pt)
  #text(size: 10.5pt)[One gate thawed per stage, trained to a plateau — three
  unfreeze orderings $times$ two initialisations on DIV2K-8q]
  #v(3pt)
  #text(size: 9pt, fill: rgb("#555"))[Generated #datetime.today().display("[year]-[month]-[day]")]
]

= Question

The `QFTBasis(8, 8)` (72 U(2) gates) starts either at the QFT-family *identity*
($T_(t=0)(x) = x$) or *Haar-random*. We *unfreeze one gate per stage*, train it to
a plateau, then thaw the next. Does the *order* — block-growth (`bg`),
left$arrow.r$right (`lr`), right$arrow.r$left (`rl`) — or the *initialisation*
change the trajectory or the endpoint?

#figure(image("reference/qft_unfreeze_ordering.svg", width: 90%),
  caption: [The three orderings on a QFT circuit (badge = unfreeze step, fill =
  block-stage $k$). *bg*: stage $k$ completes QFT$(k, k)$. *lr*: QFT construction
  order. *rl*: reverse. The same index sequences (at $m = 8$) drive every run.])

= Setup

Fixed batch of 50 training images; `MSELoss` on the top-10% coefficients. pdft's
JIT Adam step runs with a per-stage frozen set (moments reset per stage); a
Riemannian grad-norm probe gives the plateau signal. A stage ends at
$norm(g) < 10^(-5)$ *or* $abs(Delta L) < 10^(-5)$ (min 5 steps), else an 800-step
cap. Random init: Haar $U(2)$ per Hadamard, uniform phase per controlled-phase,
seed shared across orderings.

= Training dynamics

#figure(image("figures/training_dynamics.svg", width: 100%),
  caption: [Absolute *training* top-$k$ MSE loss vs cumulative step, one curve per
  ordering (rows: identity / random init). Tick labels mark *which gate* (e.g.
  `H7`, `CP3,1`) was thawed at each of the largest drops. Endpoints report
  *train MSE* and *test PSNR\@$rho{=}.20$* — note these are *different* metrics
  (training top-10%-coefficient loss vs test-set image reconstruction keeping
  20%), so a curve can sit higher in train loss yet score a higher test PSNR.
  Per-cell loss+grad-norm staircases: `div2k_8q/<init>/figures/`.])

= Reconstruction PSNR (DIV2K)

Test PSNR (dB) at the final stage, with cumulative steps and plateau-trigger
counts (grad-norm / loss-$Delta$ / cap):

#for ds in ("div2k_8q",) [
  #table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right, center),
    stroke: 0.4pt + luma(180), inset: (x: 5pt, y: 3pt),
    table.header([*init*], [*order*], [$rho{=}.05$], [$rho{=}.10$], [$rho{=}.15$],
                 [$rho{=}.20$], [*steps (g/$Delta$/cap)*]),
    ..inits.map(init => {
      let mo = man(ds, init).orderings
      orders.map(((ok, olab)) => {
        let o = mo.at(ok); let p = o.final_psnr; let t = o.trigger_counts
        (init, olab, f1(p.at("0.05")), f1(p.at("0.1")), f1(p.at("0.15")),
         f1(p.at("0.2")),
         [#str(o.total_steps) (#str(t.grad_norm)/#str(t.loss_delta)/#str(t.max_steps))])
      })
    }).flatten()
  )
]

= Block-size baseline (comparison)

`qft_progressive` reaches the *same* QFT$(8,8)$ on DIV2K via a *block-size*
curriculum (train all $k(k+1)$ gates of a $2^k times 2^k$ QFT at once, $k = 1..8$):

#align(center)[#table(
  columns: 7, align: (center, center, center, right, right, right, right),
  stroke: 0.4pt + luma(180), inset: (x: 5.5pt, y: 3pt),
  table.header([*$k$*], [*block*], [*\# gates*], [$rho{=}.05$], [$rho{=}.10$],
               [$rho{=}.15$], [$rho{=}.20$]),
  ..qp.stages.map(s => (str(s.k), str(s.block_size) + $times$ + str(s.block_size),
    str(s.n_trainable), f1(s.psnr.at("0.05")), f1(s.psnr.at("0.1")),
    f1(s.psnr.at("0.15")), f1(s.psnr.at("0.2")))).flatten()
)]

It saturates by $k = 2$; the full QFT$(8,8)$ endpoint is
*#f1(qp.stages.last().psnr.at("0.2")) dB* \@$rho{=}.20$. The gate-unfreeze sweep
lands at the same value (identity `bg`/`lr` #f1(man("div2k_8q", "identity").orderings.bg.final_psnr.at("0.2"))/#f1(man("div2k_8q", "identity").orderings.lr.final_psnr.at("0.2")) dB,
random $approx$31.7), so the QFT$(8,8)$ optimum is *schedule-independent*. For
reference, the best *any* prior trained basis reaches on DIV2K is
#f1(pastmax.div2k_8q.by_rho.at("0.20").psnr) dB (`#pastmax.div2k_8q.by_rho.at("0.20").method`);
the QFT family sits $approx$2 dB under the richer U(4) families.

= Reading

The staircase shows each gate's marginal value: a tall drop is a gate that
mattered (mostly the early, small-block gates — see the marked steps), a flat
plateau a gate left near where it was thawed. All three orderings and both inits
converge to $approx$31.7 dB, so *order and curriculum set only the path, not the
destination* — `rl` merely trails by $approx$1 dB en route. The train-loss order
need not match the test-PSNR order (`bg` has the *highest* train MSE yet the
*best* test PSNR): the loss is a top-10%-coefficient proxy on the train batch,
PSNR a 20%-keep reconstruction on the test set, so the two rank slightly
differently in this near-flat optimum.
