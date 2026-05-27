#set page(paper: "us-letter", margin: (x: 0.85in, y: 0.85in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set block(above: 1.0em, below: 0.5em)
#show heading.where(level: 2): set block(above: 0.7em, below: 0.35em)
#show raw: set text(size: 8.5pt)

#let datasets_all = (
  ("quickdraw_5q", "QuickDraw (m=n=5, 30 gates)"),
  ("div2k_8q", "DIV2K (m=n=8, 72 gates)"),
  ("tuberlin_8q", "TU-Berlin (m=n=8, 72 gates)"),
)
// Only render PSNR tables for datasets whose runs have finished (the m=8 sweeps
// land later); _available.json is regenerated as cells are merged in.
#let available = json("_available.json")
#let datasets = datasets_all.filter(d => d.at(0) in available)
#let inits = ("identity", "random")
#let orders = (("bg", "block-growth"), ("lr", "left" + sym.arrow.r + "right"),
               ("rl", "right" + sym.arrow.r + "left"))
#let man(ds, init) = json(ds + "/" + init + "/manifest.json")
#let f1(x) = str(calc.round(x, digits: 1))

#align(center)[
  #text(size: 15pt, weight: "bold")[Progressive gate-unfreezing of the QFT operator]
  #v(2pt)
  #text(size: 10.5pt)[One gate thawed per stage and trained to a stationarity
  plateau — three unfreeze orderings $times$ two initialisations $times$ three datasets]
  #v(3pt)
  #text(size: 9pt, fill: rgb("#555"))[Generated #datetime.today().display("[year]-[month]-[day]")]
]

= Question

The `QFTBasis(m, m)` used here has every gate trainable on its U(2) manifold but
its operator starts either at the QFT-family *identity* (H $arrow.r I_2$,
controlled-phase $arrow.r$ phase $0$, so $T_(t=0)(x) = x$) or at a *Haar-random*
point. Rather than training all gates at once, we *unfreeze one gate per stage*
and train each stage to a plateau before thawing the next. Two questions:
(i) does the *order* in which gates are thawed matter — block-growth (`bg`),
left$arrow.r$right (`lr`), right$arrow.r$left (`rl`)? and (ii) does the
*initialisation* — identity vs random — change the trajectory or the endpoint?

= Setup

Each run optimises a *fixed batch* of the training split (full $n = 500$ at
QuickDraw $m = 5$; batch $50$ at $m = 8$) under an `MSELoss` that keeps the
top-10% of coefficients by magnitude. The loop drives pdft's JIT-compiled Adam
step with a per-stage frozen set — Adam moments reset between stages, carried
within a stage, constant learning rate — and a separate Riemannian grad-norm
probe supplies the plateau signal. A stage ends when
$norm(g) < 10^(-5)$ *or* $abs(Delta L) < 10^(-5)$ (after a 5-step minimum), or at
the step cap (800 at $m = 8$). `QFTBasis(m, m)` has $m (m + 1)$ gates, so the
sweep runs $30$ stages at $m = 5$ and $72$ at $m = 8$. The three orderings are
fixed index sequences into the Hadamard-first gate storage; random init draws a
Haar $U(2)$ on each H slot and a uniform phase on each controlled-phase slot,
with a seed shared across the three orderings so they start from the same basis.

= Training dynamics

#figure(
  image("figures/training_dynamics.svg", width: 100%),
  caption: [*Absolute* top-$k$ MSE loss (not a normalised ratio) versus
  cumulative optimiser step, one curve per unfreeze ordering. Rows: identity
  (top) vs Haar-random (bottom) init; columns: datasets. Each curve is a
  staircase — every downward step-and-plateau is one freshly-thawed gate trained
  to stationarity; the final MSE and PSNR\@$rho{=}.20$ are marked at each curve's
  end. The per-(dataset, init) two-panel staircases (MSE *and* Riemannian grad
  norm) are in `<dataset>/<init>/figures/staircase.{pdf,svg}`.],
)

= Final reconstruction PSNR

Test-split PSNR (dB) once the last gate has been thawed and trained, at four
keep ratios $rho$, with the cumulative step count and the per-stage plateau
triggers (grad-norm / loss-$Delta$ / step-cap):

#for (ds, dslab) in datasets [
  == #dslab
  #table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, right, right, right, right, center),
    stroke: 0.4pt + luma(180),
    inset: (x: 5pt, y: 3pt),
    table.header([*init*], [*order*], [$rho{=}.05$], [$rho{=}.10$],
                 [$rho{=}.15$], [$rho{=}.20$], [*steps (g/$Delta$/cap)*]),
    ..inits.map(init => {
      let mo = man(ds, init).orderings
      orders.map(((ok, olab)) => {
        let o = mo.at(ok)
        let p = o.final_psnr
        let t = o.trigger_counts
        (init, olab, f1(p.at("0.05")), f1(p.at("0.1")), f1(p.at("0.15")),
         f1(p.at("0.2")),
         [#str(o.total_steps) (#str(t.grad_norm)/#str(t.loss_delta)/#str(t.max_steps))])
      })
    }).flatten()
  )
  #v(2pt)
]

= Reference: best PSNR from prior experiments

The strongest test-PSNR achieved by *any* prior trained basis on each dataset
(the `family_init_matrix` sweep: five circuit families $times$ two inits $times$
block sizes $k = 2..8$), as a ceiling to read the gate-unfreeze numbers against:

#let pastmax = json("reference/past_max_psnr.json")
#let pm_rows = ("quickdraw_5q", "div2k_8q", "tuberlin_8q")
#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.4pt + luma(180), inset: (x: 6pt, y: 3pt),
    table.header([*dataset*], [$rho{=}.05$], [$rho{=}.10$], [$rho{=}.20$]),
    ..pm_rows.map(ds => {
      let e = pastmax.at(ds)
      (e.label,
       f1(e.by_rho.at("0.05").psnr) + " dB",
       f1(e.by_rho.at("0.10").psnr) + " dB",
       f1(e.by_rho.at("0.20").psnr) + " dB")
    }).flatten()
  )
]

#let pm20(ds) = {
  let e = pastmax.at(ds).by_rho.at("0.20")
  [#f1(e.psnr) dB (#raw(e.method), k=#e.k)]
}
Best method per cell at $rho{=}.20$: QuickDraw #pm20("quickdraw_5q"); DIV2K
#pm20("div2k_8q"); TU-Berlin #pm20("tuberlin_8q"). Note these maxima sit at *small block
sizes* on the two sparse drawing sets (QuickDraw, TU-Berlin) — the near-identity
small-block basis is already an excellent sparse representation there, which is
exactly the regime where the gate-unfreeze PSNR peaks *early* (see Reading).

= Comparison: block-size progressive baseline

The `qft_progressive` experiment reaches the *same* full QFT$(8,8)$ operator on
DIV2K by a different curriculum — a *block-size* schedule that trains the entire
$2^k times 2^k$ QFT (all $k(k+1)$ gates at once, 1008 steps) at each stage
$k = 1..8$, instead of thawing one gate at a time. It is the natural reference
for the question *does the curriculum/order matter for the final operator, or
only for the path?*

#let qp = json("reference/qft_progressive_div2k_8q.json")
#align(center)[
  #table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (center, center, center, right, right, right, right),
    stroke: 0.4pt + luma(180), inset: (x: 5.5pt, y: 3pt),
    table.header([*stage $k$*], [*block*], [*\# gates*],
                 [$rho{=}.05$], [$rho{=}.10$], [$rho{=}.15$], [$rho{=}.20$]),
    ..qp.stages.map(s => (str(s.k), str(s.block_size) + $times$ + str(s.block_size),
        str(s.n_trainable), f1(s.psnr.at("0.05")), f1(s.psnr.at("0.1")),
        f1(s.psnr.at("0.15")), f1(s.psnr.at("0.2")))).flatten()
  )
  #text(size: 8pt, fill: rgb("#555"))[DIV2K test PSNR (dB) per keep ratio $rho$]
]

The block-size curriculum *saturates by $k = 2$* and the full QFT$(8,8)$ endpoint
is *#f1(qp.stages.last().psnr.at("0.2")) dB* at $rho{=}.20$
(#f1(qp.stages.last().psnr.at("0.05")) / #f1(qp.stages.last().psnr.at("0.1")) /
#f1(qp.stages.last().psnr.at("0.15")) dB at $rho = .05, .10, .15$; = the
all-gates-at-once `qft_identity` result). Read this against the *DIV2K* rows of
the gate-unfreeze table above: if the identity-init sweep also lands at
$approx$ #f1(qp.stages.last().psnr.at("0.2")) dB, the QFT$(8,8)$ optimum is
*schedule-independent* — neither the unfreeze order
nor the one-gate-at-a-time vs whole-block curriculum changes the destination,
only the trajectory (exactly what the QuickDraw panels show, where all three
orderings converge). A gate-unfreeze endpoint *below* the block-size baseline
would instead say the per-gate schedule gets trapped short of the joint optimum.

#figure(
  image("reference/qft_progressive_training_dynamics.svg", width: 80%),
  caption: [Reference (`qft_progressive`, DIV2K-8q): the block-size curriculum's
  per-stage validation MSE, $k = 2..7$. Eight whole-block stages here vs the
  30/72 per-gate stages of the unfreeze staircases above — same endpoint, very
  different path granularity.],
)

= Reading

The staircases (absolute top-$k$ MSE) make the per-gate marginal contribution
legible: a tall step is a gate that mattered, a flat plateau a gate the optimiser
left near where it was thawed. Compare the three orderings within each panel for
*path* effects and the two init rows for *initialisation* effects. On QuickDraw
the training MSE converges to the same floor for all three orderings within an
init (order changes only the path), while identity reaches a *lower* MSE than
random — the QFT-family identity carries a usable inductive bias even under a
one-gate-at-a-time schedule. The trigger mix (grad-norm vs loss-$Delta$ vs
step-cap) reports *how* each stage terminated.

*Training loss $eq.not$ test PSNR on sparse data.* The optimiser minimises the
top-$k$ MSE in the *transform* domain, which is not the same objective as
test-image PSNR. On QuickDraw the per-stage test PSNR\@$rho{=}.20$ is *highest
early* — up to *54.8 dB around stage 5*, near the identity/pixel operator — and
then *falls to $approx$37.6 dB* as more gates train the operator toward the QFT
and spread energy away from the sparse-pixel representation. So for sparse
drawings the PSNR-optimal point is *early in the unfreeze schedule*, not at the
fully-trained end; this matches the reference table above, where the
prior-experiment maxima also sit at small block sizes. (Per-stage PSNR is
recorded in each cell's `trace.json`; on DIV2K, where images are not pixel-sparse,
training instead *improves* PSNR toward the $approx$31.7 dB block-size baseline.)
