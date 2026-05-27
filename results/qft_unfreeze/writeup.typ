#set page(paper: "us-letter", margin: (x: 0.85in, y: 0.85in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set block(above: 1.0em, below: 0.5em)
#show heading.where(level: 2): set block(above: 0.7em, below: 0.35em)
#show raw: set text(size: 8.5pt)

#let datasets = (
  ("quickdraw_5q", "QuickDraw (m=n=5, 30 gates)"),
  ("div2k_8q", "DIV2K (m=n=8, 72 gates)"),
  ("tuberlin_8q", "TU-Berlin (m=n=8, 72 gates)"),
)
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
  caption: [Windowed training loss $L / L_0$ versus cumulative optimiser step,
  one curve per unfreeze ordering. Rows: identity (top) vs Haar-random (bottom)
  init; columns: datasets. Each curve is a staircase — every downward
  step-and-plateau is one freshly-thawed gate trained to stationarity. The
  per-(dataset, init) two-panel staircases (loss *and* Riemannian grad norm)
  are in `<dataset>/<init>/figures/staircase.{pdf,svg}`.],
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

= Reading

The staircases make the per-gate marginal contribution legible: a tall step is a
gate that mattered, a flat plateau a gate that the optimiser left near where it
was thawed. Compare the three orderings within each panel for *path* effects and
the two rows for *initialisation* effects — if identity and random converge to
the same final PSNR (right-hand end of each curve and the table's $rho$ columns),
the unfreeze trajectory is a property of the operator/optimiser, not of the
starting point; if they differ, the QFT-family identity carries a usable
inductive bias even under a one-gate-at-a-time schedule. The trigger mix
(grad-norm vs loss-$Delta$ vs step-cap) reports *how* each stage terminated: a
high grad-norm count means stages genuinely reached stationarity within the cap.
