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

#figure(image("reference/qft_unfreeze_ordering.svg", width: 100%),
  caption: [The QFT$(5)$ circuit (one axis) with every gate named: $H_i$ =
  Hadamard on $q_i$; $M_(j,i)$ = controlled-phase between $q_j$ and $q_i$ ($j > i$).
  The three orderings thaw these same gates in the sequences below; the $m = 8$
  runs use the analogous 72-gate circuit.]) <fig-circuit>

The three unfreeze orderings traverse the gates of @fig-circuit as:
- *block-growth* (`bg`): finish QFT$(k)$ before adding qubit $k{+}1$ —
  $H_1$; $H_2, M_(2,1)$; $H_3, M_(3,1), M_(3,2)$; $H_4, M_(4,1), M_(4,2), M_(4,3)$;
  $H_5, M_(5,1), ..., M_(5,4)$. Long-range couplings come last.
- *left$arrow.r$right* (`lr`): QFT construction order —
  $H_1, M_(2,1), M_(3,1), M_(4,1), M_(5,1), H_2, M_(3,2), ..., H_5$.
- *right$arrow.r$left* (`rl`): the reverse — $H_5$ first, $H_1$ last.

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
  `H7`, `CP3,1`) was thawed at each of the largest drops. The y-axis is the
  *training* top-10%-coefficient MSE loss; endpoints are annotated with the
  separate *test PSNR\@$rho{=}.20$* (image reconstruction on held-out data) — the
  two are different metrics, so the curve's height does not determine the PSNR.
  Per-cell loss+grad-norm staircases: `div2k_8q/<init>/figures/`.])

#figure(image("reference/qft_progressive_training_dynamics.svg", width: 74%),
  caption: [*Block-size* curriculum (`qft_progressive`, DIV2K) for comparison:
  per-stage validation MSE for block sizes $k = 2..7$, each one whole
  $2^k times 2^k$ QFT trained 1008 steps. Eight whole-block stages here vs the 72
  one-gate-at-a-time stages above — same $approx$31.7 dB endpoint (table below).])

= Reconstruction PSNR (DIV2K)

Gate-unfreeze endpoints *together with* the `qft_progressive` block-size baseline
(train all $k(k+1)$ gates of a $2^k times 2^k$ QFT at once, per stage $k$). DIV2K
test PSNR (dB) at four keep ratios $rho$:

// gate-unfreeze ("current") and block-size ("previous") rows as (label, (4 raw PSNRs)).
#let gu = {
  let acc = ()
  for init in inits {
    for (ok, _olab) in orders {
      let p = man("div2k_8q", init).orderings.at(ok).final_psnr
      acc.push(("unfreeze · " + init + " · " + ok,
                (p.at("0.05"), p.at("0.1"), p.at("0.15"), p.at("0.2"))))
    }
  }
  acc
}
#let bs = qp.stages.map(s => (
  "block-size · k=" + str(s.k) + " (" + str(s.block_size) + "×" + str(s.block_size) + ")",
  (s.psnr.at("0.05"), s.psnr.at("0.1"), s.psnr.at("0.15"), s.psnr.at("0.2"))))
#let cmax(rows, j) = calc.max(..rows.map(r => r.at(1).at(j)))
#let gmax = range(4).map(j => cmax(gu, j))     // per-ρ best gate-unfreeze
#let bmax = range(4).map(j => cmax(bs, j))     // per-ρ best block-size
#let mk(v, m, c) = if calc.abs(v - m) < 1e-4 { text(fill: c, weight: "bold")[#f1(v)] } else [#f1(v)]
#let cellrows(rows, maxes, c) = {
  let acc = ()
  for (lab, vs) in rows {
    acc.push(lab)
    for j in range(4) { acc.push(mk(vs.at(j), maxes.at(j), c)) }
  }
  acc
}
#let clj = json("reference/classical_div2k.json")
#let cl = ("block_dct_8", "block_fft_8").map(k => ("classical · " + clj.at(k).label,
  (clj.at(k).psnr.at("0.05"), clj.at(k).psnr.at("0.1"),
   clj.at(k).psnr.at("0.15"), clj.at(k).psnr.at("0.2"))))
#let clmax = range(4).map(j => cmax(cl, j))    // per-ρ best classical transform
#align(center)[#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  stroke: 0.4pt + luma(180), inset: (x: 6pt, y: 3pt),
  table.header([*configuration*], [$rho{=}.05$], [$rho{=}.10$], [$rho{=}.15$], [$rho{=}.20$]),
  ..cellrows(gu, gmax, rgb("#0072B2")),       // current = gate-unfreeze, max in blue
  table.hline(stroke: 0.8pt),
  table.cell(colspan: 5, inset: (y: 1.3pt), stroke: none)[],   // gap -> double rule
  table.hline(stroke: 0.8pt),
  ..cellrows(bs, bmax, rgb("#CC0000")),        // previous = block-size, max in red
  table.hline(stroke: 0.5pt),
  ..cellrows(cl, clmax, rgb("#009E73")),       // classical reference, max in green
)]

All learned QFT schemes reach the *same* QFT$(8,8)$ endpoint, $approx$#f1(qp.stages.last().psnr.at("0.2")) dB
\@$rho{=}.20$: the block-size curriculum saturates by $k = 2$ and the gate-unfreeze
sweep lands there from either init and any order — *schedule-independent* (`rl`
trails $approx$1 dB; every gate-unfreeze stage ended on the loss-$Delta$ plateau,
none at the step cap). For reference, the best *any* prior trained basis reaches
is #f1(pastmax.div2k_8q.by_rho.at("0.20").psnr) dB (#raw(pastmax.div2k_8q.by_rho.at("0.20").method)).
The classical *block-DCT 8×8* is in fact the strongest here
(#f1(cl.at(0).at(1).at(3)) dB \@$rho{=}.20$), ahead of every learned QFT basis;
*block-FFT 8×8* is weaker (#f1(cl.at(1).at(1).at(3)) dB) — the QFT-family operators
sit between the two classical block transforms.

= Reading

The staircase shows each gate's marginal value: the big drops are the *early*
gates (the marked Hadamards, e.g. `H9`, `H2` for block-growth); later gates only
fine-tune. One caveat on the endpoint labels — the train-loss order need not match
the test-PSNR order (`bg` has the *highest* train MSE yet the *best* test PSNR):
the loss is a top-10%-coefficient proxy on the train batch, PSNR a 20%-keep
reconstruction on the test set.
