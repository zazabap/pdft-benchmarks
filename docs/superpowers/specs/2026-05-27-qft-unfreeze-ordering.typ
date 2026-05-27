#import "@preview/cetz:0.4.2": canvas, draw

// Three QFT gate-unfreezing orderings on one circuit, stacked as a single figure:
//   (a) block-growth   (b) left→right sweep   (c) right→left sweep

#set page(width: 15.5cm, height: auto, margin: 14pt, fill: white)
#set text(font: "New Computer Modern", size: 10pt)

// Wong palette, one hue per block-stage k (k = highest qubit a gate touches).
#let stagecol = (
  rgb("#0072B2"), rgb("#E69F00"), rgb("#009E73"), rgb("#CC79A7"), rgb("#D55E00"),
)
#let fillc(k) = stagecol.at(k - 1).lighten(62%)

// geometry
#let GY = 0.9
#let STEP = 0.86
#let OX = 0.7
#let qy(i) = -(i - 1) * GY        // q1 at y=0 (top) ... q5 at y=-4 (bottom)
#let gx(col) = OX + col * STEP

// unfreeze badge number per ordering, as a function of construction column
#let unum(mode, col, bg) = {
  if mode == "lr" { col } else if mode == "rl" { 16 - col } else { bg }
}

// Hadamard: box ON the wire. stage k, block-growth order bg.
#let hgate(mode, col, q, k, bg) = {
  import draw: *
  let x = gx(col); let y = qy(q); let w = 0.48; let h = 0.42
  let o = unum(mode, col, bg)
  on-layer(1, {
    rect((x - w/2, y - h/2), (x + w/2, y + h/2), fill: fillc(k), stroke: 0.7pt)
    content((x, y), text(8pt)[$H$])
    content((x - w/2 - 0.02, y + h/2 + 0.02), anchor: "south-east", text(7pt, weight: "bold")[#o])
  })
}

// Controlled-phase M, paper tensor-network style: box at MIDPOINT between the two
// wires, legs to a dot on each. j = upper qubit, t = lower partner. stage = t.
#let mgate(mode, col, j, t, bg) = {
  import draw: *
  let x = gx(col); let yu = qy(j); let yd = qy(t); let ym = (yu + yd) / 2
  let w = 0.42; let h = 0.36
  let o = unum(mode, col, bg)
  on-layer(1, {
    line((x, yu), (x, ym + h/2), stroke: 0.6pt)
    line((x, yd), (x, ym - h/2), stroke: 0.6pt)
    rect((x - w/2, ym - h/2), (x + w/2, ym + h/2), fill: fillc(t), stroke: 0.7pt)
    content((x, ym), text(7pt)[$M$])
    circle((x, yu), radius: 0.05, fill: black, stroke: none)
    circle((x, yd), radius: 0.05, fill: black, stroke: none)
    content((x - w/2 - 0.02, ym + h/2 + 0.02), anchor: "south-east", text(7pt, weight: "bold")[#o])
  })
}

#let circuit(mode) = canvas({
  import draw: *
  let x0 = OX + 0.30; let x1 = gx(15) + 0.5
  for i in range(1, 6) {
    let y = qy(i)
    line((x0, y), (x1, y), stroke: 0.6pt)
    content((x0 - 0.38, y), text(8pt)[$q_#i$])
  }
  hgate(mode, 1, 1, 1, 1);   mgate(mode, 2, 1, 2, 3);   mgate(mode, 3, 1, 3, 5)
  mgate(mode, 4, 1, 4, 8);   mgate(mode, 5, 1, 5, 12);  hgate(mode, 6, 2, 2, 2)
  mgate(mode, 7, 2, 3, 6);   mgate(mode, 8, 2, 4, 9);   mgate(mode, 9, 2, 5, 13)
  hgate(mode, 10, 3, 3, 4);  mgate(mode, 11, 3, 4, 10); mgate(mode, 12, 3, 5, 14)
  hgate(mode, 13, 4, 4, 7);  mgate(mode, 14, 4, 5, 15); hgate(mode, 15, 5, 5, 11)
  // sweep-direction arrow for the lr / rl variants
  let ay = qy(5) - 0.65
  if mode == "lr" {
    line((gx(1), ay), (gx(15), ay), mark: (end: "straight"), stroke: 1pt + gray.darken(10%))
  } else if mode == "rl" {
    line((gx(15), ay), (gx(1), ay), mark: (end: "straight"), stroke: 1pt + gray.darken(10%))
  }
})

#let panel(tag, desc, mode) = [
  #text(10pt, weight: "bold")[#tag] #h(5pt) #text(8.5pt, fill: gray.darken(25%))[#desc]
  #v(1pt)
  #align(center, circuit(mode))
]

// ---- document ----
#align(center)[
  #text(13pt, weight: "bold")[QFT gate-unfreezing — three orderings (single axis, m = 5)]
]
#v(3pt)
#align(center, block(width: 96%, text(8.5pt)[
  Same QFT(5) circuit in the paper's tensor-network notation (controlled-phase $M$ = box between
  two wires with legs to a dot on each). Bold badge = #text(weight: "bold")[unfreeze step] for that
  ordering; fill = block-stage #text(weight: "bold")[k] (= highest qubit a gate touches). Cumulative:
  each step thaws one more gate, the rest stay frozen at identity. (No SWAP gates — bit-reversal is
  folded into the index convention.)
]))
#v(9pt)

#panel("(a) block-growth", [stage k completes QFT(k, k); long-range couplings delayed — the gate-level mirror of the block sweep], "bg")
#v(9pt)
#panel("(b) left → right sweep", [unfreeze in QFT construction order 1→15; q1 couples to all of q2–q5 before $H_2$ at step 6], "lr")
#v(9pt)
#panel("(c) right → left sweep", [reverse construction order; last-built gate ($H_5$) first, $H_1$ last], "rl")

#v(11pt)
#align(center)[
  #text(8.5pt, weight: "bold")[Block-stage colours:] #h(6pt)
  #for k in range(1, 6) [
    #box(width: 10pt, height: 10pt, fill: fillc(k), stroke: .5pt) #h(3pt) #text(8.5pt)[k = #k] #h(12pt)
  ]
]
