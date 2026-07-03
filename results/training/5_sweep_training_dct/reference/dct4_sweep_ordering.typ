#import "@preview/cetz:0.4.2": canvas, draw
// Illustrative 2-D DCT-IV(3,3): two independent DCT-IV(3) blocks, one per image
// axis (axis 0 = q1..q3, axis 1 = q4..q6). The controlled O(2)-twiddle DCT-IV
// has four gate kinds — H_q (Hadamard/branch single-qubit), U4 (mirror-CNOT,
// 2q), CRY (single-angle controlled-R_y twiddle, 2q), CP (Delta-sign
// controlled-phase, 2q). Gates are shown in the circuit's (emission) order;
// the number on each gate is its position in the SWEEP visit order (the basis'
// tensor-slot order). fwd sweeps 1 -> N, rev sweeps N -> 1, and — unlike the
// one-gate-per-stage unfreeze study — every gate is revisited on every sweep.
// The DIV2K runs use the analogous DCT-IV(8,8): 214 gates, axis 1 = q9..q16.
#set page(width: 20cm, height: auto, margin: 12pt, fill: white)
#set text(font: "New Computer Modern", size: 10pt)

#let GY = 0.92
#let STEP = 1.02
#let OX = 1.5
#let GAP = 0.95           // vertical gap between the two axes
#let gx(col) = OX + (col - 1) * STEP
#let qy(q) = if q <= 3 { -(q - 1) * GY } else { -(q - 1) * GY - GAP }

#let cH  = rgb("#0072B2")   // Hadamard / branch single-qubit
#let cU4 = rgb("#D55E00")   // mirror CNOT
#let cCRY = rgb("#009E73")  // twiddle controlled-R_y
#let cCP = rgb("#CC79A7")   // Delta-sign controlled-phase
#let tint(c) = c.lighten(80%)

// single-qubit Hadamard box; n = sweep-order index
#let hgate(col, q, n) = {
  import draw: *
  let x = gx(col); let y = qy(q); let w = 0.6; let h = 0.5
  on-layer(1, {
    rect((x - w/2, y - h/2), (x + w/2, y + h/2), fill: tint(cH), stroke: 0.7pt + cH)
    content((x, y), text(8pt)[$H$])
    content((x + w/2 - 0.02, y + h/2 - 0.02), anchor: "north-east",
            text(5pt, fill: cH)[#n])
  })
}
// 2-qubit gate between qubits a (upper) and b (lower); lab = short kind label,
// col-color c, sweep index n. Dots on both wires + a mid box.
#let cgate(col, a, b, lab, c, n) = {
  import draw: *
  let x = gx(col); let yu = qy(a); let yd = qy(b); let ym = (yu + yd) / 2
  let w = 0.62; let h = 0.42
  on-layer(1, {
    line((x, yu), (x, yd), stroke: 0.6pt + c)
    circle((x, yu), radius: 0.05, fill: c, stroke: none)
    circle((x, yd), radius: 0.05, fill: c, stroke: none)
    rect((x - w/2, ym - h/2), (x + w/2, ym + h/2), fill: tint(c), stroke: 0.7pt + c)
    content((x, ym), text(5.5pt)[#lab])
    content((x + w/2 - 0.02, ym + h/2 - 0.02), anchor: "north-east",
            text(4.5pt, fill: c)[#n])
  })
}

#align(center, canvas({
  import draw: *
  let ncol = 17
  let x0 = OX - 0.55; let x1 = gx(ncol) + 0.6
  for q in range(1, 7) {
    let y = qy(q)
    line((x0, y), (x1, y), stroke: 0.55pt + luma(120))
    content((x0 - 0.4, y), text(8pt)[$q_#q$])
  }
  content((OX - 1.25, (qy(1) + qy(3)) / 2), anchor: "east",
          text(8.5pt, fill: luma(60))[*axis 0*\ (rows)])
  content((OX - 1.25, (qy(4) + qy(6)) / 2), anchor: "east",
          text(8.5pt, fill: luma(60))[*axis 1*\ (cols)])

  // ---- axis 0: DCT-IV(3) on q1..q3.  (col, gate, sweep#) ----
  cgate(1, 3, 2, "U4", cU4, 4);  cgate(2, 3, 1, "U4", cU4, 5)
  hgate(3, 3, 6)
  cgate(4, 1, 3, "CRY", cCRY, 7); cgate(5, 2, 3, "CRY", cCRY, 8)
  cgate(6, 3, 2, "U4", cU4, 9);  cgate(7, 3, 1, "U4", cU4, 10)
  cgate(8, 2, 1, "U4", cU4, 11)
  hgate(9, 2, 12)
  cgate(10, 1, 2, "CRY", cCRY, 13); cgate(11, 2, 1, "U4", cU4, 14)
  hgate(12, 1, 15); hgate(13, 1, 1)
  cgate(14, 2, 1, "CP", cCP, 16)
  hgate(15, 2, 2)
  cgate(16, 3, 2, "CP", cCP, 17)
  hgate(17, 3, 3)

  // ---- axis 1: identical DCT-IV(3) on q4..q6 (same columns) ----
  cgate(1, 6, 5, "U4", cU4, 4);  cgate(2, 6, 4, "U4", cU4, 5)
  hgate(3, 6, 6)
  cgate(4, 4, 6, "CRY", cCRY, 7); cgate(5, 5, 6, "CRY", cCRY, 8)
  cgate(6, 6, 5, "U4", cU4, 9);  cgate(7, 6, 4, "U4", cU4, 10)
  cgate(8, 5, 4, "U4", cU4, 11)
  hgate(9, 5, 12)
  cgate(10, 4, 5, "CRY", cCRY, 13); cgate(11, 5, 4, "U4", cU4, 14)
  hgate(12, 4, 15); hgate(13, 4, 1)
  cgate(14, 5, 4, "CP", cCP, 16)
  hgate(15, 5, 2)
  cgate(16, 6, 5, "CP", cCP, 17)
  hgate(17, 6, 3)

  // ---- sweep-order note under the circuit ----
  let yb = qy(6) - 1.0
  content(((gx(1) + gx(ncol)) / 2, yb), anchor: "north",
          text(7.5pt)[Superscripts give the *sweep visit order* (basis tensor-slot order):
          `fwd` visits gates $1 arrow.r N$, `rev` visits $N arrow.r 1$.])
  content(((gx(1) + gx(ncol)) / 2, yb - 0.4), anchor: "north",
          text(7.5pt, fill: luma(90))[#sym.arrow.ccw The full sweep repeats until the accepted
          fraction $arrow.r 0$ — *every* gate is revisited on *every* pass (Gauss–Seidel).])
}))

#v(2pt)
#align(center, text(8pt)[
  #box(baseline: 0.15em, rect(width: 0.5em, height: 0.5em, fill: cH.lighten(80%), stroke: 0.6pt + cH)) #h(0.15em) *H* Hadamard / branch (1q) #h(1.2em)
  #box(baseline: 0.15em, rect(width: 0.5em, height: 0.5em, fill: cU4.lighten(80%), stroke: 0.6pt + cU4)) #h(0.15em) *U4* mirror CNOT (2q) #h(1.2em)
  #box(baseline: 0.15em, rect(width: 0.5em, height: 0.5em, fill: cCRY.lighten(80%), stroke: 0.6pt + cCRY)) #h(0.15em) *CRY* twiddle $R_y$ (2q) #h(1.2em)
  #box(baseline: 0.15em, rect(width: 0.5em, height: 0.5em, fill: cCP.lighten(80%), stroke: 0.6pt + cCP)) #h(0.15em) *CP* $Delta$-sign phase (2q)
])
