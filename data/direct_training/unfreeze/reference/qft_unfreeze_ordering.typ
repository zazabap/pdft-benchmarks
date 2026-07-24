#import "@preview/cetz:0.4.2": canvas, draw
// 2-D QFT(5,5): two independent QFT(5) blocks, one per image axis
// (axis 0 = q1..q5, axis 1 = q6..q10). Every gate named: H_i (Hadamard on q_i),
// M_(j,i) (controlled-phase between q_j and q_i, j>i). The DIV2K runs use the
// analogous QFT(8,8) (axis 1 = q9..q16). Block-growth grows a square 2^k block,
// thawing qubit k on BOTH axes per stage; lr/rl traverse the gates left/right.
#set page(width: 17.5cm, height: auto, margin: 12pt, fill: white)
#set text(font: "New Computer Modern", size: 10pt)

#let GY = 0.92
#let STEP = 1.0
#let OX = 1.35
#let GAP = 0.85          // vertical gap between the two axes
#let gx(col) = OX + col * STEP
#let qy(q) = if q <= 5 { -(q - 1) * GY } else { -(q - 1) * GY - GAP }
#let gfill = rgb("#dceefb")

#let hgate(col, q) = {
  import draw: *
  let x = gx(col); let y = qy(q); let w = 0.64; let h = 0.5
  on-layer(1, {
    rect((x - w/2, y - h/2), (x + w/2, y + h/2), fill: gfill, stroke: 0.7pt)
    content((x, y), text(8pt)[$H_#q$])
  })
}
// controlled-phase between upper qubit j and lower qubit t (same axis, t>j); name M_(t,j)
#let mgate(col, j, t) = {
  import draw: *
  let x = gx(col); let yu = qy(j); let yd = qy(t); let ym = (yu + yd) / 2
  let w = 0.72; let h = 0.44
  on-layer(1, {
    line((x, yu), (x, ym + h/2), stroke: 0.6pt)
    line((x, yd), (x, ym - h/2), stroke: 0.6pt)
    rect((x - w/2, ym - h/2), (x + w/2, ym + h/2), fill: gfill, stroke: 0.7pt)
    content((x, ym), text(5.5pt)[$M_(#t,#j)$])
    circle((x, yu), radius: 0.05, fill: black, stroke: none)
    circle((x, yd), radius: 0.05, fill: black, stroke: none)
  })
}

#align(center, canvas({
  import draw: *
  let x0 = OX + 0.3; let x1 = gx(15) + 0.55
  for q in range(1, 11) {
    let y = qy(q)
    line((x0, y), (x1, y), stroke: 0.6pt)
    content((x0 - 0.42, y), text(8pt)[$q_#q$])
  }
  // axis brackets / labels
  content((OX - 1.05, (qy(1) + qy(5)) / 2), anchor: "east",
          text(8.5pt, fill: rgb("#0072B2"))[*axis 0*\ (rows)])
  content((OX - 1.05, (qy(6) + qy(10)) / 2), anchor: "east",
          text(8.5pt, fill: rgb("#009E73"))[*axis 1*\ (cols)])
  // axis 0: QFT(5) on q1..q5
  hgate(1, 1); mgate(2, 1, 2); mgate(3, 1, 3); mgate(4, 1, 4); mgate(5, 1, 5)
  hgate(6, 2); mgate(7, 2, 3); mgate(8, 2, 4); mgate(9, 2, 5)
  hgate(10, 3); mgate(11, 3, 4); mgate(12, 3, 5)
  hgate(13, 4); mgate(14, 4, 5)
  hgate(15, 5)
  // axis 1: QFT(5) on q6..q10 (same columns)
  hgate(1, 6); mgate(2, 6, 7); mgate(3, 6, 8); mgate(4, 6, 9); mgate(5, 6, 10)
  hgate(6, 7); mgate(7, 7, 8); mgate(8, 7, 9); mgate(9, 7, 10)
  hgate(10, 8); mgate(11, 8, 9); mgate(12, 8, 10)
  hgate(13, 9); mgate(14, 9, 10)
  hgate(15, 10)
}))
