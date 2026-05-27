#import "@preview/cetz:0.4.2": canvas, draw
// Single QFT(5) circuit with every gate labelled by name: H_i (Hadamard on q_i)
// and M_(j,i) (controlled-phase between q_j and q_i, j>i). The three unfreeze
// orderings (block-growth / left->right / right->left) traverse these same gates
// in different sequence; they are described in words in the writeup, not drawn
// as separate circuits.
#set page(width: 17cm, height: auto, margin: 12pt, fill: white)
#set text(font: "New Computer Modern", size: 10pt)

#let GY = 1.0
#let STEP = 1.0
#let OX = 0.7
#let qy(i) = -(i - 1) * GY
#let gx(col) = OX + col * STEP
#let gfill = rgb("#dceefb")

#let hgate(col, q) = {
  import draw: *
  let x = gx(col); let y = qy(q); let w = 0.66; let h = 0.5
  on-layer(1, {
    rect((x - w/2, y - h/2), (x + w/2, y + h/2), fill: gfill, stroke: 0.7pt)
    content((x, y), text(8pt)[$H_#q$])
  })
}

// controlled-phase between upper qubit j and lower qubit t (t > j); name M_(t,j)
#let mgate(col, j, t) = {
  import draw: *
  let x = gx(col); let yu = qy(j); let yd = qy(t); let ym = (yu + yd) / 2
  let w = 0.74; let h = 0.46
  on-layer(1, {
    line((x, yu), (x, ym + h/2), stroke: 0.6pt)
    line((x, yd), (x, ym - h/2), stroke: 0.6pt)
    rect((x - w/2, ym - h/2), (x + w/2, ym + h/2), fill: gfill, stroke: 0.7pt)
    content((x, ym), text(6.5pt)[$M_(#t,#j)$])
    circle((x, yu), radius: 0.05, fill: black, stroke: none)
    circle((x, yd), radius: 0.05, fill: black, stroke: none)
  })
}

#align(center, canvas({
  import draw: *
  let x0 = OX + 0.3; let x1 = gx(15) + 0.6
  for i in range(1, 6) {
    let y = qy(i)
    line((x0, y), (x1, y), stroke: 0.6pt)
    content((x0 - 0.42, y), text(8pt)[$q_#i$])
  }
  hgate(1, 1);  mgate(2, 1, 2);  mgate(3, 1, 3);  mgate(4, 1, 4);  mgate(5, 1, 5)
  hgate(6, 2);  mgate(7, 2, 3);  mgate(8, 2, 4);  mgate(9, 2, 5)
  hgate(10, 3); mgate(11, 3, 4); mgate(12, 3, 5)
  hgate(13, 4); mgate(14, 4, 5)
  hgate(15, 5)
}))
