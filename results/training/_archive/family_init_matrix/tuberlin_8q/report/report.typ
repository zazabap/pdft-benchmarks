#set document(title: "TU-Berlin progressive block-size sweeps")
#set page(paper: "a4", margin: 1.8cm, numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(2pt); it; v(2pt) }

#let data = json("data.json")
#let refs = data.at("_refs", default: (:))
#let combos = (
  ("rich", "identity"), ("rich", "random"),
  ("qft", "identity"), ("qft", "random"),
  ("tebd", "identity"), ("tebd", "random"),
  ("entangled_qft", "identity"), ("entangled_qft", "random"),
  ("mera", "identity"), ("mera", "random"),
)
#let fam_short = ("rich": "rich", "qft": "qft", "tebd": "tebd",
                  "entangled_qft": "ent-qft", "mera": "mera")
#let mera_k = (2, 4, 8)
#let kmax = 8
#let cell(rho_key, fam, init, k) = {
  let d = data.at(rho_key, default: (:)).at(fam + "/" + init, default: (:))
  if fam == "mera" and not (k in mera_k) { text(fill: rgb("#aaa"), size: 8pt)[n/a] }
  else if str(k) in d { [#str(calc.round(d.at(str(k)), digits: 1))] }
  else { text(fill: rgb("#c33"))[\u{2014}] }
}
#let results_table(rho_key) = align(center)[
  #table(
    columns: (auto, auto) + (auto,) * (kmax - 1),
    align: (left, center) + (right,) * (kmax - 1),
    stroke: 0.4pt + luma(180),
    inset: (x: 5.5pt, y: 3.5pt),
    table.header([*family*], [*init*],
      ..range(2, kmax + 1).map(k => [*$k$=#str(k)*\\ #text(size: 6.5pt)[(#str(calc.pow(2, k)))]])),
    ..combos.map(((fam, init)) => (
      text(fill: rgb(if init == "identity" { "#0a0a0a" } else { "#666" }))[#fam_short.at(fam)],
      text(size: 8pt)[#init],
      ..range(2, kmax + 1).map(k => cell(rho_key, fam, init, k)),
    )).flatten()
  )
]
#let rate_tables() = {
  for (lab, rk) in (([$rho = 0.20$ — 5#sym.times], "rho020"), ([$rho = 0.10$ — 10#sym.times], "rho010"),
                    ([$rho = 0.05$ — 20#sym.times], "rho005"), ([$rho = 0.01$ — 100#sym.times], "rho001")) {
    text(weight: "bold", size: 9.5pt)[#lab]; results_table(rk); v(3pt)
  }
}

#align(center)[
  #text(size: 14pt, weight: "bold")[Progressive block-size sweeps on TU-Berlin]
  #v(3pt)
  #text(size: 10.5pt)[Five families $times$ two inits $times$ four rates ($m=n=8$, $256 times 256$)]
  #v(2pt)
  #text(size: 8.5pt, fill: rgb("#666"))[Mean test PSNR (dB) over 50 sketches. #text(fill: rgb("#aaa"))[n/a] = stage undefined (mera needs $k$ a power of 2).]
]

= High-resolution sketches (block-sparse)

TU-Berlin sketches are $approx 96.5%$ white — block-sparse — so block-DCT-8 is near-lossless at the light rate ($approx 100.7$ dB mean \@ $rho=0.20$) but collapses to $approx 4.8$ dB by $rho=0.01$.

#figure(image("comparison_grid.svg", width: 100%),
  caption: [PSNR vs block size $k$ at four keep ratios. One colour per family,
    solid = identity / dashed = random init; grey dashed = block-DCT-8 (value per
    panel).])

= Observations

- *The block-size curve inverts at the light rate* (small blocks win) — opposite of DIV2K.
- *Family ranking flips with rate*: QFT-family peaks $approx 88$ dB \@ $rho=0.20$ ($k=3$); `rich` leads \@ $rho=0.05$; QFT-family again \@ $rho=0.01$.
- *Heavy-compression crossover is extreme*: block-DCT $100.7 arrow.r 4.8$ dB across the rates; learned circuits beat it by $approx 18$ dB at $rho=0.01$. Identity #sym.gt.tri random (up to 25--30 dB).

= Per-rate tables (mean PSNR, dB)
#rate_tables()
