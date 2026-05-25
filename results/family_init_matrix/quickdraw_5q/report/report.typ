#set document(title: "QuickDraw progressive block-size sweeps: 5 families x 2 inits x 3 rates")
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
#let mera_k = (2, 4)
#let cell(rho_key, fam, init, k) = {
  let d = data.at(rho_key).at(fam + "/" + init, default: (:))
  if fam == "mera" and not (k in mera_k) { text(fill: rgb("#aaa"), size: 8pt)[n/a] }
  else if str(k) in d { [#str(calc.round(d.at(str(k)), digits: 1))] }
  else { text(fill: rgb("#c33"))[\u{2014}] }
}
#let results_table(rho_key) = align(center)[
  #table(
    columns: (auto, auto) + (auto,) * 4,
    align: (left, center) + (right,) * 4,
    stroke: 0.4pt + luma(180),
    inset: (x: 6pt, y: 4pt),
    table.header([*family*], [*init*],
      ..range(2, 6).map(k => [*$k$=#str(k)*\ #text(size: 7pt)[(#str(calc.pow(2, k)))]])),
    ..combos.map(((fam, init)) => (
      text(fill: rgb(if init == "identity" { "#0a0a0a" } else { "#666" }))[#fam_short.at(fam)],
      text(size: 8.5pt)[#init],
      ..range(2, 6).map(k => cell(rho_key, fam, init, k)),
    )).flatten()
  )
]

#align(center)[
  #text(size: 14pt, weight: "bold")[Progressive block-size sweeps on QuickDraw]
  #v(3pt)
  #text(size: 10.5pt)[Five families $times$ two inits $times$ three rates ($m=n=5$, $32 times 32$)]
  #v(2pt)
  #text(size: 8.5pt, fill: rgb("#666"))[
    Mean test PSNR (dB) over 50 drawings. #text(fill: rgb("#aaa"))[n/a] = stage
    undefined (mera needs $k in {2,4}$ here).
  ]
]

= A coarse-drawing regime

QuickDraw is $32 times 32$ ($m=n=5$), so the curriculum spans $k=1..5$ (block
sizes $2..32$; shown $k=2..5$). The drawings are sparse strokes on a *dark*
background. At this low resolution $8 times 8$-block transforms have little to
exploit: classical block-DCT-8 reaches only
#str(refs.at("block_dct_8@0.2", default: (mean: 0)).at("mean")) dB at $rho=0.20$,
so the global learned circuits beat it. Setup otherwise matches the DIV2K /
TU-Berlin matrix (1008 steps/stage, `MSELoss` top-$k$).

= Light compression: $rho = 0.20$ (5#sym.times)
#results_table("rho020")
#v(3pt)
#figure(image("comparison_rho020.svg", width: 80%),
  caption: [PSNR \@ $rho=0.20$. The QFT-derived families (identity, all four
    bit-identical $approx 39.5$ dB) dominate — above `rich` ($approx 35$) and well
    above block-DCT-8 (grey). Flat in $k$.])

= Heavy compression: $rho = 0.05$ (20#sym.times)
#results_table("rho005")
#v(3pt)
#figure(image("comparison_rho005.svg", width: 80%),
  caption: [PSNR \@ $rho=0.05$. All families bunch near block-DCT-8
    ($approx 17$ dB); the identity advantage vanishes, random init edges ahead
    at some $k$.])

= Very heavy compression: $rho = 0.01$ (100#sym.times)
#results_table("rho001")
#v(3pt)
#figure(image("comparison_rho001.svg", width: 80%),
  caption: [PSNR \@ $rho=0.01$. Everything collapses toward block-DCT-8
    ($approx 12.9$ dB); families lie within $approx 2$ dB.])

= Observations

- *Learned circuits beat block-DCT at every rate (unlike the $256²$ sets),* because
  $8 times 8$-block DCT is weak on $32 times 32$ drawings — at $rho=0.20$ the
  QFT-derived families ($approx 39.5$ dB) clear block-DCT ($26.6$) by $approx 13$ dB.
- *The QFT-family leads and is bit-identical under identity init,* beating `rich`
  by $approx 4$ dB at $rho=0.20$; the curve is flat in $k$ (like DIV2K).
- *Identity #sym.gt.tri random at the light rate only;* at $rho=0.05\/0.01$ the
  families converge to block-DCT and the init advantage erodes (random sometimes
  edges ahead).

#v(4pt)
#text(size: 8pt, fill: gray)[
  Working artefact. Combos from `results/family_init_matrix/quickdraw_5q/<family>_<init>/`;
  rho=0.01 re-evaluated from checkpoints. Mean PSNR over the seed=42 / 50-drawing test split.
]
