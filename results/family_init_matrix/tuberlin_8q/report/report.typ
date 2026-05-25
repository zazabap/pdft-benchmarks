#set document(title: "TU-Berlin progressive block-size sweeps: 5 families x 2 inits x 2 rates")
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
#let cell(rho_key, fam, init, k) = {
  let d = data.at(rho_key).at(fam + "/" + init, default: (:))
  if fam == "mera" and not (k in mera_k) { text(fill: rgb("#aaa"), size: 8pt)[n/a] }
  else if str(k) in d { [#str(calc.round(d.at(str(k)), digits: 1))] }
  else { text(fill: rgb("#c33"))[\u{2014}] }
}
#let results_table(rho_key) = align(center)[
  #table(
    columns: (auto, auto) + (auto,) * 7,
    align: (left, center) + (right,) * 7,
    stroke: 0.4pt + luma(180),
    inset: (x: 5.5pt, y: 4pt),
    table.header([*family*], [*init*],
      ..range(2, 9).map(k => [*$k$=#str(k)*\ #text(size: 6.5pt)[(#str(calc.pow(2, k)))]])),
    ..combos.map(((fam, init)) => (
      text(fill: rgb(if init == "identity" { "#0a0a0a" } else { "#666" }))[#fam_short.at(fam)],
      text(size: 8pt)[#init],
      ..range(2, 9).map(k => cell(rho_key, fam, init, k)),
    )).flatten()
  )
]

#align(center)[
  #text(size: 14pt, weight: "bold")[Progressive block-size sweeps on TU-Berlin sketches]
  #v(3pt)
  #text(size: 10.5pt)[Five families $times$ two inits $times$ two compression rates ($m=n=8$, $256 times 256$)]
  #v(2pt)
  #text(size: 8.5pt, fill: rgb("#666"))[
    Generated #datetime.today().display("[year]-[month]-[day]"). Mean test PSNR (dB)
    over 50 sketches. #text(fill: rgb("#c33"))[\u{2014}] = still training;
    #text(fill: rgb("#aaa"))[n/a] = stage undefined (mera needs $k in {2,4,8}$).
  ]
]

= Sketches are a different regime

The TU-Berlin set is line drawings: $approx 96.5%$ white pixels with thin black
ink. They are *block-sparse* — most $8 times 8$ tiles are blank — so a classical
block transform is near-lossless: block-DCT-8 reaches
#str(refs.at("block_dct_8@0.2", default: (mean: 0)).at("mean")) dB at $rho=0.2$
(median #str(refs.at("block_dct_8@0.2", default: (median: 0)).at("median"))),
versus $approx 32$ dB on DIV2K. This inverts the DIV2K story, so we report the
learned circuits at *two* compression rates: the headline $rho=0.20$ (5#sym.times)
and an aggressive $rho=0.05$ (20#sym.times).

The experiment is otherwise identical to the DIV2K matrix: each stage $k$ trains a
$2^k times 2^k$ inner circuit replicated by `BlockedBasis` (bare at $k=8$), 1008
steps, `MSELoss` top-$k$. Inits: *identity* (gates dropped to identity operator)
and *random* (rich Haar $U(2)\/U(4)$; qft Haar-on-H + random phases;
tebd\/entangled_qft\/mera native seed). Families run $k=2..8$ except `mera`
($k in {2,4,8}$).

= Light compression: $rho = 0.20$ (5#sym.times)

#results_table("rho020")
#v(4pt)
#figure(image("comparison_rho020.svg", width: 88%),
  caption: [PSNR \@ $rho=0.20$. Block-sparse sketches make *small* blocks win:
    the curve falls with $k$. All learned circuits sit below classical block-DCT-8
    (grey).])

#pagebreak()

= Heavy compression: $rho = 0.05$ (20#sym.times)

#results_table("rho005")
#v(4pt)
#figure(image("comparison_rho005.svg", width: 88%),
  caption: [PSNR \@ $rho=0.05$. At $20 times$ compression the learned circuits
    become *competitive with* block-DCT-8 (grey) — rich (identity) edges above it
    at $k=3,4$. The per-$k$ curve also flattens.])

#pagebreak()

= Very heavy compression: $rho = 0.01$ (100#sym.times)

#results_table("rho001")
#v(4pt)
#figure(image("comparison_rho001.svg", width: 88%),
  caption: [PSNR \@ $rho=0.01$. block-DCT-8 *collapses* to $approx 4.8$ dB; the
    learned circuits plateau $approx 23$ dB, clearing it by $approx 18$ dB. The
    QFT-derived families edge `rich` here. $k=2$ degenerates (too few coefficients
    survive $100 times$ pooling on a $4 times 4$ block).])

= Observations

- *Block sparsity dominates the light-compression regime, and the curve
  inverts.* At $rho=0.20$ PSNR *falls* with block size $k$ — small blocks win
  because each blank tile costs almost no coefficients — the opposite of DIV2K's
  flat $approx 33$ dB. The best learned circuits peak at $approx 88$ dB ($k=3$,
  the QFT-derived families) but still trail classical block-DCT-8
  (median $approx 94$ dB): on near-lossless-compressible sketches the fixed block
  transform is hard to beat.

- *The family ranking flips between rates.* At $rho=0.20$ the QFT-derived
  families (qft\/tebd\/entangled_qft) *beat* `rich` at the small blocks
  ($approx 88$ vs $79$ dB at $k=3$). At $rho=0.05$ this reverses: `rich`
  (identity) leads at $approx 37$ dB and even edges *above* block-DCT-8
  ($approx 36$ dB) at $k=3,4$. When only 5% of coefficients survive, learning a
  genuine energy-compacting transform pays off and the classical near-lossless
  advantage evaporates.

- *Identity init beats random for every family, by a lot.* On sketches the gap is
  far larger than on DIV2K — up to $approx 25$--$30$ dB at large $k$ ($rho=0.2$,
  e.g. rich $k=7$: $61.5$ vs $33.9$). A random start is genuinely damaging here.

- *The QFT-family identity-init equivalence partly survives.* Under identity init
  `qft` and `entangled_qft` are bit-identical at every $k$ (as on DIV2K); at
  $rho=0.05$ all four QFT-derived families coincide, but at $rho=0.20$ they split
  into a qft\/entangled_qft pair and a tebd\/mera pair. `rich` (complex $U(4)$)
  stands apart from all of them.

#v(4pt)
#text(size: 8pt, fill: gray)[
  Working artefact (report-first; not yet cellified into `results/`). All combos
  from `/tmp/tuberlin/<family>_<init>/`; classical refs computed on the same
  seed=42 / 50-sketch test split.
]
