#set document(title: "Circuit family × initialisation × dataset: a progressive block-size study")
#set page(paper: "a4", margin: 1.9cm, numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(3pt); it; v(2pt) }

#let div2k = json("div2k_8q/report/data.json")
#let tb = json("tuberlin_8q/report/data.json")

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
#let fmt(x) = str(calc.round(x, digits: 1))
#let cell(tabledata, fam, init, k) = {
  let d = tabledata.at(fam + "/" + init, default: (:))
  if fam == "mera" and not (k in mera_k) { text(fill: rgb("#aaa"), size: 8pt)[n/a] }
  else if str(k) in d { [#fmt(d.at(str(k)))] }
  else { text(fill: rgb("#c33"))[\u{2014}] }
}
#let results_table(tabledata) = align(center)[
  #table(
    columns: (auto, auto) + (auto,) * 7,
    align: (left, center) + (right,) * 7,
    stroke: 0.4pt + luma(180),
    inset: (x: 5.5pt, y: 3.5pt),
    table.header([*family*], [*init*],
      ..range(2, 9).map(k => [*$k$=#str(k)*\ #text(size: 6.5pt)[(#str(calc.pow(2, k)))]])),
    ..combos.map(((fam, init)) => (
      text(fill: rgb(if init == "identity" { "#0a0a0a" } else { "#666" }))[#fam_short.at(fam)],
      text(size: 8pt)[#init],
      ..range(2, 9).map(k => cell(tabledata, fam, init, k)),
    )).flatten()
  )
]

#align(center)[
  #text(size: 15pt, weight: "bold")[Circuit family $times$ initialisation $times$ dataset]
  #v(2pt)
  #text(size: 11pt)[A progressive block-size study on DIV2K and TU-Berlin ($m=n=8$, $256 times 256$)]
  #v(3pt)
  #text(size: 9pt, fill: rgb("#555"))[Generated #datetime.today().display("[year]-[month]-[day]") · mean test PSNR over the test split]
]

#v(2pt)

*Summary.* We train five quantum-circuit transform families — `rich` (complex
$U(4)$ gates), `qft`, `tebd`, `entangled_qft`, `mera` — under a block-size
curriculum, from two initialisations (identity-operator and random), on two
datasets (natural images and sketches). Two themes emerge. *The dataset sets
the regime*: on DIV2K the block-size curve is flat and `rich` leads, while on
block-sparse TU-Berlin the curve inverts (small blocks win), the family ranking
flips with the rate, and classical block-DCT is near-lossless. *And a common
rate-crossover holds on both*: classical block-DCT dominates the learned
circuits at light compression but the learned circuits overtake it as
compression tightens. Throughout, `rich` (complex $U(4)$) leads the learned
families, and *identity init is never worse than random* — often dramatically
better.

= Setup

Each `*_progressive` sweep trains a circuit family under a *block-size
curriculum*: at stage $k$ the inner basis is a $2^k times 2^k$ circuit replicated
across the image by a `BlockedBasis` wrapper (the bare full circuit at $k=8$).
Stages are independent — no warm-start — and each trains for 1008 steps
(generalized preset, 112 epochs) under an `MSELoss` that keeps the top-$k$
coefficients by magnitude.

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + luma(180),
  inset: 5pt,
  [*Families*], [`rich` (complex $U(4)$ 2-qubit gates), `qft`, `tebd`,
    `entangled_qft`, `mera`],
  [*Initialisations*], [*identity* — every gate dropped to its manifold identity
    (H $arrow.r$ $I_2$, controlled-phase $arrow.r$ phase 0), so the circuit starts
    as the identity operator. *random* — `rich`: Haar $U(2)\/U(4)$; `qft`: Haar
    $U(2)$ on Hadamards + random controlled-phase angles;
    `tebd`\/`entangled_qft`\/`mera`: native seeded random gates.],
  [*Stage range*], [`rich`\/`qft`: $k=1..8$; `tebd`\/`entangled_qft`: $k=2..8$;
    `mera`: $k in {2,4,8}$ ($m$ must be a power of 2). Tables show $k=2..8$.],
  [*Datasets*], [*DIV2K-8q* — 500 natural images, centre-crop + LANCZOS to
    $256 times 256$. *TU-Berlin* — 500 line-drawing sketches, $approx 96.5%$
    white pixels, same preprocessing.],
  [*Metric*], [mean test PSNR \@ keep-ratio $rho$ (fraction of coefficients
    retained); $rho=0.20$ is the headline rate.],
)

= DIV2K-8q: natural images

#results_table(div2k)
#v(3pt)
#figure(image("div2k_8q/report/comparison_rho020.svg", width: 88%),
  caption: [DIV2K, $rho=0.20$ (5#sym.times). One colour per family,
    solid = identity / dashed = random init; grey dashed = classical block-DCT-8
    reference. `rich` leads the learned families but sits just below block-DCT.])
#figure(image("div2k_8q/report/comparison_rho005.svg", width: 88%),
  caption: [DIV2K, $rho=0.05$ (20#sym.times). The families bunch near block-DCT-8;
    `rich` draws level with it.])
#figure(image("div2k_8q/report/comparison_rho001.svg", width: 88%),
  caption: [DIV2K, $rho=0.01$ (100#sym.times). Every learned family clears
    block-DCT-8 by $approx 6$--$7$ dB — the learned transforms win at aggressive
    compression.])

*Findings.*
- *`rich` leads the learned families but classical block-DCT wins at the light
  rate.* At $rho=0.20$ `rich` peaks at $33.7$ dB, $approx 1.5$ dB above every
  other family — yet still just below the classical block-DCT-8 reference
  ($34.0$ dB). Complex $U(4)$ is the one architectural feature that helps, but it
  does not beat the fixed block transform here.
- *The learned circuits overtake block-DCT as compression tightens.* block-DCT-8
  falls faster with the rate ($34.0 arrow.r 26.1 arrow.r 14.9$ dB at
  $rho = 0.20\/0.05\/0.01$) than the learned circuits do: `rich` draws level at
  $rho=0.05$ ($26.4$ vs $26.1$) and *every* family clears block-DCT by
  $approx 6$--$7$ dB at $rho=0.01$ ($approx 21$ vs $14.9$).
- *Under identity init the four QFT-derived families are bit-identical* at every
  $k$ despite very different gate counts ($k=8$: qft 72, tebd 32, entangled_qft
  80, mera 44). From an identity start they all converge to the same QFT
  operator — the extra gates train to no-ops.
- *Identity init $gt.eq$ random* for every family; the curve is essentially flat
  in $k$ ($approx 33$ dB for `rich`, $approx 31.7$ for the rest) at $rho=0.20$.

#pagebreak()

= TU-Berlin: sketches (a different regime)

Sketches are *block-sparse* — most $8 times 8$ tiles are blank — so a classical
block transform is near-lossless: block-DCT-8 reaches
#fmt(tb.at("_refs").at("block_dct_8@0.2").at("mean")) dB mean /
#fmt(tb.at("_refs").at("block_dct_8@0.2").at("median")) median at $rho=0.20$
(versus $approx 32$ on DIV2K), but only
#fmt(tb.at("_refs").at("block_dct_8@0.05").at("mean")) dB at $rho=0.05$. We
therefore report two compression rates.

== Light compression: $rho = 0.20$ (5#sym.times)
#results_table(tb.at("rho020"))
#v(2pt)
#figure(image("tuberlin_8q/report/comparison_rho020.svg", width: 74%),
  caption: [TU-Berlin, $rho=0.20$. The curve *falls* with block size; the
    QFT-derived families peak above `rich` at $k=3$; all trail block-DCT-8 (grey).])

== Heavy compression: $rho = 0.05$ (20#sym.times)
#results_table(tb.at("rho005"))
#v(2pt)
#figure(image("tuberlin_8q/report/comparison_rho005.svg", width: 74%),
  caption: [TU-Berlin, $rho=0.05$. `rich` (identity) leads and edges *above*
    block-DCT-8 at $k=3,4$; the per-$k$ curve flattens.])

*Findings.*
- *The block-size curve inverts.* Small blocks win — each blank tile costs almost
  no coefficients — the opposite of DIV2K's flat curve.
- *The family ranking flips with the rate.* At $rho=0.20$ the QFT-derived families
  peak $approx 88$ dB at $k=3$, above `rich` ($approx 79$); at $rho=0.05$ `rich`
  (identity) leads ($approx 37$ dB) and edges above block-DCT-8 ($approx 36$).
- *Identity #sym.gt.tri random* — the gap reaches $approx 25$--$30$ dB at large $k$, far
  wider than on DIV2K. The QFT-family identity equivalence partly survives:
  `qft` $equiv$ `entangled_qft` at all $k$, and all four QFT-derived families
  coincide at $rho=0.05$.

= Cross-dataset discussion

#table(
  columns: (auto, 1fr, 1fr),
  stroke: 0.4pt + luma(180),
  inset: 5pt,
  [], [*DIV2K (natural)*], [*TU-Berlin (sketch)*],
  [block-size curve], [flat in $k$], [falls with $k$ (small blocks win)],
  [best learned family], [`rich` (everywhere)], [QFT-family \@ $rho$=0.2; `rich` \@ $rho$=0.05],
  [vs classical block-DCT], [block-DCT wins \@ $rho$=0.2; learned overtakes by $rho$=0.01],
    [block-DCT wins \@ $rho$=0.2; `rich` edges it \@ $rho$=0.05],
  [identity vs random], [identity $gt.eq$ random ($<1$ dB)], [identity #sym.gt.tri random (up to 25--30 dB)],
  [absolute PSNR \@ $rho$=0.2], [$approx 31$--$34$ dB], [$approx 60$--$88$ dB],
)

The contrast is governed by *energy compaction relative to the data's sparsity
structure*. Natural-image energy is spread across scales, so a global / large
circuit and the complex-$U(4)$ richness of `rich` help, and the curve is flat;
sketches are block-sparse, so a small per-block transform already captures almost
everything and a fixed block-DCT is near-lossless. On both datasets the fixed
block transform degrades faster than the learned circuits as the rate tightens,
so learning only pays off at aggressive compression.

= Conclusions

+ *The dataset sets the regime.* Conclusions about "which circuit family is best"
  do not transfer between natural images and sketches; even the sign of the
  block-size trend flips.
+ *Learned circuits beat classical block-DCT only at aggressive compression.* At
  the light rate block-DCT wins on both datasets; the learned transforms overtake
  it as $rho$ shrinks (by $rho=0.01$ on DIV2K, $rho=0.05$ on sketches for `rich`).
+ *Complex $U(4)$ (`rich`) is the only architectural feature that consistently
  helps* — it leads the learned families on both datasets.
+ *Identity initialisation dominates random* everywhere, decisively so on
  sketches; the structured identity start is a strong, cheap prior.

#v(4pt)
#text(size: 8pt, fill: gray)[
  Data: `results/family_init_matrix/{div2k_8q,tuberlin_8q}/<family>_<init>/`.
  Reproduce a sweep with `experiments/qft_progressive.py --dataset <d> --family <f>
  --init <i>`; regenerate figures with each report's `render_fig.py`.
]
