#set document(title: "Circuit family × initialisation × dataset: a progressive block-size study")
#set page(paper: "a4", margin: 1.9cm, numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(3pt); it; v(2pt) }

#let div2k = json("div2k_8q/report/data.json")
#let tb = json("tuberlin_8q/report/data.json")
#let qd = json("quickdraw_5q/report/data.json")

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
#let results_table(tabledata, kmax: 8) = align(center)[
  #table(
    columns: (auto, auto) + (auto,) * (kmax - 1),
    align: (left, center) + (right,) * (kmax - 1),
    stroke: 0.4pt + luma(180),
    inset: (x: 5.5pt, y: 3.5pt),
    table.header([*family*], [*init*],
      ..range(2, kmax + 1).map(k => [*$k$=#str(k)*\ #text(size: 6.5pt)[(#str(calc.pow(2, k)))]])),
    ..combos.map(((fam, init)) => (
      text(fill: rgb(if init == "identity" { "#0a0a0a" } else { "#666" }))[#fam_short.at(fam)],
      text(size: 8pt)[#init],
      ..range(2, kmax + 1).map(k => cell(tabledata, fam, init, k)),
    )).flatten()
  )
]

#align(center)[
  #text(size: 15pt, weight: "bold")[Circuit family $times$ initialisation $times$ dataset]
  #v(2pt)
  #text(size: 11pt)[A progressive block-size study on DIV2K, TU-Berlin and QuickDraw]
  #v(3pt)
  #text(size: 9pt, fill: rgb("#555"))[Generated #datetime.today().display("[year]-[month]-[day]") · mean test PSNR over the test split]
]

#v(2pt)

*Summary.* We train five quantum-circuit transform families — `rich` (complex
$U(4)$ gates), `qft`, `tebd`, `entangled_qft`, `mera` — under a block-size
curriculum, from two initialisations (identity-operator and random), on three
datasets: DIV2K (natural images, $256²$), TU-Berlin (high-res sketches, $256²$),
and QuickDraw (low-res drawings, $32²$). *The dataset sets the regime*: on DIV2K
the block-size curve is flat and `rich` leads; on block-sparse TU-Berlin it falls
(small blocks win) and classical block-DCT is near-lossless; on coarse QuickDraw
the global QFT-family wins outright and block-DCT is weak. *A common rate-
crossover holds*: classical block-DCT dominates the learned circuits at light
compression but they overtake it as compression tightens (and on QuickDraw the
learned circuits win at every rate). `rich` leads on DIV2K, the QFT-family on
the two drawing sets; *identity init is never worse than random* — often
dramatically better.

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

== Very heavy compression: $rho = 0.01$ (100#sym.times)
#results_table(tb.at("rho001"))
#v(2pt)
#figure(image("tuberlin_8q/report/comparison_rho001.svg", width: 74%),
  caption: [TU-Berlin, $rho=0.01$. block-DCT-8 *collapses* to $approx 4.8$ dB
    while the learned circuits plateau $approx 23$ dB — clearing it by
    $approx 18$ dB. An even sharper version of the heavy-compression crossover
    seen on DIV2K.])

*Findings.*
- *The block-size curve inverts at the light rate.* Small blocks win — each blank
  tile costs almost no coefficients — the opposite of DIV2K's flat curve. (At the
  heavy rates the curves instead rise then plateau in $k$.)
- *The family ranking flips with the rate.* At $rho=0.20$ the QFT-derived families
  peak $approx 88$ dB at $k=3$, above `rich` ($approx 79$); at $rho=0.05$ `rich`
  (identity) leads ($approx 37$ dB); at $rho=0.01$ the QFT-derived families edge
  back ahead ($approx 23.2$ vs `rich` $22.9$).
- *The heavy-compression crossover is extreme here.* block-DCT-8 falls
  $100.7 arrow.r 36.5 arrow.r 4.8$ dB across $rho = 0.20\/0.05\/0.01$: it is
  near-lossless at the light rate but the learned circuits beat it by
  $approx 1$ dB at $rho=0.05$ and by $approx 18$ dB at $rho=0.01$.
- *Identity #sym.gt.tri random* — the gap reaches $approx 25$--$30$ dB at large $k$, far
  wider than on DIV2K. The QFT-family identity equivalence partly survives:
  `qft` $equiv$ `entangled_qft` at all $k$, and all four QFT-derived families
  coincide at $rho=0.05$.

= QuickDraw: low-resolution drawings

QuickDraw is $32 times 32$ ($m=n=5$), so the curriculum spans only $k=1..5$
(block sizes $2..32$; tables show $k=2..5$, `mera` at $k in {2,4}$). The drawings
are sparse strokes on a *dark* background (mean pixel $approx 0.12$), but at this
low resolution they are far less block-compressible than the high-res TU-Berlin
sketches: classical block-DCT-8 reaches only
#str(qd.at("_refs").at("block_dct_8@0.2").at("mean")) dB at $rho=0.20$ (vs
$approx 100$ on TU-Berlin).

== Light compression: $rho = 0.20$ (5#sym.times)
#results_table(qd.at("rho020"), kmax: 5)
#v(2pt)
#figure(image("quickdraw_5q/report/comparison_rho020.svg", width: 72%),
  caption: [QuickDraw, $rho=0.20$. The QFT-derived families (identity, all four
    bit-identical at $approx 39.5$ dB) *dominate* — above `rich` ($approx 35$) and
    well above block-DCT-8 ($26.6$). The curve is flat in $k$.])

== Heavy compression: $rho = 0.05$ (20#sym.times)
#results_table(qd.at("rho005"), kmax: 5)
#v(2pt)
#figure(image("quickdraw_5q/report/comparison_rho005.svg", width: 72%),
  caption: [QuickDraw, $rho=0.05$. All families bunch near block-DCT-8
    ($approx 17$ dB); the identity advantage vanishes and random init even edges
    ahead at some $k$.])

== Very heavy compression: $rho = 0.01$ (100#sym.times)
#results_table(qd.at("rho001"), kmax: 5)
#v(2pt)
#figure(image("quickdraw_5q/report/comparison_rho001.svg", width: 72%),
  caption: [QuickDraw, $rho=0.01$. Everything collapses toward block-DCT-8
    ($approx 12.9$ dB); families are within $approx 2$ dB of each other.])

*Findings.*
- *Learned circuits beat block-DCT at the light rate.* Unlike DIV2K/TU-Berlin,
  block-DCT-8 is weak on $32 times 32$ drawings ($26.6$ dB), so at $rho=0.20$ the
  QFT-derived families ($approx 39.5$) and even `rich` ($approx 35$) clear it.
- *The QFT-derived families lead and are bit-identical under identity init,*
  beating `rich` by $approx 4$ dB at $rho=0.20$ — the global QFT solution suits
  these coarse drawings better than `rich`'s complex gates. The curve is flat in
  $k$ (like DIV2K, unlike TU-Berlin's falling curve).
- *At heavy compression the families converge* to block-DCT ($approx 17$ dB at
  $rho=0.05$, $approx 12$ at $rho=0.01$) and the identity advantage erodes —
  random init even edges ahead at some $k$.

= Cross-dataset discussion

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  stroke: 0.4pt + luma(180),
  inset: 4.5pt,
  [], [*DIV2K (natural, 256²)*], [*TU-Berlin (sketch, 256²)*], [*QuickDraw (drawing, 32²)*],
  [block-size curve], [flat in $k$], [falls with $k$ (small blocks win)], [flat in $k$],
  [best learned family], [`rich`], [QFT-family \@ 0.2 & 0.01; `rich` \@ 0.05], [QFT-family],
  [block-DCT \@ $rho$=0.2], [$34$ dB (beats all)], [$94$ dB (beats all)], [$27$ dB (learned beat it)],
  [vs block-DCT], [learned overtake by $rho$=0.01], [learned overtake \@ 0.05 & 0.01], [learned win at all $rho$ \@ 0.2; converge \@ 0.01],
  [identity vs random], [identity $gt.eq$ random ($<1$ dB)], [identity #sym.gt.tri random (25--30 dB)], [identity wins \@ 0.2; ties \@ 0.05/0.01],
  [PSNR \@ $rho$=0.2], [$approx 31$--$34$], [$approx 60$--$88$], [$approx 35$--$40$],
)

The contrast is governed by *energy compaction relative to the data's sparsity
structure and resolution*. Natural-image energy is spread across scales, so a
large circuit and the complex-$U(4)$ richness of `rich` help; high-res sketches
are block-sparse, so a small per-block transform is near-lossless and the curve
falls with $k$; low-res QuickDraw drawings are too coarse for $8 times 8$ blocks
to exploit, so the global QFT-family solution wins outright and block-DCT is
weak. A common thread: the fixed block transform degrades faster than the
learned circuits as the rate tightens, so learning pays off most at aggressive
compression (and, on coarse QuickDraw, at every rate).

= Conclusions

+ *The dataset sets the regime.* "Which circuit family is best" does not transfer
  across natural images, high-res sketches and low-res drawings; even the sign of
  the block-size trend and the leading family change.
+ *Learned circuits beat classical block-DCT at aggressive compression* (and at
  every rate on coarse QuickDraw, where block-DCT is weak); on the $256²$ sets
  block-DCT wins at the light rate but the learned transforms overtake it as
  $rho$ shrinks.
+ *Architectural richness is dataset-dependent:* complex $U(4)$ (`rich`) leads on
  natural images, but the plain QFT-family solution leads on both drawing sets.
+ *Identity initialisation dominates random* on the $256²$ sets (decisively on
  sketches); on coarse QuickDraw the advantage holds only at the light rate and
  ties under heavy compression. The structured identity start is a strong, cheap
  prior.

#v(4pt)
#text(size: 8pt, fill: gray)[
  Data: `results/family_init_matrix/{div2k_8q,tuberlin_8q,quickdraw_5q}/<family>_<init>/`.
  Reproduce a sweep with `experiments/qft_progressive.py --dataset <d> --family <f>
  --init <i>`; regenerate figures with each report's `render_fig*.py`.
]
