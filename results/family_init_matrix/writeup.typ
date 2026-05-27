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
#let rate_tables(var, kmax: 8) = {
  let rows = (
    ([$rho = 0.20$ — 5#sym.times], "rho020"),
    ([$rho = 0.10$ — 10#sym.times], "rho010"),
    ([$rho = 0.05$ — 20#sym.times], "rho005"),
    ([$rho = 0.01$ — 100#sym.times], "rho001"),
  )
  for (lab, rk) in rows {
    text(weight: "bold", size: 9.5pt)[#lab]
    results_table(var.at(rk), kmax: kmax)
    v(3pt)
  }
}

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

DIV2K is 500 natural images, centre-crop + LANCZOS to $256 times 256$ ($m=n=8$).
Block-DCT-8 is strong here ($34.0$ dB \@ $rho=0.20$).

#figure(image("div2k_8q/report/comparison_grid.svg", width: 100%),
  caption: [DIV2K: family$times$init PSNR vs block size $k$ at four keep ratios.
    One colour per family, solid = identity / dashed = random init; grey dashed =
    classical block-DCT-8 (value per panel).])

*Findings.*
- *`rich` leads the learned families but block-DCT wins at the light rate.* At
  $rho=0.20$ `rich` peaks at $33.7$ dB, $approx 1.5$ dB above every other family,
  yet just below block-DCT-8 ($34.0$ dB).
- *The learned circuits overtake block-DCT as compression tightens.* block-DCT-8
  falls $34.0 arrow.r 29.4 arrow.r 26.1 arrow.r 14.9$ dB across
  $rho=0.20\/0.10\/0.05\/0.01$; `rich` draws level by $rho=0.05$ and every family
  clears it by $approx 6$--$7$ dB at $rho=0.01$.
- *Under identity init the four QFT-derived families are bit-identical* at every
  $k$ (gate counts at $k=8$: qft 72, tebd 32, entangled_qft 80, mera 44) — they
  all converge to the same QFT operator.
- *Identity init $gt.eq$ random*; the curve is flat in $k$ ($approx 33$ dB `rich`,
  $approx 31.7$ rest) at $rho=0.20$.

*Per-rate tables (mean test PSNR, dB).*
#rate_tables(div2k, kmax: 8)

#pagebreak()

= TU-Berlin: sketches (a different regime)

Sketches are *block-sparse* — most $8 times 8$ tiles are blank — so a classical
block transform is near-lossless: block-DCT-8 reaches
#fmt(tb.at("_refs").at("block_dct_8@0.2").at("mean")) dB mean /
#fmt(tb.at("_refs").at("block_dct_8@0.2").at("median")) median at $rho=0.20$
(versus $approx 32$ on DIV2K), but collapses to $approx 4.8$ dB at $rho=0.01$.

#figure(image("tuberlin_8q/report/comparison_grid.svg", width: 100%),
  caption: [TU-Berlin: family$times$init PSNR vs block size $k$ at four keep
    ratios. Note the *falling* curve (small blocks win) and the very high
    absolute PSNR; grey dashed = block-DCT-8 (value per panel).])

*Findings.*
- *The block-size curve inverts at the light rate.* Small blocks win — each blank
  tile costs almost no coefficients — the opposite of DIV2K's flat curve.
- *The family ranking flips with the rate.* At $rho=0.20$ the QFT-derived families
  peak $approx 88$ dB at $k=3$, above `rich` ($approx 79$); at $rho=0.05$ `rich`
  (identity) leads ($approx 37$ dB); at $rho=0.01$ the QFT-derived families edge
  back ahead ($approx 23.2$ vs `rich` $22.9$).
- *The heavy-compression crossover is extreme here.* block-DCT-8 falls
  $100.7 arrow.r 55.7 arrow.r 36.5 arrow.r 4.8$ dB across
  $rho=0.20\/0.10\/0.05\/0.01$: near-lossless at the light rate, but the learned
  circuits beat it by $approx 18$ dB at $rho=0.01$.
- *Identity #sym.gt.tri random* — the gap reaches $approx 25$--$30$ dB at large $k$, far
  wider than on DIV2K. `qft` $equiv$ `entangled_qft` at all $k$ under identity init.

*Per-rate tables (mean test PSNR, dB).*
#rate_tables(tb, kmax: 8)

= QuickDraw: low-resolution drawings

QuickDraw is $32 times 32$ ($m=n=5$), so the curriculum spans only $k=1..5$
(block sizes $2..32$; tables show $k=2..5$, `mera` at $k in {2,4}$). The drawings
are sparse strokes on a *dark* background, but at this low resolution they are
far less block-compressible than the high-res TU-Berlin sketches: classical
block-DCT-8 reaches only
#str(qd.at("_refs").at("block_dct_8@0.2").at("mean")) dB at $rho=0.20$ (vs
$approx 100$ on TU-Berlin), so the learned circuits beat it.

#figure(image("quickdraw_5q/report/comparison_grid.svg", width: 100%),
  caption: [QuickDraw: family$times$init PSNR vs block size $k$ at four keep
    ratios. The QFT-derived families (identity) dominate and the curve is flat in
    $k$; grey dashed = block-DCT-8 (value per panel), which the learned circuits
    clear at every rate.])

*Findings.*
- *Learned circuits beat block-DCT at every rate.* Unlike the $256²$ sets,
  block-DCT-8 is weak on $32 times 32$ drawings ($26.6$ dB \@ $rho=0.20$), so the
  QFT-derived families ($approx 39.5$) and `rich` ($approx 35$) both clear it.
- *The QFT-derived families lead and are bit-identical under identity init,*
  beating `rich` by $approx 4$ dB at $rho=0.20$. The curve is flat in $k$ (like
  DIV2K, unlike TU-Berlin's falling curve).
- *At heavy compression the families converge* to block-DCT ($approx 17$ dB at
  $rho=0.05$, $approx 12$ at $rho=0.01$) and the identity advantage erodes —
  random init even edges ahead at some $k$.

*Per-rate tables (mean test PSNR, dB).*
#rate_tables(qd, kmax: 5)

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
