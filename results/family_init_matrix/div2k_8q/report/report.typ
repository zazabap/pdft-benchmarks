#set document(title: "Progressive block-size sweeps: 5 families x 2 inits (DIV2K-8q)")
#set page(paper: "a4", margin: 1.8cm, numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(2pt); it; v(2pt) }

#let data = json("data.json")

// Display order, grouped by family. Each entry: (family, init).
#let combos = (
  ("rich", "identity"), ("rich", "random"),
  ("qft", "identity"), ("qft", "random"),
  ("tebd", "identity"), ("tebd", "random"),
  ("entangled_qft", "identity"), ("entangled_qft", "random"),
  ("mera", "identity"), ("mera", "random"),
)
#let fam_short = (
  "rich": "rich", "qft": "qft", "tebd": "tebd",
  "entangled_qft": "ent-qft", "mera": "mera",
)
#let mera_k = (2, 4, 8)
#let cell(fam, init, k) = {
  let key = fam + "/" + init
  let d = data.at(key, default: (:))
  if fam == "mera" and not (k in mera_k) {
    text(fill: rgb("#aaa"), size: 8pt)[n/a]
  } else if str(k) in d {
    [#str(d.at(str(k)))]
  } else {
    text(fill: rgb("#c33"))[\u{2014}]
  }
}
#let done(key) = data.at(key, default: (:)).keys().len() > 0

#align(center)[
  #text(size: 14pt, weight: "bold")[Progressive block-size sweeps across circuit families]
  #v(3pt)
  #text(size: 10.5pt)[Five families $times$ two inits on DIV2K-8q ($m=n=8$, $256 times 256$)]
  #v(2pt)
  #text(size: 8.5pt, fill: rgb("#666"))[
    Generated #datetime.today().display("[year]-[month]-[day]").
    Test PSNR \@ $rho = 0.20$ (dB) per curriculum stage. Cells: value = final;
    #text(fill: rgb("#c33"))[\u{2014}] = sweep still running;
    #text(fill: rgb("#aaa"))[n/a] = stage undefined for that family.
  ]
]

= The experiment

Each `*_progressive` sweep trains a circuit family under a block-size
*curriculum*: at stage $k$ the inner basis is a $2^k times 2^k$ circuit
replicated across the $256 times 256$ image by a `BlockedBasis` wrapper (bare
full circuit at $k = 8$). Stages are independent (no warm-start); each trains
1008 steps (generalized preset, 112 epochs, `MSELoss` top-$k$ at $rho = 0.2$).

This run completes the 2-D matrix of *family* $times$ *initialisation*:

#table(
  columns: (auto, 1fr),
  stroke: 0.4pt + luma(180),
  inset: 5pt,
  [*Families*], [`rich` (complex $U(4)$), `qft`, `tebd`, `entangled_qft`, `mera`],
  [*Inits*], [
    *identity* — every gate dropped to its manifold identity (H $arrow.r$ $I_2$,
    controlled-phase $arrow.r$ phase 0), so the circuit starts as the identity
    operator. \
    *random* — `rich`: Haar $U(2)\/U(4)$; `qft`: Haar $U(2)$ on Hadamards +
    random controlled-phase angles; `tebd`\/`entangled_qft`\/`mera`: native
    seeded random gates. Per-stage seed $=$ seed $+ k$.
  ],
  [*Stage range*], [`rich`\/`qft`: $k = 1..8$; `tebd`\/`entangled_qft`: $k = 2..8$;
    `mera`: $k in {2,4,8}$ only ($m$ must be a power of 2). Table shows $k = 2..8$.],
)

= Results

#align(center)[
  #table(
    columns: (auto, auto) + (auto,) * 7,
    align: (left, center) + (right,) * 7,
    stroke: 0.4pt + luma(180),
    inset: (x: 6pt, y: 4.5pt),
    table.header(
      [*family*], [*init*],
      ..range(2, 9).map(k => [*$k$=#str(k)*\ #text(size: 7pt)[(#str(calc.pow(2, k)))]]),
    ),
    ..combos.map(((fam, init)) => (
      text(fill: rgb(if init == "identity" { "#0a0a0a" } else { "#555" }))[#fam_short.at(fam)],
      text(size: 8.5pt)[#init],
      ..range(2, 9).map(k => cell(fam, init, k)),
    )).flatten()
  )
]

#v(8pt)

#figure(
  image("comparison.svg", width: 95%),
  caption: [Test PSNR \@ $rho = 0.20$ vs curriculum stage. One colour per family,
    solid = identity init, dashed = random init (open markers). `mera` appears
    only at its valid $k in {2,4,8}$. *The four solid QFT-family curves (qft,
    tebd, entangled_qft, mera identity) coincide exactly* and plot on top of one
    another near $31.7$--$32$ dB; only `rich` (blue) clears the block-8 line.],
)

= Observations

- *Only `rich` clears the classical block-8 reference.* Its complex $U(4)$ gates
  reach $33.4$--$33.7$ dB, a full $1.5$--$2$ dB above every other family
  ($approx 31.7$--$32$). The extra expressivity of the complex two-qubit gates is
  the one thing in this study that genuinely buys reconstruction quality.

- *Under identity init, the four QFT-derived families are bit-identical at every
  stage* — `qft`, `tebd`, `entangled_qft`, `mera` all converge to
  $32.05 \/ 32.26 \/ 31.97 \/ 31.66 ... \/ 31.66$ despite very different gate
  counts (at $k=8$: qft 72, tebd 32, entangled_qft 80, mera 44 trainable gates).
  From an identity start under the top-$k$ MSE objective they all collapse onto
  the *same* QFT operator; the extra brick-wall / entangling / MERA gates train
  back to no-ops. Identity init is a strong attractor to the QFT solution.

- *Identity init beats random init for every family.* Randomising the start lands
  the optimiser in worse, family-specific basins: the QFT-derived families drop
  $approx 0.5$--$1$ dB at the larger blocks ($k >= 5$) and stop coinciding, while
  `rich` (random) shadows `rich` (identity) within $approx 0.5$ dB but never
  beats it. The complex-$U(4)$ freedom adds capacity; randomising the
  initialisation does not.

- *Small blocks are the great equaliser.* At $k = 2, 3$ every non-`rich` family
  sits at exactly $32.05 \/ 32.26$ regardless of init — the $4 times 4$ / $8 times 8$
  inner block has a single dominant top-$k$ optimum that everything finds.

#v(4pt)
#text(size: 8pt, fill: gray)[
  Working artefact (report-first; not yet cellified into `results/`). Identity
  `rich`\/`qft` from committed `results/{rich,qft}_progressive`; all other
  combos from `/tmp/<family>_<init>/`.
]
