#set page(paper: "us-letter", margin: (x: 0.7in, y: 0.7in))
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em, first-line-indent: 0pt)
#show heading.where(level: 1): set block(above: 1.0em, below: 0.5em)
#show heading.where(level: 2): set block(above: 0.8em, below: 0.4em)
#show raw: set text(size: 8.5pt)

= Training the QFT operator: three findings

This is the synthesis for the `results/training/` studies. They share
one subject — the parametric QFT operator as a learned image transform
— and one motivating gap. On DIV2K-8q the headline random-init `qft`
basis trains to $31.29$ dB at $rho = 0.20$, while `blocked_8`
(a `BlockedBasis(QFTBasis(3,3), 5, 5)`, a strictly *less* expressive
submanifold of QFT(8,8)) reaches $32.26$ dB. A more expressive family
ending up $0.97$ dB *worse* is the puzzle. Is $31.29$ a ceiling of the
QFT family, or a floor of the optimiser? The three studies answer this
from three directions, and a fourth study asks whether the training
objective itself was well-chosen.

== Finding 1 — The blocked optimum lies *inside* the QFT family

*(`qft_warmstart_from_trained_blocked/`)* Take the trained blocked
solution, embed its $48$ tensor entries into a full `QFTBasis(8, 8)`
with every other gate pinned to identity — a point whose operator
equals the trained blocked one bit-exactly — then train all $288$
parameters from there. The warm-started QFT reaches $32.26$ dB at
$rho = 0.20$, *exactly the blocked optimum*, versus $31.29$ dB from
random init. (QuickDraw: warm-start $30.07$ dB reproduces the blocked
source $30.05$ within $0.02$ dB; random init lags $5.7$ dB.) So the
blocked optimum is reachable from inside QFT(8,8); the headline gap is
an *optimisation* failure of random-init Adam, not a structural limit
of the family. *Block-QFT $subset$ QFT.*

== Finding 2 — Direct training reaches good QFT operators

*(`qft_identity_init/`, `qft_unfreeze/`)* The analytic-QFT init is not
required. Initialising every gate at its manifold identity
($T_(t=0)(x) = x$) and training under the identical budget reaches
$31.66$ dB at $rho = 0.20$ — $0.37$ dB *above* the analytic-init
`qft`. What carries the result is the *gate topology* (which qubits the
H and controlled-phase gates act on), shared by both inits, not the
analytic phase values. Progressive *gate-unfreezing* — thawing one
gate at a time and training each to a plateau — goes further:
block-growth order from identity reaches $31.98$ dB, approaching the
blocked optimum, and unfreeze order matters ($31.8$–$32.0$ dB for
block-growth / left$arrow.r$right vs $30.8$ for right$arrow.r$left).
Good QFT operators are *trainable directly*, given the right protocol.

== Finding 3 — The training objective (top-k) is a real lever

*(`qft_topk_sweep/`)* Every cell above optimises `MSELoss` on a fixed
top $10%$ of coefficients while evaluation spans $rho in [0.05, 0.20]$.
Sweeping the training top-$k$ shows the fixed $10%$ is *not* optimal
and the best value is dataset-dependent: DIV2K wants a *broader*
objective (top $20%$ reaches $31.66$ dB, $+0.37$ over $10%$), QuickDraw
a *narrower* one (top $5$–$10%$; $>= 15%$ loses $approx 0.8$ dB).
"Train at the rate you deploy at" is the wrong rule — one
dataset-specific top-$k$ is best across all eval rates at once.

== Synthesis

The QFT family is more capable than the headline number suggests. Every
route that escapes random-init Adam's basin climbs toward the blocked
optimum:

#table(
  columns: (auto, auto, auto),
  align: (left, right, left),
  stroke: 0.5pt,
  table.header([*route to a QFT operator*], [*PSNR\@$rho = 0.20$*], [*study*]),
  [random-init joint training (headline)], [31.29], [baseline],
  [identity-init joint training],           [31.66], [Finding 2],
  [top-$k = 20%$ joint training],           [31.66], [Finding 3],
  [progressive gate-unfreezing (bg, id.)],  [31.98], [Finding 2],
  [warm-start from trained blocked],        [*32.26*], [Finding 1],
  [— `blocked_8` reference optimum —],      [32.26], [structure side],
)

Read top to bottom, $31.29$ dB is an *optimisation floor*, not the
family ceiling: the blocked optimum ($32.26$ dB) sits inside QFT(8,8)
(Finding 1), and identity init, a broader training top-$k$, and
progressive unfreezing each reach deeper into the family toward it
(Findings 2–3) — while the training objective that produced the
original floor was itself suboptimal (Finding 3). All training here is
on QFT; the *structure* side (`results/structure/`) extends the
comparison to the other parametric families.

#v(0.4em)
_Per-study detail: `qft_warmstart_from_trained_blocked/writeup.pdf`,
`qft_identity_init/writeup.pdf`, `qft_unfreeze/writeup.pdf`,
`qft_topk_sweep/writeup.pdf`._
