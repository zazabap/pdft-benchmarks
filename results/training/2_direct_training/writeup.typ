#set page(paper: "us-letter", margin: (x: 0.7in, y: 0.7in))
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em, first-line-indent: 0pt)
#show heading.where(level: 1): set block(above: 1.0em, below: 0.5em)
#show heading.where(level: 2): set block(above: 0.8em, below: 0.4em)
#show raw: set text(size: 8.5pt)

= Direct training reaches good QFT operators

Part 1 (`1_structure_inclusion/`) showed the trained blocked optimum lives
*inside* the QFT family, reachable by warm-start. This part asks the
complementary question: can good QFT operators be found by training
*directly* — without the blocked solution to start from? Two lines of
evidence say yes. Both train on DIV2K-8q ($m = n = 8$) under the headline
preset ($1008$ steps, seed 42), so they compare directly against the
headline `qft` cell ($31.29$ dB at $rho = 0.20$) and the `blocked_8`
reference optimum ($32.26$ dB).

== Identity init and L1 regularisation (`identity_l1/`)

*The analytic-QFT prior is not required.* Initialising every gate at its
manifold identity ($T_(t=0)(x) = x$) — same gate topology as `qft`, no
analytic phase values — and training under the identical budget reaches
$31.66$ dB at $rho = 0.20$, *$0.37$ dB above* the analytic-init `qft`. What
carries the result is the gate topology (which qubits the Hadamards and
controlled-phase gates act on), shared by both inits; the analytic phases
are incidental. (Detail: `identity_l1/writeup.pdf`.)

*L1-to-identity regularisation finds dataset-adaptive sparse circuits.*
Adding a sparsity term that pulls gates toward identity,
$cal(L) = "MSE-topK"(theta) + lambda dot R(theta)$, swept over $lambda$,
turns training into a search for a *sparse* QFT operator. On DIV2K-8q the
regularised optimum matches `blocked_8` with a structurally *different*
sparse circuit; on sparse QuickDraw sketches it collapses to the *identity*
basis, beating every block transform by tens of dB. A block-masked L2
companion behaves analogously. So direct training, suitably regularised,
*discovers* compression-appropriate structure rather than being handed it.
(Detail: `identity_l1/writeup_reg_sweep.pdf`; the L1-init-anchor transfer
test across other topologies lives in the same cell tree.)

== Progressive gate-unfreezing (`unfreeze/`)

Rather than train all $72$ gates jointly, thaw *one gate per stage* and
train each to a plateau before unfreezing the next. This curriculum reaches
*deeper* into the family than joint training:

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*unfreeze order*], [*identity init*], [*random init*], [],
  ),
  [block-growth (`bg`)],          [*31.98*], [31.66], [PSNR\@$rho = 0.20$ (dB)],
  [left$arrow.r$right (`lr`)],     [31.84], [31.66], [],
  [right$arrow.r$left (`rl`)],     [30.81], [30.81], [],
)

The best curriculum (block-growth from identity, $31.98$ dB) clears the
headline joint-trained `qft` by $0.69$ dB and approaches the blocked
optimum. The *order* matters — growing a square block (`bg`) or following
the QFT construction order (`lr`) both beat the reversed order (`rl`) by
$\~1$ dB, and `rl` is insensitive to initialisation while `bg`/`lr` benefit
from the identity start. (Detail + per-stage staircase dynamics:
`unfreeze/writeup.pdf`.)

== Takeaway

Good QFT operators are *trainable directly*, given the right protocol. The
structural prior that matters is the gate topology, not the analytic phase
values: from a featureless identity init, plain Adam ($31.66$ dB),
L1-regularised search (dataset-adaptive sparse circuits), and
gate-by-gate unfreezing ($31.98$ dB) all reach operators at or above the
headline `qft`, climbing toward the blocked optimum that part 1 placed
inside the family. Part 3 (`3_training_topk/`) then shows the training
objective itself — the top-k truncation rate — is a further lever.
