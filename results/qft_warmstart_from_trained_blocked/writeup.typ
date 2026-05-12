#set page(paper: "us-letter", margin: (x: 0.7in, y: 0.7in))
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em, first-line-indent: 0pt)
#show heading.where(level: 1): set block(above: 1.0em, below: 0.5em)
#show heading.where(level: 2): set block(above: 0.8em, below: 0.4em)
#show raw: set text(size: 8.5pt)

= Warm-starting unblocked QFT from a trained blocked basis

*Question.* The headline experiment trains the unblocked
`qft` basis at $m = n = 8$ on DIV2K-8q to $31.29$ dB at $rho = 0.20$,
while `blocked_8` (a `BlockedBasis(QFTBasis(3, 3), 5, 5)` whose
operator is QFT(3,3)$thin times.circle thin$QFT(3,3) replicated across
a $32 times 32$ grid of $8 times 8$ blocks) reaches $32.26$ dB — a
$0.97$ dB gap from a strictly *less* parametrically expressive family.
QFT(8, 8) has $288$ trainable tensor entries; QFT(3, 3) has $48$. The
blocked basis is a special low-dimensional submanifold of QFT(8, 8),
and yet random-init Adam on the larger family ends up worse. Is the
blocked optimum reachable from inside the QFT(8, 8) family, or is it
structurally outside?

*Construction.* The trained blocked solution is a specific point in the
QFT(8, 8) parameter space — by construction. We embed it explicitly:

  $thin thin$ Take the trained $48$ tensor entries of
  `BlockedBasis(QFTBasis(3,3), block_log_m=5, block_log_n=5).inner`.
  Place them in the QFT(8, 8) circuit at the qubit positions
  corresponding to the inner block. Pin every other gate to identity
  ($H arrow.r I_2$, controlled-phase $arrow.r$ phase 0, equivalently
  $[[1, 1], [1, 1]]$). The resulting `QFTBasis(8, 8)` has all $288$
  parameters present and trainable, but its initial operator equals
  the trained blocked operator *bit-exactly*: numerical
  $max_(i,j) abs(T_text("warm") thin x - T_text("blocked") thin x)_(i j) = 0$
  on a random complex $256 times 256$ test input.

After this warm-start, we train the full QFT(8, 8) under the same
preset that produced the headline `qft` and `blocked_8` numbers
($1008$ steps, batch $50$, validation split $0.15$, Adam with cosine LR
warmup, no early stopping).

*Result (QuickDraw, $m = n = 5$).* The same construction with inner
$=$ QFT(3, 3) embedded in a $4 times 4$ block grid (so QFT(5, 5),
$30$ tensors total). After $1008$ steps:

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*basis*], $rho = 0.05$, $rho = 0.10$, $rho = 0.15$, $rho = 0.20$,
  ),
  [`qft` (random init)],         [16.72], [19.58], [22.05], [24.35],
  [`blocked` (warm-start source)], [18.12], [22.40], [26.20], [30.05],
  [`qft_warmstart_blocked`],     [*18.13*], [*22.41*], [*26.22*], [*30.06*],
)

The warm-started run *reproduces the trained blocked PSNR to within
$0.02$ dB at every keep-ratio*. Random-init `qft` lags by $5.7$ dB at
$rho = 0.20$. So on QuickDraw the trained blocked optimum is reachable
from inside the QFT(5, 5) parameter family — what was missing from the
headline `qft` run was *initialization*, not parametric expressivity.

*Result (DIV2K-8q, $m = n = 8$).* The headline gap is $0.97$ dB at
$rho = 0.20$ (`qft` $31.29$ vs `blocked_8` $32.26$). After warm-start
and full re-training:

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*basis*], $rho = 0.05$, $rho = 0.10$, $rho = 0.15$, $rho = 0.20$,
  ),
  [`qft` (random init)],            [25.09], [27.57], [29.53], [31.29],
  [`blocked_8` (warm-start source)], [25.18], [28.09], [30.30], [32.26],
  [`qft_warmstart_blocked_8`],      [*25.18*], [*28.09*], [*30.30*], [*32.26*],
)

The warm-started run reproduces `blocked_8` *to two decimal places at
every keep-ratio*. Random-init `qft` lags by $0.52$, $0.77$, and $0.97$ dB at
$rho = 0.10$, $0.15$, and $0.20$ respectively. So on
DIV2K too the trained blocked optimum sits inside the QFT(8, 8) family
and is reached by Adam from this warm-start init.

#figure(
  image("figures/loss_curves_div2k_8q.svg", width: 100%),
  caption: [DIV2K-8q. Validation MSE per training step (log y).
    Blue: `qft` from random init (~$269 arrow.r 148$). Orange:
    `blocked_8` headline run, the warm-start source (~$159 arrow.r
    129$). Green: `qft_warmstart_blocked_8` — *flat at the trained
    blocked floor for all $1008$ steps* (val MSE stays in
    $[129.32, 129.38]$ across every checkpoint). Adam started at
    the blocked optimum, computed near-zero gradients on every
    minibatch, and the operator did not drift. The qft (blue)
    curve crosses below the blocked starting point around step
    $approx 200$ and converges to its own basin near $148$,
    confirming that the random-init qft basin is *different from
    and worse than* the blocked basin even though the latter sits
    inside the QFT(8, 8) parameter family. Loss definition:
    $sum_(i j) abs(x_(i j) - (T_theta^dagger thin "top"_K (T_theta thin x))_(i j))^2$
    summed over $65536$ pixels of a $256 times 256$ image at
    $K = 6554$ ($rho = 0.10$), averaged over the $75$-image
    validation split.]
)

#figure(
  image("figures/loss_curves_quickdraw.svg", width: 100%),
  caption: [QuickDraw. Validation MSE per training step (log y).
    Same three curves as the DIV2K figure with smaller absolute
    scales (32×32 images, $1024$ pixels, $K = 102$). Blue: `qft`
    random init (~$30 arrow.r 15$). Orange: `blocked` warm-start
    source (~$19 arrow.r 8.7$). Green: `qft_warmstart_blocked`
    again *flat at the blocked floor* (val MSE in $[8.67, 8.68]$
    throughout). Loss definition is the same form as DIV2K with
    $K = 102$, summed over $1024$ pixels, averaged over the
    $75$-image validation split.]
)

*What this rules out.* The headline `qft` $<$ `blocked_8` gap is *not*
a parametric-family limitation — the trained blocked operator lives
inside QFT(8, 8) and inside QFT(5, 5), and Adam keeps it there to
within numerical noise once it starts there. The flat green curves
on both datasets show the warm-start point is a stable local minimum
of the QFT-family loss, not a saddle Adam wanders away from. The
$0.97$ dB DIV2K gap (and $5.7$ dB QuickDraw gap) between random-init
qft and blocked is therefore an *optimisation-from-random-init*
phenomenon: random qft init lies in a different, shallower basin,
and Adam does not cross between basins in $1008$ steps. This
experiment does not characterise *why* — only that the blocked
optimum is achievable in the QFT family given the right starting
point.

*Reproducibility.* `experiments/qft_warmstart_blocked.py
--dataset {quickdraw, div2k_8q}`. The construction of the warm-start
QFT lives in `pdft_benchmarks.bases.qft_warm_from_trained_blocked`
and is asserted bit-exact against the loaded trained blocked basis at
the start of every run.

*Provenance note.* The first version of this experiment (committed
to PR #34 before pdft#16) showed a dramatic green-curve spike at
step $approx 10$, which was misattributed to "Adam's first-update
sign behaviour at a near-flat local minimum." Investigation
(`pdft-benchmarks` PR #34 review thread; root cause in `pdft#16`)
traced the spike to a regression in `pdft.circuit.builder.compile_circuit`
where the H-gate handler in the new stepped-tensordot implementation
applied $"conj"(H)$ to the input on the inverse path instead of the
proper adjoint $H^dagger = "conj"(H^T)$. For symmetric Hadamards the
two coincide; for trained-perturbed Hadamards they differ, inflating
loss by ~$12 times$ on the broken-inverse code path. After fixing
`pdft` and rerunning, the green curve flattens entirely. Both
findings are real and clean; the original "Adam transient" reading
of the figure was an artefact of the bug, not optimisation
behaviour.
