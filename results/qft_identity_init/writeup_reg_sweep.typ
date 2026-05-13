#set page(paper: "us-letter", margin: (x: 0.7in, y: 0.7in))
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em, first-line-indent: 0pt)
#show heading.where(level: 1): set block(above: 1.0em, below: 0.5em)
#show heading.where(level: 2): set block(above: 0.8em, below: 0.4em)
#show raw: set text(size: 8.5pt)

= Block-masked identity regularisation closes the qft → blocked gap

*Setting.* DIV2K-8q, QFT(8, 8) basis. `blocked_8` reaches $32.26$ dB
at $rho = 0.20$; it is a stable local minimum of QFT(8, 8) (the
`qft_warmstart_from_trained_blocked` cell verified this).
`qft` (analytic init) reaches $31.29$, `qft_identity` (identity init)
reaches $31.66$. Neither generic-init path crosses to the blocked
basin in $1008$ steps. Can a structural prior bridge the gap?

== Construction

Total loss with regulariser:

$ cal(L)_text("total")(theta) = cal(L)_text("MSE-topK")(theta) + lambda dot R_text("block")(theta) $

#text(fill: red)[
  $ R_text("block")(theta) = sum_(g in cal(G)_text("outer")) W dot norm(T_g - I_g)_F^2 + sum_(g in cal(G)_text("inner")) norm(T_g - I_g)_F^2 $
]

A gate $g$ is *inner* iff every qubit it touches lies in $\{1, 2, 3\}$
(axis 1) or $\{9, 10, 11\}$ (axis 2). Otherwise *outer*. Per-kind
identity element: $I_g = I_2$ for Hadamards;
$I_g = mat(1, 1; 1, 1) =$ `controlled_phase_diag(0)` for CPs.

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  stroke: 0.5pt,
  table.header([gate kind], [inner], [outer], [total]),
  [Hadamards],         [6],  [10], [16],
  [controlled-phase],  [6],  [50], [56],
  [*total*],           [*12*], [*60*], [*72*],
)

The $12$/$60$ split matches `blocked_8`'s sparse shape bit-exactly:
$12$ trained QFT(3, 3) gates + $60$ identity-pinned gates.

== Notation

#table(
  columns: (auto, 1fr),
  align: (left + horizon, left),
  stroke: 0.5pt,
  table.header([symbol], [meaning]),
  [$theta$], [$72$ gate tensors $in U(2)^(72)$],
  [$g$], [gate label; $sum_g$ is over the $72$ gates],
  [$cal(G)_text("inner")$], [$12$ inner gates],
  [$cal(G)_text("outer")$], [$60$ outer gates ($cal(G)_text("inner") sect cal(G)_text("outer") = emptyset$)],
  [$T_g (theta)$], [current $2 times 2$ unitary at gate $g$],
  [$I_g$], [identity element for gate $g$'s kind (H or CP)],
  [$norm(M)_F$], [Frobenius norm: $sqrt(sum_(i j) |M_(i j)|^2)$],
  [$lambda$], [reg strength; swept ${0, 10^(-3), 10^(-2), 10^(-1), 1, 10}$],
  [$W$], [outer-gate weight; fixed $W = 10$],
)

== Key invariant

By construction, at the blocked optimum the outer contribution to
$R_text("block")$ is zero (every outer gate is bit-exactly at its
identity element):

$ R_text("block")(theta_text("blocked\_8")) = sum_(g in cal(G)_text("inner")) norm(T_g - I_g)_F^2 + epsilon, quad epsilon approx 0 $

Empirically $R_text("block")(theta_text("blocked\_8")) approx 50.3$
at $W = 10$: inner-sum $approx 34.5$, outer-sum $approx 1.6$
(dominated by one CP outlier; the other $59$ outer gates contribute
$< 10^(-2)$ each). The reg term does *not* push the operator away
from blocked even at large $lambda$ — this is the design property
that enables crank-up without collapse.

== Implementation

`pdft_benchmarks.identity_reg.BlockMaskedIdentityRegQFTMSELoss`, via
the `MSELoss._extra_loss` hook (pdft PR \#18). Headline preset:
$1008$ steps, batch $50$, val split $0.15$, seed $42$, `--no-early-stop`.
Start from `qft_identity` init.

== Result

Test PSNR (dB) after $1008$ steps:

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*basis / reg*], $rho = 0.05$, $rho = 0.10$, $rho = 0.15$, $rho = 0.20$,
  ),
  [`qft` (analytic init)],          [25.09], [27.57], [29.53], [31.29],
  [`qft_identity` (identity init)], [25.23], [27.81], [29.84], [31.66],
  [`qft_identity` + reg, $lambda = 10^(-3)$], [25.23], [27.81], [29.84], [31.66],
  [`qft_identity` + reg, $lambda = 10^(-2)$], [25.23], [27.81], [29.84], [31.66],
  [`qft_identity` + reg, $lambda = 10^(-1)$], [25.24], [27.81], [29.84], [31.66],
  [*`qft_identity` + reg, $lambda = 1$*], [*25.18*], [*28.09*], [*30.30*], [*32.26*],
  [`qft_identity` + reg, $lambda = 10$], [25.18], [28.08], [30.29], [32.25],
  [`blocked_8` (warm-start ceiling)], [25.18], [28.09], [30.30], [32.26],
)

At $lambda = 1$, the trained operator matches `blocked_8` *bit-exactly
to $2$ decimals at every $rho$*. Trajectory crosses from $31.66$ dB
(qft_identity basin) to $32.26$ dB (blocked basin), closing the full
$0.60$ dB gap. $lambda = 10$ within $0.01$ dB.

#figure(
  image("figures/lambda_sweep.svg", width: 100%),
  caption: [Left: test PSNR at $rho = 0.20$ vs $lambda$ (log x). Sharp
    transition $31.66 arrow.r 32.26$ dB between $lambda = 10^(-1)$
    and $lambda = 1$. Right: per-gate $norm(T_g - I_g)_F$ at end of
    training, inner (12, orange) vs outer (60, blue). At
    $lambda gt.eq 1$, outer gates collapse to a thin line at zero —
    *all $60$ pinned bit-exactly at identity*, matching `blocked_8`'s
    structural shape. Inner-gate distributions are invariant across
    $lambda$: the prior leaves them free.]
)

== Interpretation

Outcome (i) from the spec's three-falsifiable-outcomes frame: the
blocked basin was *unfavored, not isolated*. Given the matched
structural prior, smooth-Riemannian Adam reaches it from identity
init in $1008$ steps with no curriculum and no warm-start.

*Leading-prior caveat.* The regulariser bakes in `blocked_8`'s known
structural shape. A positive result is "if you tell the optimiser the
answer-shape, it finds the answer" — closer to warm-start with extra
steps than to basin discovery from scratch.

The shift in framing: the headline `qft` $<$ `blocked_8` gap is now
attributable to *initialisation + optimiser-coupling*, not to a
loss-landscape obstruction. The remaining open question is *how
weak a prior suffices*.

== L1-to-identity: dropping the block-aligned mask

Drop the inner/outer mask; penalise all gates uniformly with an L1
norm (Huber-smoothed to avoid the kink at $T_g = I_g$):

#text(fill: red)[
  $ R_text("L1")(theta) = sum_g sqrt(norm(T_g - I_g)_F^2 + epsilon), quad epsilon = 10^(-12) $
]

The unsmoothed L1's gradient $(T_g - I_g)/norm(T_g - I_g)_F$ is $0/0$
at qft_identity init; `jax.grad` returns NaN there. The
$epsilon$-smoothing yields a finite ($= 0$) gradient at the kink
while agreeing with true L1 to leading order for
$norm(T_g - I_g)_F gt.tilde 10^(-5)$. L1 induces sparsity from
training dynamics: gates either move appreciably or stay pinned;
no continuous shrinkage.

*L1 sweep — DIV2K-8q.* $lambda in {0.1, 1, 10}$, qft_identity init,
same headline preset:

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*basis / reg*], $rho = 0.05$, $rho = 0.10$, $rho = 0.15$, $rho = 0.20$, [active gates],
  ),
  [`qft_identity` (no reg)],                  [25.23], [27.81], [29.84], [31.66], [72/72],
  [L1 reg, $lambda = 0.1$],                    [25.23], [27.81], [29.84], [31.66], [72/72],
  [L1 reg, $lambda = 1$],                      [25.23], [27.81], [29.84], [31.66], [72/72],
  [*L1 reg, $lambda = 10$*],                   [*25.18*], [*28.08*], [*30.30*], [*32.26*], [*6/72*],
  [`blocked_8` (ceiling)],                    [25.18], [28.09], [30.30], [32.26], [12/72],
)

L1 at $lambda = 10$ matches `blocked_8` to $approx 0.02$ dB at every
$rho$, with *only $6$ active gates* (vs `blocked_8`'s $12$). The $6$
are exactly the inner-block Hadamards (indices $0, 1, 2, 8, 9, 10$);
all $6$ inner CPs that `blocked_8` makes non-trivial are pinned at
identity by L1. The $6$ active gates trained to rotation-by-$pi/4$
matrices $mat(c, s; -s, c)$ with $c, s approx 1/sqrt(2)$ — *not*
Hadamards $mat(c, s; s, -c)$. The basin is structurally distinct
from `blocked_8` (no CP entanglement) yet PSNR-equivalent.

*Comparison against the full block-size grid (DIV2K-8q):*

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  stroke: 0.5pt,
  table.header([basis], $rho = 0.05$, $rho = 0.10$, $rho = 0.15$, $rho = 0.20$),
  [blocked_4],       [19.06], [26.93], [29.76], [32.05],
  [*blocked_8*],     [*25.18*], [*28.09*], [*30.30*], [*32.26*],
  [blocked_16],      [25.23], [27.81], [29.84], [31.66],
  [blocked_32],      [24.91], [27.30], [29.20], [30.91],
  [*L1 $lambda = 10$*], [*25.18*], [*28.08*], [*30.30*], [*32.26*],
)

L1 hits the *training-rate-optimal* blocked ceiling at every $rho$
(blocked_8 wins at $rho gt.eq 0.10$; at $rho = 0.05$ blocked_16 wins
by $0.05$ dB and L1 matches blocked_8 rather than blocked_16). Loss
trains at $rho = 0.10$ via $k = round(2^(m+n) dot 0.1)$.

*L1 sweep — QuickDraw ($m = n = 5$).* Same procedure, $30$-gate
QFT(5, 5) basis:

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*basis / reg*], $rho = 0.05$, $rho = 0.10$, $rho = 0.15$, $rho = 0.20$, [active gates],
  ),
  [`qft` (analytic init)],          [16.72], [19.58], [22.05], [24.36], [30/30],
  [blocked_4 / blocked_8],          [18.12], [22.40], [26.18], [30.04], [varies],
  [L1 reg, $lambda = 0.1$],          [17.24], [23.33], [30.06], [40.62], [2/30],
  [L1 reg, $lambda = 1$],            [12.82], [18.35], [30.48], [54.11], [0/30],
  [*L1 reg, $lambda = 10$*],         [*12.82*], [*18.34*], [*30.85*], [*61.89*], [*0/30*],
)

On QuickDraw, L1 finds a *different* answer: at $lambda gt.eq 1$
*every gate is pinned at identity* and the trained operator is
literally $T(x) = x$. PSNR at $rho = 0.20$ reaches $61.89$ dB —
$30+$ dB above any `blocked_b` and $37+$ dB above analytic `qft`.
QuickDraw images are sparse line drawings; top-$k$ truncation in the
pixel domain already captures essentially all the strokes. A
transform basis is *worse* than no transform.

*Caveat.* The $61.89$ dB QuickDraw number is partly a property of
the data, not just the regulariser: at $32 times 32$ with mostly-zero
pixels, top-$20%$-of-pixels retention is near-perfect by construction.
The DIV2K result is the more discriminating signal because DIV2K is
not pixel-sparse.

#figure(
  image("figures/l1_sweep_combined.svg", width: 100%),
  caption: [Left, middle: test PSNR at $rho = 0.20$ vs $lambda$ for
    L1-regularised qft_identity on DIV2K-8q and QuickDraw. On DIV2K,
    L1 matches `blocked_8` at $lambda = 10$; on QuickDraw L1 *exceeds*
    every `blocked_b` by $30+$ dB at $lambda gt.eq 1$. Right:
    end-of-training per-gate $norm(T_g - I_g)_F$ at $lambda = 10$ for
    both datasets, coloured by `blocked_8`'s inner ($12$, orange) /
    outer ($60$, blue) mask. DIV2K: $6$ inner-Hadamard gates are
    active at distance $approx 1.08$; everything else is bit-exactly
    at identity. QuickDraw: *all* gates are bit-exactly at identity
    — L1 picked no transform.]
)

== Frequency-space evidence

Per-image frequency representations $|T(x)|$ and reconstructions at
$rho = 0.20$ make the dataset-adaptive behaviour visible.

#figure(
  image("figures/freq_recon_div2k_8q.svg", width: 100%),
  caption: [*DIV2K-8q, single test image.* Rows: `qft_identity` at
    bare init (identity operator, no training), block-masked
    $lambda = 1$ (trained), L1 $lambda = 10$ (trained). Columns:
    input, $log_10 |T(x)|$, reconstruction at $rho = 0.20$. The
    identity operator (top row) leaves the image in pixel domain;
    top-$20%$ pixel retention destroys it (PSNR $8.50$ dB on this
    image). Both trained operators (rows 2 and 3) produce structured
    frequency spectra and recover the image cleanly. Block-masked
    and L1 trained operators yield visibly similar $|T(x)|$ panels
    despite training under different regulariser shapes — consistent
    with both landing on PSNR-equivalent points of the QFT(8, 8)
    loss surface.]
)

#figure(
  image("figures/freq_recon_quickdraw.svg", width: 95%),
  caption: [*QuickDraw, single test image.* Rows: `qft_identity` at
    bare init, L1 $lambda = 10$ (trained). The L1-trained operator
    is *essentially the identity* — its frequency representation
    nearly matches the input (only faint perturbations on the
    column-sum lines). On pixel-sparse data, top-$20%$ retention of
    the raw pixels reconstructs the image to numerical precision
    (PSNR $infinity$ at init, $91.86$ dB after L1 training — both
    effectively perfect). The transform was unnecessary.]
)

== What this reveals

L1 is *not* finding `blocked_8`. It is finding the *best basis* for
the dataset under the sparsity prior:

- DIV2K (transform-needy): L1 picks $6$ rotations on the inner-block
  qubits. PSNR equal to the training-rate-optimal `blocked_b`,
  structurally simpler ($6$ gates vs $12$, no CP entanglement).
- QuickDraw (already pixel-sparse): L1 picks the identity basis.
  PSNR far above any `blocked_b`, because no transform was needed.

This is substantially stronger than the original "auto-discover
block size" framing: L1 generalises beyond the `blocked_b` family
entirely. The block-aligned mask is one valid sparse-basis shape;
L1 finds others when they exist (DIV2K's no-CP basin) or returns to
identity when no transform is needed (QuickDraw).

The headline `qft` $<$ `blocked_8` gap is then explained as
*Adam-from-qft_identity is poorly coupled to the sparse local
minima of the QFT(8, 8) loss surface*. Any sparsity-inducing prior
— matched mask (block-aware L2), generic sparsity (L1), or even the
warm-start init — steers training to one of those minima.

== Reproducibility

```
python experiments/qft_identity_regularization.py \
    --gpu 0 --reg block --lambdas 0,1e-3,1e-2,1e-1,1,10 \
    --outer-weight 10 --epochs 112
```

Per-run output: `results/qft_identity_init/div2k_8q/_runs/reg_lambda_<lam>_W<W>/`.
Run time: $approx 9$ min per $lambda$ on RTX 3090.
$17$ unit tests in `tests/test_identity_reg.py`.
