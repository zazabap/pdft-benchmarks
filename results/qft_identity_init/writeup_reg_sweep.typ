#set page(paper: "us-letter", margin: (x: 0.7in, y: 0.7in))
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em, first-line-indent: 0pt)
#show heading.where(level: 1): set block(above: 1.0em, below: 0.5em)
#show heading.where(level: 2): set block(above: 0.8em, below: 0.4em)
#show raw: set text(size: 8.5pt)

= Block-masked identity regularisation closes the qft → blocked gap

*Question.* The `qft_warmstart_from_trained_blocked` experiment showed
that `blocked_8`'s optimum (32.26 dB\@ $rho = 0.20$ on DIV2K-8q) sits
inside the QFT(8, 8) parameter family: warm-starting full QFT(8, 8) at
the trained-blocked operator yields val MSE flat at $[129.32, 129.38]$
for $1008$ steps, so it is a stable local minimum. The `qft_identity`
ablation showed that vanilla Adam from identity init lands in a
different basin ($31.66$ dB) and the analytic-QFT init lands in yet
another ($31.29$ dB) — neither generic-init path crosses to the blocked
basin in $1008$ steps. *Is the blocked basin reachable from identity
init with the right inductive bias, or is it structurally isolated
under smooth-Riemannian optimisation?*

*Construction.* `blocked_8`'s optimum is a *sparse* configuration in
the QFT(8, 8) gate space: $60$ of $72$ gates pinned to their identity
element, only the $12$ gates on the inner $3$-qubit-per-axis subspace
non-trivial. We add a block-structure-aware L2 pull toward identity to
the loss:

$ cal(L)_text("total")(theta) = cal(L)_text("MSE-topK")(theta) + lambda dot R_text("block")(theta) $

$ R_text("block")(theta) = sum_(g in cal(G)_text("outer")) W dot norm(T_g - I_g)_F^2 + sum_(g in cal(G)_text("inner")) norm(T_g - I_g)_F^2 $

The construction makes four design choices, each motivated by a property
of `blocked_8`'s optimum.

*(1) Per-gate-kind identity elements.* $I_g = I_2 = mat(1, 0; 0, 1)$ for
Hadamards; $I_g = mat(1, 1; 1, 1)$ for controlled-phase (this is
`controlled_phase_diag(0)`, the $2 times 2$ diagonal-stack
representation of the $4 times 4$ identity — CP with phase $0$).
These are *the gates that `blocked_8` has at the outer positions*,
not the QFT analytic values. Choosing the right per-kind reference
is what makes "outer gates at blocked_8's optimum contribute exactly
$0$ to $R_text("block")$" hold bit-exactly, which is the key
invariant below.

*(2) Frobenius-squared distance, not L1 or geodesic.*
$norm(T_g - I_g)_F^2$ is smooth and differentiable everywhere, with
Euclidean gradient $nabla_(T_g) = 2 w_g (T_g - I_g)$ where
$w_g in {1, W}$ is the per-gate weight. For small displacements from
identity (the regime the regulariser targets), the Euclidean
Frobenius distance on $bb(C)^(2 times 2)$ agrees with the $U(2)$
geodesic distance to first order, so the Riemannian projection in
`pdft.train_basis_batched` preserves most of the reg signal. L1 was
a candidate (induces actual sparsity rather than shrinkage) but
introduces non-smoothness at $T_g = I_g$ and was deferred as a
follow-up.

*(3) Inner/outer mask matched to `blocked_8`.* A gate is *inner* iff
every qubit it touches lies in the inner range $\{1, 2, 3\}$
(axis 1) or $\{9, 10, 11\}$ (axis 2). The mask is built by walking
the QFT decomposition's gate sequence
`_qft_gates_1d(m) + _qft_gates_1d(n)` in canonical sort order
(Hadamard-first) and classifying each gate's qubit support — the
same classification used by `qft_warm_from_trained_blocked` to
decide which gates take the trained-inner values vs which are
pinned to identity in the warm-start construction. For QFT(8, 8) at
inner $= (3, 3)$:

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  stroke: 0.5pt,
  table.header([gate kind], [inner], [outer], [total]),
  [Hadamards], [6 (axes 1+2: $q in \{1,2,3,9,10,11\}$)], [10], [16],
  [controlled-phase], [6 ($binom(3, 2) = 3$ per axis)], [50], [56],
  [*total*], [*12*], [*60*], [*72*],
)

The $12$-vs-$60$ split is *bit-exactly* the structural shape of
`blocked_8`'s optimum: $12$ trained QFT(3, 3) gates + $60$
identity-pinned gates.

*(4) Single multiplier $W >> 1$ on the outer set.* Outer gates are
penalised $W$ times more strongly than inner gates. This is
load-bearing: at the blocked optimum, the $12$ inner gates take
*non-trivial* trained-QFT(3, 3) values, so their L2 distance from
identity is large. Under uniform L2 (regulariser family (a),
$W = 1$), a large $lambda$ would pull these $12$ useful gates *back*
toward identity along with the $60$ outer ones — the prior would
*fight* the blocked optimum. The matched-mask weighting separates
the two roles: $lambda$ controls overall strength; $W$ controls how
much harder outer-pinning is enforced than inner shrinkage. Using a
single $W$ rather than two independent multipliers
$(lambda_text("outer"), lambda_text("inner"))$ keeps the sweep
one-dimensional. We fix $W = 10$ for this sweep; $W in {1, 10, 100}$
is a flagged follow-up.

*Key invariant.* The construction guarantees
$R_text("block")(theta_text("blocked\_8")) =
sum_(g in cal(G)_text("inner")) norm(T_g - I_g)_F^2 + epsilon$
with $epsilon approx 0$ (the outer contribution at the blocked
endpoint), so increasing $lambda$ does not push the operator *away*
from blocked. The $epsilon$ is the small drift of $60$ outer gates
during the warm-start cell's $1008$-step training (one CP outlier
contributes $approx 1.1$ to the outer sum; the rest are
$< 10^(-2)$). Quantitatively: at $W = 10$, the inner-sum is
$approx 34.5$ and the outer-sum is $approx 1.6$, giving
$R_text("block")(theta_text("blocked")) approx 50.3$ — dominated by
the *legitimate* inner cost, with only $approx 3 %$ from the small
outer drift. This is the design property that lets us crank
$lambda$ up to $10$ without destabilising training, as the result
section confirms.

Implementation: `pdft_benchmarks.identity_reg.BlockMaskedIdentityRegQFTMSELoss`,
via pdft's `MSELoss._extra_loss` hook (pdft PR \#18). We sweep
$lambda in {0, 10^(-3), 10^(-2), 10^(-1), 1, 10}$ at fixed $W = 10$
under the headline preset ($1008$ steps, batch $50$, val split $0.15$,
seed $42$, `--no-early-stop`), starting from `qft_identity` init.

With per-image MSE $approx 129$ at the blocked optimum, the planned
grid spans reg fractions $0.04 % arrow.r 390 %$ of MSE.

*Result.* Test-set reconstruction PSNR after $1008$ steps:

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

*The blocked basin is reachable from identity init under the matched
structural prior.* At $lambda = 1$ the reg-driven training matches
`blocked_8`'s PSNR *bit-exactly to two decimal places at every
keep-ratio* (25.18, 28.09, 30.30, 32.26 dB). The trajectory crosses
basins from $31.66$ dB (the qft_identity basin) to $32.26$ dB (the
blocked basin), closing the full $0.60$ dB gap. $lambda = 10$ is within
$0.01$ dB of the blocked ceiling at every $rho$ — the reg has saturated
to "pin outer gates" without destabilising training.

*Figure evidence.*

#figure(
  image("figures/lambda_sweep.svg", width: 100%),
  caption: [DIV2K-8q. Left: test PSNR\@ $rho = 0.20$ vs $lambda$ (log
    x). The marker at $lambda = 10^(-7)$ is the $lambda = 0$ cell
    (numerically distinct from the rest of the sweep), star-marked.
    The transition is sharp: $31.66$ dB at $lambda in {0, 10^(-3),
    10^(-2), 10^(-1)}$, jumping to $32.26$ dB at $lambda = 1$ and
    staying within $0.01$ dB at $lambda = 10$. Reference lines: orange
    dashed = analytic-init `qft` ($31.29$); green dotted =
    `blocked_8` ($32.26$). Right: per-gate $norm(T_g - I_g)_F$ at end
    of training, split into inner (12 gates, orange) and outer (60
    gates, blue). At $lambda in {0, 10^(-3), 10^(-2), 10^(-1)}$, outer
    gates spread up to $tilde 0.2$ — Adam drifts them off identity.
    At $lambda in {1, 10}$, *outer gates collapse to a thin line at
    zero* — all 60 are pinned bit-exactly at identity, matching
    `blocked_8`'s structural shape. Inner-gate distributions are
    invariant across $lambda$: the prior leaves them free to take
    their MSE-driven values.]
)

The right panel is the direct mechanistic signature: at $lambda gt.eq 1$
the structural prior is *load-bearing* — outer gates are *pinned*,
inner gates are *free*, and the optimiser lands in the configuration
that produces the blocked PSNR.

*Interpretation.* This is outcome (i) from the spec's
three-falsifiable-outcomes frame: *the blocked basin was unfavored,
not isolated*, under smooth-Riemannian Adam. From the qft_identity
basin, the matched structural prior bridges the gap in 1008 steps with
no curriculum and no warm-start at the blocked operator itself. The
optimiser does not need to discover the 12-vs-60 split; given a prior
that points at it, smooth optimisation reaches the floor.

The sweep also rules out the trivially-strong reg failure mode: at
$lambda = 10$ the reg fraction of MSE at the blocked endpoint is
${approx} 390 %$, well past where "pin everything to identity" would
have driven test PSNR back below qft_identity. Instead it landed
within $0.01$ dB of the blocked ceiling — the *block-masked* shape of
the regulariser is what allows large $lambda$ without collapse: only
outer gates are pulled hard, and the matched W gives the 12 inner
gates the slack to take their trained QFT(3, 3) values.

*Leading-prior caveat.* This regulariser bakes in
`blocked_8`'s known structural shape (the inner/outer mask is matched
to inner $= (3, 3)$, the partition of `blocked_8`). A positive result
is consistent with "if you tell the optimiser the answer-shape, it
finds the answer" — closer to warm-start with extra steps than to
basin discovery from scratch. What this experiment *does* establish:
the blocked basin is *not pathological under smooth-Riemannian
optimisation* — there is a stable trajectory from identity init that
ends inside it. What it *does not* establish: whether weaker priors
(uniform L2-to-identity, L1-to-identity) would also bridge the gap.
Those are the natural follow-ups; if even uniform L2 closes the gap,
the structural information was not load-bearing.

*What this opens up.* The headline `qft` $<$ `blocked_8` gap is now
attributable to *initialisation + optimiser-coupling* rather than to a
loss-landscape obstruction. Random or analytic init places Adam in a
basin from which it doesn't cross to blocked in 1008 steps; matched
prior placement does. This shifts the research question: from "is the
blocked optimum reachable in the QFT family?" to "what is the
*minimum* structural information needed to reach it?" Future work:
sweep $W in {1, 10, 100}$ (with $W = 1$ degenerating to uniform L2);
implement L1-to-identity for comparison; characterise the boundary
between the two basins (e.g., via interpolation along a
linear/geodesic path).

*Reproducibility.*

```
python experiments/qft_identity_regularization.py \
    --gpu 0 --lambdas 0,1e-3,1e-2,1e-1,1,10 \
    --outer-weight 10 --epochs 112
```

Construction lives in
`pdft_benchmarks.identity_reg.BlockMaskedIdentityRegQFTMSELoss`. The
hook in `pdft.loss._scalar_loss` (zazabap/pdft PR \#18) lets the loss
subclass add the reg term via an `_extra_loss(tensors)` method.
Per-run output: `results/qft_identity_init/div2k_8q/_runs/reg_lambda_<lam>_W<W>/`
(metrics, loss_history, trained tensors). Run time: $approx 9$ min per
$lambda$ on RTX 3090 (cuda:0), $approx 54$ min total for the sweep.
$13$ unit tests covering the identity table, inner mask, and
reg-loss arithmetic; $lambda = 0$ end-to-end equivalence to vanilla
MSELoss training is locked in by
`tests/test_identity_reg.py::test_reg_loss_lam_zero_e2e_matches_base_mse_training`.
