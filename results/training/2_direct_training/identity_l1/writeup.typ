#set page(paper: "us-letter", margin: (x: 0.7in, y: 0.7in))
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em, first-line-indent: 0pt)
#show heading.where(level: 1): set block(above: 1.0em, below: 0.5em)
#show heading.where(level: 2): set block(above: 0.8em, below: 0.4em)
#show raw: set text(size: 8.5pt)

= Identity-initialised QFT on DIV2K-8q

*Question.* The canonical `qft` basis is constructed from the
analytic QFT decomposition: every Hadamard is the exact
$[[1, 1], [1, -1]] / sqrt(2)$ matrix and the $k$-th controlled-phase
gate carries phase $phi = 2 pi / 2^k$. This is a strong prior — the
init *is* the textbook FFT operator. After $1008$ training steps the
operator drifts to a point that improves test PSNR by $1.9$ dB over
the analytic FFT at $rho = 0.20$, suggesting the optimum is not the
analytic QFT but something nearby that Adam reaches from there. Two
readings are possible: (i) the analytic-QFT prior is *useful* and
training amounts to a short refinement of an already-good
initialisation, or (ii) the analytic prior is *incidental* and Adam
on the U(2) Riemannian manifold finds the same optimum from any
reasonable start. This experiment tests reading (ii) directly.

*Construction.* We add a `qft_identity` factory in
`pdft_benchmarks.bases` that has the *same gate topology and gate
count* as `qft` — same Hadamards on the same qubits, same
controlled-phase gates between the same pairs — but pins every gate
to the identity element of its manifold at $t = 0$:

  $thin thin$ Hadamard $arrow.r I_2$, and controlled-phase $arrow.r$
  `controlled_phase_diag(0)` $= [[1, 1], [1, 1]]$ (the $2 times 2$
  diagonal-stack representation of the $4 times 4$ identity).

The resulting `QFTBasis(8, 8)` has all $288$ tensor entries trainable
on the U(2) manifold, but its initial operator is the literal
identity: $T_(t=0)(x) = x$ for any image $x$. Sanity check: on a
random complex $256 times 256$ test input,
$norm("forward"_(t=0)(x) - x) = 0$ exactly.

Training uses the identical preset and seed as the headline `qft`
cell: $1008$ steps, batch $50$, validation split $0.15$, Adam with
cosine LR warm-up, `--no-early-stop`, DIV2K-8q at $m = n = 8$.

*Result.* Test-set reconstruction PSNR after $1008$ steps:

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*basis*], $rho = 0.05$, $rho = 0.10$, $rho = 0.15$, $rho = 0.20$,
  ),
  [`fft` (untrained)],          [24.50], [26.54], [28.07], [29.39],
  [`qft` (analytic init)],      [25.09], [27.57], [29.53], [31.29],
  [`qft_identity` (identity init)], [*25.23*], [*27.81*], [*29.84*], [*31.66*],
  [$Delta$ (identity − analytic)], [+0.14], [+0.24], [+0.31], [+0.37],
)

`qft_identity` matches or beats `qft` at every keep-ratio, by
$0.14$–$0.37$ dB. The gap to the untrained `fft` (which equals the
$t = 0$ operator of `qft`) is $0.7$–$2.3$ dB, of which the analytic
init buys $0.6$–$1.9$ dB and the identity init buys $0.7$–$2.3$ dB.
SSIM moves in the same direction at every $rho$ ($+0.004$ to
$+0.006$); mean MSE is about $3 %$ lower across all rates.

*Loss-curve evidence.*

#figure(
  image("figures/loss_curves_div2k_8q.svg", width: 100%),
  caption: [DIV2K-8q. Left: per-step training MSE (log y). Right:
    validation MSE per epoch (log y; $1$ val pass every $9$ steps).
    Blue solid: `qft_identity` (identity init). Orange dashed: `qft`
    (analytic init). The two curves start $38 times$ apart in
    training loss ($11","561$ vs $300$ at step $1$) because the
    identity operator at $rho = 0.20$ keeps energy in the pixel
    domain rather than concentrating it at DC. Identity-init loss
    crosses below analytic-init at step $approx 173$ and stays
    there. Final training MSE: identity $151.11$ vs analytic
    $160.03$ ($-5.6 %$); final validation MSE: $139.85$ vs $148.39$
    ($-5.8 %$). Loss definition:
    $sum_(i j) abs(x_(i j) - (T_theta^dagger thin "top"_K (T_theta thin x))_(i j))^2$
    summed over $65","536$ pixels of a $256 times 256$ image at
    $K = 13","108$ ($rho = 0.20$ for the eval table; the training
    loss aggregates all four rates per the preset).]
)

*Interpretation.* Reading (ii) is consistent with the evidence. The
analytic-QFT prior is *not* required: identity-init Adam, given the
same gate topology and the same $1008$-step budget, finds an operator
that reconstructs DIV2K test images $approx 0.3$ dB *better* than the
analytic-init run at the headline rate. The $38 times$ start-loss
penalty washes out within $approx 170$ steps and the two trajectories
end in adjacent points of the same flat top-K MSE valley described
in the headline writeup — but the identity-init endpoint is a hair
deeper. Plausible mechanisms (this experiment does not distinguish
between them):

- *Random regularisation.* Identity init perturbs the operator off
  the analytic-QFT subset earlier in training; the trajectory passes
  through more of parameter space and is less likely to be pinned at
  an analytic-QFT-adjacent saddle.

- *Better effective LR schedule.* Cosine LR is calibrated to the
  $1008$-step budget. From a high-loss start, the early high-LR
  phase does more useful work (gross structure to learn) than from
  an already-near-optimum analytic init (where high LR is mostly
  destabilising). The two final basins are equally accessible; the
  schedule favours the one with more room to use early-step LR.

*What this rules out.* The conjecture "`qft` works because the
analytic QFT is structurally close to the dataset optimum" is *not*
needed to explain the headline `qft` result. The same Adam +
manifold-Riemannian setup from a featureless identity init reaches
an equivalent (and slightly better) endpoint. The structural prior
that *does* matter is the *gate topology* — which qubits the H
gates and CP gates act on — and that is shared between `qft` and
`qft_identity` by construction.

*Reproducibility.* `python experiments/div2k_8q_pca_vs_block_dct.py
--gpu 0 --bases qft_identity --no-early-stop --epochs 112 --out
results/qft_identity_init/div2k_8q/_runs/run1`. Construction lives in
`pdft_benchmarks.bases.qft_identity_basis(m, n)`. Run time was
$541$ s on an RTX 3090 (`cuda:0`, JAX FP64 complex circuit), within
$0.1 %$ of the analytic-init `qft` cell ($540.6$ s) — the
initialisation choice does not affect compute cost.
