#set page(paper: "us-letter", margin: (x: 0.9in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set block(above: 0.9em, below: 0.4em)
#show raw: set text(size: 8.5pt)
#set math.equation(numbering: "(1)")

#let s = json("block_emergence_spectrogram.json")
#let nmix = int(calc.round(calc.log(s.block_row_final, base: 2)))
#let nfroz = 8 - nmix
#let nblk = int(256 / s.b_star)
// canonical published QFT (a *different* Haar init): 16-px block, seed 98.
#let canon_block = 16
#let canon_froz = 8 - int(calc.round(calc.log(canon_block, base: 2)))

#align(center)[
  #text(size: 15pt, weight: "bold")[Emergent block structure during training]
  #v(2pt)
  #text(size: 10.5pt)[How a Haar-random `QFTBasis(8,8)` factorises itself into a
  block code — and why the block *scale* is initial-value dependent]
  #v(3pt)
  #text(size: 9pt, fill: rgb("#555"))[DIV2K-8q, Haar seed-#s.seed ·
  Generated #datetime.today().display("[year]-[month]-[day]")]
]

= Claim

The trained full-image QFT factors itself into a *block code with no block prior*
(paper sec5.3): roughly half of the Hadamard-role gates collapse to non-mixing
Pauli-Z/X (classical block indices) while the controlled-phase gates stay rich,
so the $256 times 256$ transform becomes block-diagonal up to a permutation. The
companion endpoint analysis reads this off the *converged* operator. Here we read
it off the *training trajectory*: we train one `QFTBasis(8,8)` from Haar-random
init (seed #s.seed) and snapshot the live operator at *every* one of the
#(s.loss_steps.len()) steps, so we can watch the block code crystallise — and
see that *which* block scale it lands on depends on the random start.

= Frequency space — the emergence spectrogram

Each snapshot's operator is read through its mean test-set *coefficient power
spectrum*. The 2-D transform is separable, $F = W_t X W_t^top$, so it factorises
per axis; writing $W_t in CC^(N times N)$ ($N = #s.N$) for the 1-D factor and
$Sigma$ for the image-axis pixel covariance,

$ p_t (f) = [W_t Sigma W_t^dagger]_(f f)
  = sum_(k,l) W_t [f,k] thin Sigma[k,l] thin overline(W_t [f,l])
  = bb(E)_x [thin abs((W_t x)_f)^2 thin], $ <eq-power>

i.e. the *energy* — the squared coefficient magnitude $abs((W_t x)_f)^2$ —
placed in coefficient $f$, averaged over image lines $x$ and the untransformed
axis. Since $W_t$ is unitary up to scale, $sum_f p_t (f) = norm(x)^2$ (Parseval),
so $p_t$ is purely how a fixed energy budget is distributed across coefficients.
The covariance is estimated from the #s.n_test DIV2K test
images $X^((n))$, and the two separable axes are averaged:

$ Sigma_"row"[k,l] = 1/(n_"test" N) sum_n sum_j X^((n))_(k j) thin overline(X^((n))_(l j)),
  quad p_t = 1/2 (p_t^"row" + p_t^"col"). $ <eq-cov>

The spectrogram colours $log_10 (p_t (f) \/ max_f p_t (f))$ (peak-normalised per
column). The factor $W_t$ is materialised from the checkpoint gates by
$W_t [dot.c, k] = T_theta (e_k)$, and @eq-power is checked against a direct
transform of the test images (agreement $approx 10^(-13)$).

#figure(
  image("figures/block_emergence_spectrogram.svg", width: 100%),
  caption: [*Training-time Fourier power-spectrum spectrogram.* Mean test-set
  coefficient power spectrum $p_t (f)$ (peak-normalised $log_10$, viridis) vs
  training step (linear $x$, cropped to the emergence window — the structure is
  flat through the final step 1008); the strip above is the run's per-step
  training loss on a shared step axis. At init the Haar-random operator gives a
  *structureless* spectrum (left) — a near-flat band with a few scattered peaks,
  not a low-frequency lobe. Over training it reorganises into the block-periodic
  comb (right). The orange staircase (right axis) is the gate-based effective
  block size $2^(n_("mix"))$ for the row (solid) and column (dotted) dimensions,
  falling $256 -> #s.b_star$ as Hadamard-role gates freeze; the dashed marker is
  step #s.emergence_step, where both dimensions reach their final
  $#s.block_row_final times #s.block_col_final$ block.])

Two readings stand out. First, the block code forms by a *halving cascade*. The
order parameter (orange staircase) is gate-based: for a Hadamard-role gate
$U in CC^(2 times 2)$ the mixing score is $s = 2 abs(U_(0 0)) thin abs(U_(0 1)) in [0,1]$
($1$ = Hadamard, $0$ = frozen Pauli), and the per-dimension effective block size

$ b = 2^(n_"mix"), quad n_"mix" = abs({"H-role gates with" thin s > 0.5}). $ <eq-block>

Hadamard-role gates freeze one at a time, so $b$ drops $256 -> 128 -> 64 -> 32$
and the spectral comb sharpens in lockstep. The comb is exactly the frequency-space
signature of block-diagonality: with the same intra-block transform $W_b$ acting in
all $N\/b$ blocks, $W_t approx I_(N\/b) times.o W_b$ up to a permutation, so
$p_t (f)$ becomes *periodic in the coefficient index* $f$ with period the block size
$b$ — here $#s.b_star$ pixels, i.e. $#{int(256 / s.b_star)}$ identical sub-spectra
(the $#{int(256 / s.b_star)}$ teeth). Gate collapse and this periodic comb are the
same event seen two ways. Second, the cascade *finishes after the loss has
largely flattened*: the last halving lands at step #s.emergence_step, well inside
the flat top-$k$ MSE valley, so the fine block structure is a property of the
optimum rather than something that drives the loss down — consistent with the
flat-valley / discrete-basin picture in the seed-robustness study.

= Operator space — the same cascade in $|W|$

#figure(
  image("figures/block_emergence.svg", width: 92%),
  caption: [The operator's 1-D factor $log_10 |W|$ at eight training steps. The
  Haar init (step 0) is dense — every input pixel drives every output
  coefficient. As gates freeze, $|W|$ collapses onto a block-diagonal (up to the
  QFT bit-reversal + frozen-X permutation), ending at #nblk blocks of
  #s.b_star pixels. This is the operator-space twin of the frequency-space
  spectrogram above: the off-block energy draining away *is* the comb sharpening.])

= How close is it to an *exact* block transform?

#let bpr = json("block_projection_residual.json")
#let pct(x, d) = str(calc.round(x * 100, digits: d)) + "%"

The cleanest way to put a number on "how block" the operator is, is its distance
to the nearest exact block transform. For a candidate block size $b$ we project
the trained 1-D factor $W$ ($N times N$, $N = #bpr.N$) onto block-diagonal form
and keep only what is left over:

+ partition the $N$ indices into $K = N\/b$ contiguous blocks;
+ the QFT permutes blocks (bit-reversal $+$ frozen-X), so match each input block
  to its output block by the Hungarian assignment on the block-energy matrix
  $C_(i j) = sum_(a in "out-block" i,\ c in "in-block" j) abs(W_(a c))^2$;
+ *project* — zero every entry outside a matched (in $->$ out) block — to get
  $Pi_b [W]$;
+ *residual* $r(b) = norm(W - Pi_b [W])_F \/ norm(W)_F$: the fraction of the
  operator's amplitude lying off the block structure. (It equals
  $sqrt("leakage")$ — the amplitude-domain twin of the off-block *energy* used
  in the sweep above.)

The full image transform is separable, $T = W_"row" times.o W_"col"$, so its
residual is $r_(2"D") = sqrt(1 - (1 - r_"row"^2)(1 - r_"col"^2))$. Sweeping $b$,
$r$ sits near $1$ while $b$ is below the true block and then *collapses* — the
knee locates the block scale, and the value there is the approximation error.

#figure(
  image("figures/block_projection_residual.svg", width: 74%),
  caption: [Relative Frobenius distance from the trained operator to its
  block-diagonal projection, vs candidate block size. A razor-sharp knee at
  $b = #bpr.knee$ pixels: the $2$-D residual falls from #pct(bpr.r_below_knee_2d, 0)
  at $b = #(bpr.knee / 2)$ to #pct(bpr.r_knee_2d, 2) at $b = #bpr.knee$. The
  trained image transform is thus within #pct(bpr.r_knee_2d, 2) (relative
  Frobenius) of an *exact* $#bpr.knee times #bpr.knee$ block transform, and is
  not close to any finer one.])

A caveat on the *form* of the block code: the operator is block-*localised*
(each block maps to one block, #pct(bpr.r_knee_2d, 2) off-block) but the
intra-block transforms are *not identical* across blocks — they agree only up to
per-block diagonal twiddle phases, so the distance to a literally repeated
$I_K times.o W_b$ stays large. This is why the power-spectrum comb still tiles
(phase-blind, same per-block power) even though the complex blocks differ.

= Which Hadamards freeze

#let gf = json("gate_freezing.json")             // this run (seed 0, block 32)
#let gf16 = json("gate_freezing_16px.json")      // canonical (seed 98, block 16)
#let froz_per_dim = int(gf.n_frozen / 2)
#let froz16 = int(gf16.n_frozen / 2)

The block code is set by a specific handful of gates. Each H-role gate is
classified by its mixing score $s = 2 abs(a) abs(b)$ (Hadamard if $s > 0.5$,
else frozen to Pauli-Z when $abs(a) >= abs(b)$ or Pauli-X otherwise). The
freezing is *positional*: it is always the most-significant qubits per dimension
that collapse to a Pauli — classical block-index bits — while the low qubits
keep mixing as the intra-block transform.

*This run* (seed #gf.seed, @fig-gate-freeze top): all #(gf.m + gf.n) gates start
Hadamard; #gf.n_frozen freeze (#gf.final.Z Pauli-Z, #gf.final.X Pauli-X), namely
q5#sym.dash.en q7 in both dimensions, leaving the $#gf.block_row$-pixel block.
*The canonical seed* (seed #gf16.seed, @fig-gate-freeze bottom): one more qubit
per dimension freezes — q4#sym.dash.en q7 — so #gf16.n_frozen go Pauli
(#gf16.final.Z Pauli-Z, #gf16.final.X Pauli-X), leaving the finer
$#gf16.block_row$-pixel block. Same mechanism one rung deeper: the *number* of
frozen qubits, hence the block scale, is what the initialisation selects.

#figure(
  stack(spacing: 9pt,
    image("figures/gate_freezing.svg", width: 94%),
    image("figures/gate_freezing_16px.svg", width: 94%)),
  caption: [Gate freezing for two initialisations. *Top:* seed #gf.seed —
  #gf.n_frozen gates freeze (q5#sym.dash.en q7 per dimension) into the
  $#gf.block_row$-px block. *Bottom:* seed #gf16.seed, the canonical paper
  operator — #gf16.n_frozen freeze (q4#sym.dash.en q7) into the finer
  $#gf16.block_row$-px block. Left panels: H / Pauli-Z / Pauli-X counts, initial
  vs final; right panels: the exact frozen qubit positions. The most-significant
  qubits always collapse to Pauli; the initialisation sets how many.])
  <fig-gate-freeze>

= Initial-value dependence

The *existence* of a block factorisation is robust — every Haar start we have
trained discovers one with no block prior. The *scale* is not: it is set by how
many Hadamard-role gates happen to freeze, which depends on the initialisation.
This seed-#s.seed run keeps #nmix Hadamards mixing per dimension (freezes
#nfroz), leaving a #(s.b_star)-px block (#nblk blocks). The canonical published
operator — a *different* Haar init — freezes one more per dimension (#canon_froz
vs #nfroz), landing at the finer #(canon_block)-px block (#{int(256 / canon_block)}
blocks). Same mechanism, different depth: the network reliably finds a block
code, while the random start selects which rung of the
$256 -> 128 -> 64 -> 32 -> 16$ cascade it stops on — the operator-level analogue
of the seed-to-basin endpoint spread documented in the seed-robustness writeup.
