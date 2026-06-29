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

= Which Hadamards freeze

#let gf = json("gate_freezing.json")
#let froz_per_dim = int(gf.n_frozen / 2)

The block code is set by a specific handful of gates. Each of the
#(gf.m + gf.n) Hadamard-role gates is classified by its mixing score
$s = 2 abs(a) abs(b)$ (Hadamard if $s > 0.5$, else frozen to Pauli-Z when
$abs(a) >= abs(b)$ or Pauli-X otherwise). The Haar start is all-mixing
(#gf.init.H Hadamards, #(gf.init.Z + gf.init.X) frozen); training collapses
#gf.n_frozen of them to Pauli — #gf.final.Z Pauli-Z and #gf.final.X Pauli-X —
leaving #gf.final.H still mixing.

The freezing is *positional*: in each dimension it is the
#froz_per_dim most-significant qubits (q5, q6, q7) that collapse to a Pauli,
while the low qubits q0#sym.dash.en q4 keep mixing — so the #froz_per_dim frozen
qubits act as classical block-index bits and the #(gf.m - froz_per_dim) mixing
ones carry the intra-block transform, giving the $#gf.block_row$-pixel block.

#figure(
  image("figures/gate_freezing.svg", width: 96%),
  caption: [*Left:* H-role gate classification, initial vs final — all
  #(gf.m + gf.n) gates begin as Hadamards and #gf.n_frozen freeze
  (#gf.final.Z to Pauli-Z, #gf.final.X to Pauli-X). *Right:* the exact locations
  — the most-significant qubits (q5#sym.dash.en q7) collapse to Pauli (the
  classical block index) in both row and column dimensions, while q0#sym.dash.en
  q4 stay Hadamard (the intra-block transform).])

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
