#set page(paper: "us-letter", margin: (x: 0.9in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set block(above: 0.9em, below: 0.4em)

#let qd = json("quickdraw_5q/headline_50pct.json")
#let dv = json("div2k_8q/headline_50pct.json")
#let f2(x) = str(calc.round(x, digits: 2))
#let f1(x) = str(calc.round(x, digits: 1))
#let pct(p) = f1(100 * p.ratio_vs_raw) + "%"

#align(center)[
  #text(size: 15pt, weight: "bold")[Dataset compression with a trained sparse basis]
  #v(2pt)
  #text(size: 10.5pt)[A real byte-level codec — transform, top-$k$, quantise,
  entropy-code, store, decode — demonstrating full-dataset recovery from
  #sym.lt.eq 50% of the raw bytes]
]

= Setup

Each image is stored as an actual file: coefficients under a basis, top-$k$
by magnitude, symmetric uniform $b$-bit quantisation, a kept-position
bitmask, zlib. The decoder reconstructs from the file alone plus the basis
parameter file (stored once per dataset and counted in every size below).
The codec is identical across bases; only the transform differs. Grid:
$k\/d in {0.05, dots, 0.5}$, $b in {6, 8, 10}$, all 550 images (500 train
\+ 50 held-out test, seed-42 split). All PSNR/SSIM below are on the 50
held-out test images. The real-valued rich bases were retrained at the
headline budget (1008 steps, no early stop) because the original runs
saved only metrics, not parameters (see "Reproduction gate" below).

= Headline: recovery at half the bytes

At a total budget of 50% of raw uint8 size, the best operating points are:

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  table.header([dataset / basis], [B/img], [% raw], [test PSNR (dB)], [test SSIM]),
  [QuickDraw — real rich (trained)],
    [#f1(qd.by_basis.real_rich.bytes_per_image)], [#pct(qd.by_basis.real_rich)],
    [#f2(qd.by_basis.real_rich.test.mean_psnr)], [#f2(qd.by_basis.real_rich.test.mean_ssim)],
  [QuickDraw — block DCT 8×8],
    [#f1(qd.by_basis.block_dct_8.bytes_per_image)], [#pct(qd.by_basis.block_dct_8)],
    [#f2(qd.by_basis.block_dct_8.test.mean_psnr)], [#f2(qd.by_basis.block_dct_8.test.mean_ssim)],
  [DIV2K-8q — real rich (trained)],
    [#f1(dv.by_basis.real_rich_8.bytes_per_image)], [#pct(dv.by_basis.real_rich_8)],
    [#f2(dv.by_basis.real_rich_8.test.mean_psnr)], [#f2(dv.by_basis.real_rich_8.test.mean_ssim)],
  [DIV2K-8q — block DCT 8×8],
    [#f1(dv.by_basis.block_dct_8.bytes_per_image)], [#pct(dv.by_basis.block_dct_8)],
    [#f2(dv.by_basis.block_dct_8.test.mean_psnr)], [#f2(dv.by_basis.block_dct_8.test.mean_ssim)],
)

On QuickDraw the trained real-valued rich basis recovers the dataset at
#pct(qd.by_basis.real_rich) of its raw size with a
#f2(qd.by_basis.real_rich.test.mean_psnr - qd.by_basis.block_dct_8.test.mean_psnr) dB
PSNR advantage over the identical codec built on block DCT. On DIV2K the
two are at parity
(#f2(dv.by_basis.real_rich_8.test.mean_psnr - dv.by_basis.block_dct_8.test.mean_psnr) dB),
consistent with the committed top-$k$ metrics.

#figure(
  grid(columns: 2, gutter: 8pt,
    image("quickdraw_5q/figures/rd_curves.svg"),
    image("div2k_8q/figures/rd_curves.svg")),
  caption: [Rate–distortion in real bytes per image (blob + amortised basis
  file). Left: QuickDraw. Right: DIV2K-8q. Stars mark the best operating
  point within the 50%-of-raw budget (dotted vertical). Grey dotted
  verticals: lossless deflate / PNG references.],
)

= What the advantage measures

#let dpsnr_qd = qd.by_basis.real_rich.test.mean_psnr - qd.by_basis.block_dct_8.test.mean_psnr
#let dpsnr_dv = dv.by_basis.real_rich_8.test.mean_psnr - dv.by_basis.block_dct_8.test.mean_psnr
#let mse_ratio_qd = calc.pow(10.0, dpsnr_qd / 10.0)
#let rms_ratio_qd = calc.pow(10.0, dpsnr_qd / 20.0)
#let mse_ratio_dv = calc.pow(10.0, dpsnr_dv / 10.0)
#let sd_qd = (1.0 - qd.by_basis.block_dct_8.test.mean_ssim) / (1.0 - qd.by_basis.real_rich.test.mean_ssim)
#let sd_dv = (1.0 - dv.by_basis.block_dct_8.test.mean_ssim) / (1.0 - dv.by_basis.real_rich_8.test.mean_ssim)

Per image, with pixels in $[0, 1]$ (peak level $L = 1$) and the decoded
image clamped to that range,

$ "MSE"(x, hat(x)) = 1/(H W) sum_(i, j) (x_(i j) - hat(x)_(i j))^2,
  quad
  "PSNR" = 10 log_(10) (L^2 / "MSE") = -10 log_(10) "MSE" #h(0.5em) "[dB]". $

Reported values are the means of the per-image metrics over the 50
held-out test images.

*PSNR advantage.* The logarithm turns error ratios into differences, so
the advantage at the common byte budget $B$ (here 50% of raw) is a
multiplicative error statement:

$ Delta_"PSNR" = "PSNR"_"rich" (B) - "PSNR"_"DCT" (B)
  = 10 log_(10) ("MSE"_"DCT" / "MSE"_"rich"). $

QuickDraw's $Delta_"PSNR" = #f2(dpsnr_qd)$ dB therefore says that at equal
storage the block-DCT reconstruction carries $#f1(mse_ratio_qd) times$ the
squared error ($#f2(rms_ratio_qd) times$ the RMS pixel error,
$10^(Delta\/20)$) of the trained basis. One aggregation subtlety: since
the table reports the mean of per-image decibel values, the ratio is
between *geometric* means of per-image MSE across the test split, not
arithmetic means. DIV2K's $Delta_"PSNR" = #f2(dpsnr_dv)$ dB is a factor
$#f2(mse_ratio_dv)$ — parity.

*SSIM* (Wang et al. 2004; scikit-image with data range 1: $7 times 7$
windows, $C_1 = (0.01 L)^2$, $C_2 = (0.03 L)^2$) scores window-local
agreement in luminance, contrast and structure, reaching 1 only for
identical images:

$ "SSIM"(x, hat(x)) = 1/M sum_(w = 1)^M
  ((2 mu_x mu_(hat(x)) + C_1)(2 sigma_(x hat(x)) + C_2)) /
  ((mu_x^2 + mu_(hat(x))^2 + C_1)(sigma_x^2 + sigma_(hat(x))^2 + C_2)), $

with means $mu$, variances $sigma^2$ and covariance $sigma_(x hat(x))$
taken over each sliding window $w$. Because scores sit close to 1, the
informative quantity is the *residual structural distortion*
$D = 1 - "SSIM"$, and the advantage is its ratio at matched bytes:

$ D_"DCT" / D_"rich" = (1 - "SSIM"_"DCT") / (1 - "SSIM"_"rich"). $

On QuickDraw $D_"DCT" \/ D_"rich" approx #f1(sd_qd)$ — block DCT retains
about #f1(sd_qd)$times$ the structural distortion of the trained basis at
the same stored size. On DIV2K the ratio is $#f2(sd_dv)$ — parity,
matching the PSNR picture.

= Honest accounting

*Complex coefficients cost double.* The complex rich basis stores re + im
per kept coefficient, so at matched bytes it operates at half the
coefficient count — visible as the orange curve sitting below the
real-valued blue one on both datasets
(QuickDraw: #f2(qd.by_basis.rich.test.mean_psnr) dB;
DIV2K: #f2(dv.by_basis.rich_8.test.mean_psnr) dB at the 50% budget).
This is why the real-valued variant is the storage contender.

*Lossless floor.* QuickDraw sketches are mostly background: lossless
deflate of the raw bytes is already down to $tilde.op$26% of raw size, and
per-image PNG sits at $tilde.op$39% — both vertical references fall well
left of the 50% line in the figure, comfortably below any of the
trained-basis operating points shown. The QuickDraw claim is therefore
basis-versus-basis at a matched lossy budget, not versus-PNG. DIV2K
natural images have no such floor (deflate $tilde.op$ 81%, PNG
$tilde.op$ 65% of raw) and carry the size claim directly.

*DIV2K parity.* On DIV2K the trained basis matches block DCT rather than
beating it, consistent with the committed top-$k$ metrics; the win case is
QuickDraw.

= Reproduction gate

DIV2K retraining reproduced the committed structure-tree cells exactly
(#sym.lt.eq 0.01 dB at every keep ratio, both bases). QuickDraw did not:
absolute PSNR sits 0.7–1.0 dB below the committed cells because the
QuickDraw `.npy` snapshot on this machine differs from the one the
committed run used — even the training-free classical baselines shift
(deterministic transforms, same nominal seed-42 split), while the
real_rich #sym.minus block_dct_8 *gap* reproduces within 0.2 dB at every
keep ratio. Details in `quickdraw_5q/checkpoints/GATE_NOTE.md`. All
numbers in this section are therefore within-run comparisons on the
current data snapshot.
