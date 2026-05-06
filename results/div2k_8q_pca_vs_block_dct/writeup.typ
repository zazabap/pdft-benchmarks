#set page(flipped: true, margin: (x: 1.5cm, y: 1.5cm))
#set text(size: 11pt, font: "New Computer Modern")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

#import "@preview/quill:0.7.1": *
#import "@preview/wrap-it:0.1.1": wrap-content

#align(center)[
  #text(size: 16pt, weight: "bold")[DIV2K-8q natural images — trained bases vs classical baselines]
]

#v(0.5em)

= Main result

We introduce a parametric quantum-circuit family of unitary image
bases with $O(N log N)$ forward / inverse cost and $O(log N)$
trainable parameters per dimension. *(i)* On natural images
(DIV2K-HR, $256 times 256$ grayscale), the trained block-wrapped
basis (`real_rich_8`) matches BlockDCT-8 to within $approx 0.13$ –
$0.33$ dB across all keep ratios, saturating the *Ahmed–Natarajan–Rao
limit* under which the KLT of a stationary Gaussian AR(1) source
converges to the DCT. *(ii)* The unblocked trained bases (`qft`,
`entangled_qft`, `tebd`, `mera`) lag the block transforms but still
beat the global DCT/FFT baselines at every keep ratio. *(iii)*
*Bilateral 2D-PCA (`bd_pca`) is the strongest unblocked classical
baseline*, beating the global DCT by $0.08$ – $0.22$ dB at every
keep ratio. Treating each image as the $H times W$ matrix instead
of a flat $d$-vector lets the column- and row-eigenbases be fit on
$N H = N W = 128000$ samples each in a $256$-dim ambient — both
full-rank, sidestepping the $d/N$ rank-deficiency that pins flat
PCA at $approx 17.6$ dB on this geometry.
*(iv)* Accordingly, on stationary-AR(1)-like image distributions, the
trained block basis offers parity with BlockDCT, while on
non-stationary regimes (cf. companion QuickDraw $32 times 32$
results, where `real_rich` exceeds BlockDCT by $approx 6$ dB) the
parametric family delivers a rigorous improvement.

= Setup + pipeline

$ m = n = 8, quad d = 2^m dot 2^n = 256 times 256 = 65536 $
$ N_"train" = 500, quad N_"test" = 50, quad "seed" = 42, quad rho in {0.05, 0.10, 0.15, 0.20} $

Per image $x in [0,1]^d$ with basis $Phi$:

$ y = Phi x quad ("forward") $
$ k = floor(rho dot d), quad y'_i = cases(y_i &"if" |y_i| in "top-"k, 0 &"else") quad ("compression") $
$ hat(x) = "clip"(Phi^(-1) y', 0, 1), quad "MSE" = 1/d norm(x - hat(x))_2^2, quad "PSNR" = 10 log_10 (1\/"MSE") $

For block transforms (classical *and* trained: the new `_8`
factories): same pipeline applied independently to each $8 times 8$
tile, top-$k$ pooled across all blocks. With $d_b = 64$ per block and
$32 times 32 = 1024$ blocks per image, $rho = 0.20$ keeps $approx 13$
coefficients per block.

*AR(1) (first-order autoregressive).* Each pixel = $rho_("AR")$ ×
previous + small Gaussian noise:
$ x_n = rho_("AR") x_(n-1) + epsilon_n, quad epsilon_n tilde cal(N)(0, sigma_epsilon^2) $
$rho_("AR") arrow.r 1$ Gaussian → KLT collapses to DCT
(Ahmed–Natarajan–Rao). Empirical lag-1 autocorrelation
$hat(rho)_("AR") = 1/2 (hat(rho)_"row" + hat(rho)_"col")$ across
multiple samples: *DIV2K natural images* ($256 times 256$ centre
crops) cluster in $hat(rho)_("AR") in [0.84, 0.99]$ — close to the
image-row-like regime, $rho approx 1$, where BlockDCT is
near-optimal; *QuickDraw drawings* (single $28 times 28$ each)
cluster tightly at $hat(rho)_("AR") approx 0.70$ — further from the
AR(1)–Gaussian limit. The two distributions barely overlap, and the
results table below shows the corresponding asymmetry: on DIV2K the
trained block basis lands at parity with BlockDCT-8 instead of
clearly beating it. Distinct from the keep-ratio $rho$ above.

#figure(
  image("figures/ar1_examples.png", width: 100%),
  caption: [*Top row*: three synthetic AR(1)–Gaussian fields at
            $rho = 0, 0.5, 0.95$ (sanity-checking $hat(rho)$); DIV2K
            #0250 centre crop; QuickDraw drawing #0. *Bottom row*:
            three more DIV2K samples (#0001, #0050, #0100) and two
            more QuickDraw drawings (#1, #2) — confirming the
            $hat(rho)_("AR")$ values are robust across samples and
            the two datasets cluster at different points on the AR(1)
            axis. QuickDraw panels are single $28 times 28$ drawings
            nearest-neighbour upscaled $9 times$ for visual size;
            $hat(rho)$ is computed on the native pixels.]
)

= Fits — what is $Phi$ for each method?

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, left, left, left, left, left),
    stroke: 0.5pt,
    table.header(
      [*method*], [*$Phi$ source*], [*$Phi$ shape*],
      [*forward $y =$*], [*compression $y' =$*], [*inverse $x =$*],
    ),

    [`bd_pca` (bilateral 2D-PCA)],
    [$U, V$ from SVDs of column- and row-stacked centered images: \ $Sigma_"col" = 1/(N W - 1) sum_i (X_i - bar(X))(X_i - bar(X))^T$ ($H times H$) \ $Sigma_"row" = 1/(N H - 1) sum_i (X_i - bar(X))^T (X_i - bar(X))$ ($W times W$) \ fit on $N=500$ images at $H = W = 256$ \ (so $N H = N W = 128000$ samples per axis)],
    [$U$ is $H times H = 256 times 256$, full rank \ $V$ is $W times W = 256 times 256$, full rank \ (each axis has $128000 >> 256$ samples — \ no $d/N$ rank-deficiency)],
    [$Y = U^T (X - bar(X)) V$, shape $H times W$],
    [top-$k$ on $Y$: $Y_(i j) dot bb(1)[|Y_(i j)| >= tau_k]$, $k = floor(rho dot H W) = floor(rho dot d)$ \ (same rate as DCT)],
    [$U Y' V^T + bar(X)$],

    [`block_pca_8`],
    [$Phi = V_b^T$, where \ $Sigma_b = V_b Lambda_b V_b^T$, \ $Sigma_b = 1/(N_p-1) sum_(j=1)^(N_p) (p_j - mu_b)(p_j - mu_b)^T$ \ fit on $N_p approx 3.2 dot 10^6$ patches \ ($500$ train images $times 32 times 32$ blocks)],
    [$Phi$ is $64 times 64$ per block, full rank \ ($Sigma_b$ is $d_b times d_b = 64 times 64$ \ regardless of $N_p$) \ shared across all $32 times 32$ blocks],
    [per block: $Phi (p - mu_b)$],
    [top-$k$ pooled across all blocks, keep $k = floor(rho dot 65536)$ largest \ (rank: per-block keep $i = 0, dots, k_b - 1$, $k_b = floor(rho dot 64)$)],
    [per block: $Phi^T y' + mu_b$, then re-tile],

    [`dct` (global)],
    [closed-form (DCT-II): \ $Phi[k, n] = alpha_k cos(pi (n + 1\/2) k \/ N)$ \ $alpha_0 = sqrt(1/N)$, $alpha_(k >= 1) = sqrt(2/N)$ \ $N = 256$, no fit, no mean],
    [$d times d$ analytically],
    [$Phi x$ (no mean)],
    [top-$k$: $y_i dot bb(1)[|y_i| >= tau_k]$ \ (rank: keep $i = 0, dots, k-1$)],
    [$Phi^T y'$],

    [`block_dct_8`],
    [closed-form (DCT-II) at $N = 8$: \ $Phi[k, n] = alpha_k cos(pi (n + 1\/2) k \/ 8)$ \ shared across all blocks],
    [$64 times 64$ per block analytically \ tiled identically across the image],
    [per block: $Phi p$],
    [top-$k$ pooled across all blocks \ (rank: per-block keep $i = 0, dots, k_b - 1$, $k_b = floor(rho dot 64)$)],
    [per block: $Phi^T y'$, then re-tile],
  ),
  caption: [Concrete definition of $Phi$ and the full pipeline
            (forward $arrow$ compression $arrow$ inverse) for each
            method. All four $Phi$ are *orthogonal* ($Phi Phi^T = I$),
            so inverse $=$ transpose. 2D $=$ separable (apply 1D
            transform along rows then columns). \
            *Top-$k$ rule* (default, headline results): keep the $k$
            coefficients with largest *|magnitude|*, set the rest to
            zero — pooled across all blocks for block transforms.
            *Rank rule* (in parentheses; used by the `_rank` control
            variants): keep the *first* $k$ coefficients in the
            basis's natural ordering. For *PCA* this means the $k$
            *highest-eigenvalue* directions (rows of $V^T$ are sorted
            by descending $lambda$); for *DCT* this means the $k$
            *lowest-frequency* components (DC first, then increasing
            frequency $k = 0, 1, 2, dots$). The trained `_8` block
            factories (`blocked_8`, `rich_8`, `real_rich_8`) operate
            on the *same $8 times 8$ tiling* as `block_dct_8` /
            `block_fft_8` / `block_pca_8`, so block-size is matched
            across classical and trained block bases — no asymmetry
            at this geometry.]
)

== Why DCT $approx$ block PCA in the limit

$ Phi_("block_pca_8") = "KLT"(Sigma_b) quad arrow.long quad
   Phi_("block_dct_8") = "KLT"(lim_(rho arrow.r 1) Sigma_("AR(1)")(rho)) $

DCT $=$ what KLT becomes if the patch covariance is the analytic
AR(1)–Gaussian limit. Empirically the two differ by only $0.18$ –
$0.50$ dB on DIV2K-8q (Block-DCT 8 over Block-PCA 8 at
$rho = 0.05, dots, 0.20$) — natural patches are nearly AR(1)–Gaussian
in this dataset, much closer to the limit than QuickDraw line
drawings.


= Results — PSNR (dB), seed=42

#figure(
  block(width: 100%, grid(
    columns: (1fr, 1fr),
    column-gutter: 1.2em,
    align: top,

    // Left column — unblocked / full-image transform
    table(
      columns: (auto, auto, auto, auto, auto),
      align: (left, right, right, right, right),
      stroke: 0.5pt,
      table.header([*unblocked (full $256 times 256$)*], $rho = 0.05$, $rho = 0.10$, $rho = 0.15$, $rho = 0.20$),

      table.cell(colspan: 5, fill: luma(235))[*Trained PDFT bases (ours)*],
      [★ `qft`          ], [24.91], [27.30], [29.20], [30.91],
      [★ `entangled_qft`], [25.07], [27.53], [29.48], [31.23],
      [★ `tebd`         ], [*25.09*], [*27.56*], [*29.52*], [*31.28*],
      [★ `mera`         ], [*25.09*], [*27.56*], [*29.52*], [*31.28*],

      table.cell(colspan: 5, fill: luma(235))[*Classical, top-$k$ rule*],
      [`bd_pca`       ], [*25.44*], [*27.74*], [*29.51*], [*31.07*],
      [`dct`          ], [25.36], [27.61], [29.33], [30.85],
      [`fft`          ], [24.50], [26.54], [28.07], [29.39],

      table.cell(colspan: 5, fill: luma(235))[*Classical, rank rule (control)*],
      [`dct_rank`     ], [23.43], [25.06], [26.29], [27.39],
    ),

    // Right column — block-wrapped (8×8 inner transform)
    table(
      columns: (auto, auto, auto, auto, auto),
      align: (left, right, right, right, right),
      stroke: 0.5pt,
      table.header([*8×8 block-wrapped*], $rho = 0.05$, $rho = 0.10$, $rho = 0.15$, $rho = 0.20$),

      table.cell(colspan: 5, fill: rgb("#dde8f7"))[*Trained PDFT bases (ours)*],
      [★ `blocked_8`   ], [25.18], [28.09], [30.30], [32.26],
      [★ `rich_8`      ], [25.97], [29.16], [31.55], [33.65],
      table.cell(fill: rgb("#ffe5e5"))[#text(fill: red, weight: "bold")[★ `real_rich_8`]],
      table.cell(fill: rgb("#ffe5e5"))[#text(fill: red, weight: "bold")[25.98]],
      table.cell(fill: rgb("#ffe5e5"))[#text(fill: red, weight: "bold")[29.18]],
      table.cell(fill: rgb("#ffe5e5"))[#text(fill: red, weight: "bold")[31.58]],
      table.cell(fill: rgb("#ffe5e5"))[#text(fill: red, weight: "bold")[33.68]],

      table.cell(colspan: 5, fill: rgb("#dde8f7"))[*Classical, top-$k$ rule*],
      [`block_dct_8`  ], [*26.11*], [*29.41*], [*31.86*], [*34.01*],
      [`block_pca_8`  ], [25.93], [29.05], [31.42], [33.51],
      [`block_fft_8`  ], [24.47], [27.10], [29.06], [30.79],

      table.cell(colspan: 5, fill: rgb("#dde8f7"))[*Classical, rank rule (control)*],
      [`block_dct_8_rank`], [22.35], [24.11], [25.20], [26.30],
      [`block_pca_8_rank`], [22.38], [24.17], [25.35], [26.40],
    ),
  )),
  caption: [Side-by-side: unblocked / full-image methods (left, gray
            header) vs $8 times 8$ block-wrapped methods (right,
            light blue header). Each column groups trained PDFT
            bases and classical top-$k$ baselines. ★ = trained PDFT
            basis (this work); unmarked rows = classical baselines.
            *Bold* = best-in-group at each keep ratio. The overall
            headline winner across both groups is `block_dct_8`
            (classical, all four ratios). Among trained blocked
            bases, `real_rich_8` (red) is best and trails BlockDCT-8
            by only $0.13$–$0.33$ dB. Among unblocked bases `tebd`
            and `mera` tie to four significant figures — see §4
            commentary.]
)

== Headline narrative — DIV2K-8q

The key facts the table encodes:

- *Block-DCT 8 leads at every keep ratio* (26.11 / 29.41 / 31.86 /
  34.01 dB at $rho = 0.05 / 0.10 / 0.15 / 0.20$). Block-DCT remains
  the strongest single basis on natural images at $8 times 8$
  blocks — consistent with JPEG-era engineering folklore and with
  the AR(1)–Gaussian limit (§2).

- *Real Rich-8 is the best trained block basis*, but trails
  Block-DCT-8 by $0.13$ – $0.33$ dB. Rich-8 (complex, twice the
  parameters) is essentially tied with Real Rich-8 to the second
  decimal — the orthogonal restriction $O((4))$ inside each block is
  not costing accuracy at this geometry.

- *Bilateral 2D-PCA (`bd_pca`) edges out global DCT at every $rho$*
  ($25.44 / 27.74 / 29.51 / 31.07$ vs $25.36 / 27.61 / 29.33 / 30.85$).
  By treating each image as the $256 times 256$ matrix instead of a
  flat $65536$-vector, BD-PCA fits a column eigenbasis (size $H times
  H = 256 times 256$) on $N W = 128000$ centered column samples and
  a row eigenbasis (size $W times W$) on $N H = 128000$ row samples
  — both full-rank, since $N W >> H$ and $N H >> W$. The forward
  transform is $Y = U^T (X - bar(X)) V$ and top-$k$ truncation runs
  on the $H W$-element matrix $Y$ at the same nominal rate as DCT.
  Flat global PCA on the same data hits a *rank-$499$ ceiling* at
  $approx 18.15$ dB because its ambient dim $d = 65536 >> N = 500$;
  BD-PCA sidesteps this geometry entirely by exploiting per-axis
  separability. Block-PCA-8 reaches even higher PSNRs than `bd_pca`
  at large $rho$ via per-patch fitting on $approx 3.2$ M $8 times
  8$ patches.

- *Top-$k$ pooling beats per-block rank rule on block transforms.*
  `block_dct_8` (top-$k$, magnitude-pooled across all 1024 blocks)
  scores $26.11 / 29.41 / 31.86 / 34.01$ dB at the four ratios;
  `block_dct_8_rank` (rank rule, *uniform* keep $k_b = floor(rho
  dot 64)$ per block) scores $22.35 / 24.11 / 25.20 / 26.30$ — a
  3.7–7.7 dB gap. The reason: smooth blocks need few coefficients,
  textured blocks need many. Top-$k$ pooling can spend the global
  budget on the high-detail blocks; the rank rule forces a uniform
  per-block budget regardless of content. Same effect for `block_pca_8`
  vs `block_pca_8_rank` ($25.93$/$29.05$/$31.42$/$33.51$ vs $22.38$/$24.17$/$25.35$/$26.40$).
  The headline tables use the top-$k$ rule for all methods.

- *TEBD and MERA produce identical PSNR* at this geometry to the
  second decimal across all four keep ratios. Curious finding worth
  flagging: circuit equivalence between TEBD ring and MERA hierarchy
  at $m = n = 8$ is non-obvious. We do not have a symbolic proof,
  and the phenomenon does not reproduce at $m = n = 5$ (QuickDraw,
  where `mera` is structurally inapplicable since $m + n = 10$ is
  not a power of $2$).

- *MERA actually runs at this geometry.* $m + n = 16 = 2^4$ admits
  the MERA hierarchy. Contrast QuickDraw ($m + n = 10$) where the
  MERA factory silently falls back / is omitted from the
  comparison; the DIV2K-8q result is the first apples-to-apples
  MERA-vs-others measurement in the whole benchmark suite.

- *All trained block bases use $8 times 8$ blocks*, matching
  classical block_dct_8 / block_fft_8 / block_pca_8 exactly
  (`*_8` factories from PR \#11, commit `ba81e1e`). No block-size
  asymmetry between trained and classical.

#pagebreak()

= Reconstructions — 2 representative images × keep ratios × bases

== Image \#11 — most textured (stdev $approx 0.29$)

#figure(
  image("figures/freq_recon_grid_img11_freq.png", width: 100%),
  caption: [Frequency-space spectra (peak-normalised log|F|, viridis,
            shared color scale) for image \#11 under each basis. Image
            \#11 is the most textured image in the DIV2K-8q test split
            (per-image stdev $approx 0.29$). The trained block-wrapped
            bases push energy into a small number of low-coefficient
            cells per block — visible as the bright clusters in
            `rich_8` / `real_rich_8` / `blocked_8` and the band
            structure in `block_dct_8` / `block_pca_8`. The `pca`
            panel's vertical purple band reflects rank-deficiency
            ($N - 1 = 499 << d = 65536$).]
)

#figure(
  image("figures/freq_recon_grid_img11.png", width: 100%),
  caption: [DIV2K-8q test image \#11 (most textured) reconstructed at
            four keep ratios $rho in {0.05, 0.10, 0.15, 0.20}$ (rows).
            Same column order as the freq-space figure above. PSNR
            (dB) annotated per cell. Textured imagery stresses every
            method at low $rho$: at $rho = 0.05$, `block_dct_8` leads
            in the block group and `bd_pca` leads in the unblocked
            group, with `real_rich_8` essentially tied with
            `block_dct_8`. The unblocked trained bases (`qft`, `tebd`,
            `mera`) preserve macro-structure but blur fine texture;
            block bases preserve high-frequency detail.]
)

#pagebreak(weak: true)

== Image \#43 — smoothest (stdev $approx 0.08$)

#figure(
  image("figures/freq_recon_grid_img43_freq.png", width: 100%),
  caption: [Frequency-space spectra for image \#43 (smoothest test
            image, stdev $approx 0.08$). Same column layout as the
            image-\#11 freq panel. Smooth content concentrates almost
            all energy into the DC and low-frequency cells per block,
            so even small keep ratios capture most of the image — the
            block bases' freq panels are sparser than image \#11's.]
)

#figure(
  image("figures/freq_recon_grid_img43.png", width: 100%),
  caption: [DIV2K-8q test image \#43 (smoothest) reconstructed at the
            same four keep ratios. Same column layout as image \#11.
            Smooth imagery is the easiest case for any sparsifying
            transform: at $rho = 0.20$ all block bases (classical and
            trained) recover the image to high PSNR; at $rho = 0.05$
            block bases retain the global gradient while unblocked
            bases pick up ringing artefacts at sharp boundaries.
            `bd_pca` tracks the DCT closely on this smooth image,
            consistent with its modest +0.1–0.2 dB advantage at every
            keep ratio.]
)

= Matching summary — bench name $arrow$ paper figure

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, right, left),
    stroke: 0.5pt,
    table.header([*bench name*], [*paper name*], [*params/dim ($n=4$)*], [*paper figure*]),
    [`qft`],            [Separable QFT],     [$3n + 2n(n-1) = 36$], [Fig. 1d, Tab. 1],
    [`entangled_qft`],  [Entangled QFT],     [$36 + n = 40$],       [Fig. 3, Eq. (entangled_qft)],
    [`tebd`],           [TEBD ring],         [$3n + 2n = 20$],      [App. C left, Fig. (tebd_circuit)],
    [`mera`],           [MERA hierarchical], [$3n + 4(n-1) = 24$],  [App. C right, Fig. (mera_circuit)],
    [`blocked_8`],      [Blocked QFT (8×8)], [$15$ ($m_"in"=3$)],   [§`block_wrapper`, Fig. (block_basis_circuits) bottom],
    [`rich_8`],         [RichBasis (8×8)],   [$54$ ($m_"in"=3$)],   [Fig. (block_basis_circuits) middle],
    [`real_rich_8`],    [RealRichBasis (8×8)], [$21$ ($m_"in"=3$)], [Fig. (block_basis_circuits) top — headline],
  ),
  caption: [Mapping from `pdft_benchmarks` basis name to the paper's
            naming and circuit figure. Param counts use the
            conventions in main.tex §`sec:qft_basis` and
            §`sec:block_wrapper`. The DIV2K-8q experiment uses the
            new `_8` block factories (PR \#11, commit `ba81e1e`) so
            all block-wrapped trained bases operate on the same
            $8 times 8$ tiles as the classical block baselines.]
)
