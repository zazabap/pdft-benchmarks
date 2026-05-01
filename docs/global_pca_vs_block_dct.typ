#set page(flipped: true, margin: (x: 1.5cm, y: 1.5cm))
#set text(size: 11pt, font: "New Computer Modern")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

#import "@preview/quill:0.7.1": *

#align(center)[
  #text(size: 16pt, weight: "bold")[QuickDraw — trained bases vs classical baselines]
]

#v(0.5em)

= Setup

$ m = n = 5, quad d = 2^m dot 2^n = 32 times 32 = 1024 $
$ N_"train" = 500, quad N_"test" = 50, quad "seed" = 42 $
$ "keep ratio" rho in {0.05, 0.10, 0.15, 0.20} $

= Pipeline (per image $x in [0,1]^d$, basis $Phi$)

$ y = Phi x quad ("forward") $
$ k = floor(rho dot d), quad y'_i = cases(y_i &"if" |y_i| in "top-"k, 0 &"else") $
$ hat(x) = "clip"(Phi^(-1) y', 0, 1), quad "MSE" = 1/d norm(x - hat(x))_2^2, quad "PSNR" = 10 log_10 (1\/"MSE") $

For block transforms: same pipeline applied independently to each
$8 times 8$ tile, top-$k$ pooled across all blocks.

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

    [`pca` (global)],
    [$Phi = V^T$, where \ $Sigma = V Lambda V^T$, \ $Sigma = 1/(N-1) sum_(i=1)^N (x_i - mu)(x_i - mu)^T$ \ fit on $N = 500$ images, $d = 1024$],
    [$d times d = 1024 times 1024$ \ rank $<= N - 1 = 499$ (deficient)],
    [$Phi (x - mu)$],
    [top-$k$: $y_i dot bb(1)[|y_i| >= tau_k]$, $k = floor(rho dot d)$, $rho in {.05,.10,.15,.20}$ \ (rank: keep $i = 0, dots, k-1$)],
    [$Phi^T y' + mu$],

    [`block_pca_8`],
    [$Phi = V_b^T$, where \ $Sigma_b = V_b Lambda_b V_b^T$, \ $Sigma_b = 1/(N_p-1) sum_(j=1)^(N_p) (p_j - mu_b)(p_j - mu_b)^T$ \ fit on $N_p = 8000$ patches],
    [$64 times 64$ per block, full rank \ shared across all $4 times 4$ blocks],
    [per block: $Phi (p - mu_b)$],
    [top-$k$ pooled across all blocks, keep $k = floor(rho dot 1024)$ largest \ (rank: per-block keep $i = 0, dots, k_b - 1$, $k_b = floor(rho dot 64)$)],
    [per block: $Phi^T y' + mu_b$, then re-tile],

    [`dct` (global)],
    [closed-form (DCT-II): \ $Phi[k, n] = alpha_k cos(pi (n + 1\/2) k \/ N)$ \ $alpha_0 = sqrt(1/N)$, $alpha_(k >= 1) = sqrt(2/N)$ \ $N = 32$, no fit, no mean],
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
            frequency $k = 0, 1, 2, dots$). See the Results table for
            the $4$ to $9$ dB cost of switching rule on the same
            fitted basis.]
)

== Why DCT $approx$ block PCA in the limit

$ Phi_("block_pca_8") = "KLT"(Sigma_b) quad arrow.long quad
   Phi_("block_dct_8") = "KLT"(lim_(rho arrow.r 1) Sigma_("AR(1)")(rho)) $

DCT $=$ what KLT becomes if the patch covariance is the analytic
AR(1)–Gaussian limit. Empirically the two differ by only $0.68$ dB on
QuickDraw — natural patches are nearly AR(1)–Gaussian.


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
      table.header([*unblocked (full $32 times 32$)*], $rho = 0.05$, $rho = 0.10$, $rho = 0.15$, $rho = 0.20$),

      table.cell(colspan: 5, fill: luma(235))[*Trained PDFT bases (ours)*],
      [★ `qft`          ], [*16.72*], [*19.58*], [*22.06*], [*24.35*],
      [★ `entangled_qft`], [16.72], [19.58], [22.05], [24.35],
      [★ `tebd`         ], [16.64], [19.40], [21.79], [24.01],

      table.cell(colspan: 5, fill: luma(235))[*Classical, top-$k$ rule*],
      [`pca`          ], [*17.78*], [*20.69*], [*23.12*], [*25.30*],
      [`dct`          ], [16.13], [18.44], [20.33], [22.03],
      [`fft`          ], [15.26], [17.32], [19.01], [20.56],

      table.cell(colspan: 5, fill: luma(235))[*Classical, rank rule (control)*],
      [`pca_rank`         ], [*15.56*], [*17.57*], [*19.38*], [*20.99*],
      [`dct_rank`         ], [13.84], [15.24], [16.49], [17.63],
    ),

    // Right column — block-wrapped (8×8 inner transform)
    table(
      columns: (auto, auto, auto, auto, auto),
      align: (left, right, right, right, right),
      stroke: 0.5pt,
      table.header([*8×8 block-wrapped*], $rho = 0.05$, $rho = 0.10$, $rho = 0.15$, $rho = 0.20$),

      table.cell(colspan: 5, fill: rgb("#dde8f7"))[*Trained PDFT bases (ours)*],
      table.cell(fill: rgb("#ffe5e5"))[#text(fill: red, weight: "bold")[★ `rich`]],
      table.cell(fill: rgb("#ffe5e5"))[#text(fill: red, weight: "bold")[18.82]],
      table.cell(fill: rgb("#ffe5e5"))[#text(fill: red, weight: "bold")[23.71]],
      table.cell(fill: rgb("#ffe5e5"))[#text(fill: red, weight: "bold")[28.15]],
      table.cell(fill: rgb("#ffe5e5"))[#text(fill: red, weight: "bold")[32.60]],
      [★ `real_rich`    ], [18.79], [23.68], [28.05], [32.37],
      [★ `blocked`      ], [18.12], [22.41], [26.20], [30.06],

      table.cell(colspan: 5, fill: rgb("#dde8f7"))[*Classical, top-$k$ rule*],
      [`block_dct_8`  ], [*17.20*], [*20.70*], [*23.72*], [*26.63*],
      [`block_pca_8`  ], [16.95], [20.46], [23.34], [25.95],
      [`block_fft_8`  ], [15.97], [18.74], [21.09], [23.34],

      table.cell(colspan: 5, fill: rgb("#dde8f7"))[*Classical, rank rule (control)*],
      [`block_pca_8_rank` ], [*13.60*], [*14.95*], [*16.12*], [*17.16*],
      [`block_dct_8_rank` ], [13.50], [14.81], [15.78], [16.97],
    ),
  )),
  caption: [Side-by-side: unblocked / full-image methods (left, gray
            header) vs $8 times 8$ block-wrapped methods (right,
            light blue header). Each column groups trained PDFT
            bases, classical top-$k$ baselines, and the rank-rule
            control. ★ = trained PDFT basis (this work); unmarked rows
            = classical baselines. *Bold* = best-in-group at each
            keep ratio. The overall headline winner is `rich` ★
            (block-wrapped, all four ratios). `mera` omitted
            (incompatible at $m + n = 10$).]
)

== Tensor networks of the trained bases at $m = n = 5$

10-qubit layout (rows $x_1,..,x_5$ top, cols $y_1,..,y_5$ bottom).
Seven sub-circuits arranged in a 4-column grid for side-by-side
comparison. All separable except `entangled_qft`.

#let basis-cell(name, desc, circuit) = block[
  #align(center)[
    #text(size: 9pt, weight: "bold", fill: rgb("#0a3d8c"))[#raw(name)] #h(0.3em)
    #text(size: 8pt, fill: gray)[#desc]
  ]
  #v(0.3em)
  #align(center, circuit)
]

#figure(
  block(width: 100%, grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 0.6em,
    row-gutter: 1.4em,
    align: center + horizon,

    // Top row — three unblocked variants, all at the same scale and
    // row-spacing so 10-wire heights match exactly.
    basis-cell("qft", [$F_5 times.circle F_5$, 110p], quantum-circuit(
      scale: 55%, row-spacing: 0.55em, column-spacing: 0.3em,
      lstick($x_1$), $H$, ctrl(1), ctrl(2), ctrl(3), ctrl(4), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, [\ ],
      lstick($x_2$), 1,   $M$,     1,       1,       1,       $H$, ctrl(1), ctrl(2), ctrl(3), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($x_3$), 1,   1,       $M$,     1,       1,       1,   $M$,     1,       1,       $H$, ctrl(1), ctrl(2), 1, 1, 1, [\ ],
      lstick($x_4$), 1,   1,       1,       $M$,     1,       1,   1,       $M$,     1,       1,   $M$,     1,       $H$, ctrl(1), 1, [\ ],
      lstick($x_5$), 1,   1,       1,       1,       $M$,     1,   1,       1,       $M$,     1,   1,       $M$,     1,   $M$,     $H$, [\ ],
      lstick($y_1$), $H$, ctrl(1), ctrl(2), ctrl(3), ctrl(4), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, [\ ],
      lstick($y_2$), 1,   $M$,     1,       1,       1,       $H$, ctrl(1), ctrl(2), ctrl(3), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($y_3$), 1,   1,       $M$,     1,       1,       1,   $M$,     1,       1,       $H$, ctrl(1), ctrl(2), 1, 1, 1, [\ ],
      lstick($y_4$), 1,   1,       1,       $M$,     1,       1,   1,       $M$,     1,       1,   $M$,     1,       $H$, ctrl(1), 1, [\ ],
      lstick($y_5$), 1,   1,       1,       1,       $M$,     1,   1,       1,       $M$,     1,   1,       $M$,     1,   $M$,     $H$,
    )),

    basis-cell("entangled_qft", [QFT $+$ 5 $E_k$, 115p], quantum-circuit(
      scale: 55%, row-spacing: 0.55em, column-spacing: 0.22em,
      lstick($x_1$), $H$, ctrl(1), ctrl(2), ctrl(3), ctrl(4), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ctrl(5), 1, 1, 1, 1, [\ ],
      lstick($x_2$), 1,   $M$,     1,       1,       1,       $H$, ctrl(1), ctrl(2), ctrl(3), 1, 1, 1, 1, 1, 1, 1, ctrl(5), 1, 1, 1, [\ ],
      lstick($x_3$), 1,   1,       $M$,     1,       1,       1,   $M$,     1,       1,       $H$, ctrl(1), ctrl(2), 1, 1, 1, 1, 1, ctrl(5), 1, 1, [\ ],
      lstick($x_4$), 1,   1,       1,       $M$,     1,       1,   1,       $M$,     1,       1,   $M$,     1,       $H$, ctrl(1), 1, 1, 1, 1, ctrl(5), 1, [\ ],
      lstick($x_5$), 1,   1,       1,       1,       $M$,     1,   1,       1,       $M$,     1,   1,       $M$,     1,   $M$,     $H$, 1, 1, 1, 1, ctrl(5), [\ ],
      lstick($y_1$), $H$, ctrl(1), ctrl(2), ctrl(3), ctrl(4), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, $E$, 1, 1, 1, 1, [\ ],
      lstick($y_2$), 1,   $M$,     1,       1,       1,       $H$, ctrl(1), ctrl(2), ctrl(3), 1, 1, 1, 1, 1, 1, 1, $E$, 1, 1, 1, [\ ],
      lstick($y_3$), 1,   1,       $M$,     1,       1,       1,   $M$,     1,       1,       $H$, ctrl(1), ctrl(2), 1, 1, 1, 1, 1, $E$, 1, 1, [\ ],
      lstick($y_4$), 1,   1,       1,       $M$,     1,       1,   1,       $M$,     1,       1,   $M$,     1,       $H$, ctrl(1), 1, 1, 1, 1, $E$, 1, [\ ],
      lstick($y_5$), 1,   1,       1,       1,       $M$,     1,   1,       1,       $M$,     1,   1,       $M$,     1,   $M$,     $H$, 1, 1, 1, 1, $E$,
    )),

    basis-cell("tebd", [NN ring $+$ wrap, 50p], quantum-circuit(
      scale: 55%, row-spacing: 0.55em, column-spacing: 0.45em,
      lstick($x_1$), $H$, ctrl(1), 1, 1, 1, $T$, [\ ],
      lstick($x_2$), $H$, $T$,    ctrl(1), 1, 1, 1, [\ ],
      lstick($x_3$), $H$, 1,      $T$,    ctrl(1), 1, 1, [\ ],
      lstick($x_4$), $H$, 1,      1,      $T$,    ctrl(1), 1, [\ ],
      lstick($x_5$), $H$, 1,      1,      1,      $T$, ctrl(-4), [\ ],
      lstick($y_1$), $H$, ctrl(1), 1, 1, 1, $T$, [\ ],
      lstick($y_2$), $H$, $T$,    ctrl(1), 1, 1, 1, [\ ],
      lstick($y_3$), $H$, 1,      $T$,    ctrl(1), 1, 1, [\ ],
      lstick($y_4$), $H$, 1,      1,      $T$,    ctrl(1), 1, [\ ],
      lstick($y_5$), $H$, 1,      1,      1,      $T$, ctrl(-4),
    )),

    // Bottom row — three block-wrapped variants, all at the same
    // scale so heights match within the bottom row too.
    basis-cell("blocked", [$I_4 times.circle F_3$ per dim, 30p], quantum-circuit(
      scale: 80%, row-spacing: 0.55em, column-spacing: 0.45em,
      lstick($x_1$), gate($I$), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($x_2$), gate($I$), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($x_3$), 1, $H$, ctrl(1), ctrl(2), 1, 1, 1, [\ ],
      lstick($x_4$), 1, 1,   $M$,     1,       $H$, ctrl(1), 1, [\ ],
      lstick($x_5$), 1, 1,   1,       $M$,     1,   $M$,     $H$, [\ ],
      lstick($y_1$), gate($I$), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($y_2$), gate($I$), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($y_3$), 1, $H$, ctrl(1), ctrl(2), 1, 1, 1, [\ ],
      lstick($y_4$), 1, 1,   $M$,     1,       $H$, ctrl(1), 1, [\ ],
      lstick($y_5$), 1, 1,   1,       $M$,     1,   $M$,     $H$,
    )),

    basis-cell("rich", [$I_4 times.circle U^"inner"$, 108p], quantum-circuit(
      scale: 80%, row-spacing: 0.55em, column-spacing: 0.45em,
      lstick($x_1$), gate($I$), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($x_2$), gate($I$), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($x_3$), 1, $H$, mqgate($U^((4))$, n: 2), 1, mqgate($U^((4))$, n: 3), 1, 1, [\ ],
      lstick($x_4$), 1, 1, 1, 1, 1, $H$, mqgate($U^((4))$, n: 2), [\ ],
      lstick($x_5$), 1, 1, 1, 1, 1, 1, 1, 1, [\ ],
      lstick($y_1$), gate($I$), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($y_2$), gate($I$), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($y_3$), 1, $H$, mqgate($U^((4))$, n: 2), 1, mqgate($U^((4))$, n: 3), 1, 1, [\ ],
      lstick($y_4$), 1, 1, 1, 1, 1, $H$, mqgate($U^((4))$, n: 2), [\ ],
      lstick($y_5$), 1, 1, 1, 1, 1, 1, 1, 1,
    )),

    basis-cell("real_rich", [$I_4 times.circle O^"inner"$, 42p (HEADLINE)], quantum-circuit(
      scale: 80%, row-spacing: 0.55em, column-spacing: 0.45em,
      lstick($x_1$), gate($I$), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($x_2$), gate($I$), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($x_3$), 1, $H$, mqgate($O^((4))$, n: 2), 1, mqgate($O^((4))$, n: 3), 1, 1, [\ ],
      lstick($x_4$), 1, 1, 1, 1, 1, $H$, mqgate($O^((4))$, n: 2), [\ ],
      lstick($x_5$), 1, 1, 1, 1, 1, 1, 1, 1, [\ ],
      lstick($y_1$), gate($I$), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($y_2$), gate($I$), 1, 1, 1, 1, 1, 1, [\ ],
      lstick($y_3$), 1, $H$, mqgate($O^((4))$, n: 2), 1, mqgate($O^((4))$, n: 3), 1, 1, [\ ],
      lstick($y_4$), 1, 1, 1, 1, 1, $H$, mqgate($O^((4))$, n: 2), [\ ],
      lstick($y_5$), 1, 1, 1, 1, 1, 1, 1, 1,
    )),

  )),
  caption: [Six trained-basis topologies at QuickDraw geometry
            ($m = n = 5$, $10$ qubits per circuit). Top row —
            unblocked: `qft`, `entangled_qft`, `tebd` (all at the same
            scale and row-spacing so their 10-wire heights match
            exactly). Bottom row — block-wrapped: `blocked`, `rich`,
            `real_rich` (headline). `mera` is structurally
            inapplicable here ($m + n = 10 != 2^p$) and omitted.
            Param counts cover both row and col registers.]
)

== Geometry comparison at a glance

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, left, right, right),
    stroke: 0.5pt,
    table.header([*basis*], [*topology*], [*row–col coupling*], [*params/dim*], [*PSNR at $rho = 0.20$*]),
    [`qft`],           [all-to-all per dim],   [none], [55],  [24.35],
    [`entangled_qft`], [all-to-all + 5 cross], [yes ($E_k$)], [60], [24.35],
    [`tebd`],          [NN ring per dim],      [none], [25],  [24.01],
    [`blocked`],       [outer-$I$ + inner QFT-3],     [none], [15], [30.06],
    [`rich`],          [outer-$I$ + inner $U^((4))$], [none], [54], [32.60],
    [`real_rich`],     [outer-$I$ + inner $O^((4))$], [none], [21], [32.37],
  ),
  caption: [All bases at QuickDraw geometry $m = n = 5$. Param counts
            are per dim (multiply by 2 for full transform unless
            row/col are tied). PSNR from Table earlier in this doc.]
)

#pagebreak()

= Reconstructions — 2 representative images × keep ratios × bases

== Image #0 — cat sketch

#figure(
  image("figures/freq_recon_grid_img0_freq.png", width: 100%),
  caption: [Frequency-space spectra (peak-normalised log|F|, viridis,
            shared color scale) for image #0 under each basis. Same
            column order as the recon grid below. Note the trained
            block-wrapped bases (blue headers) push energy into a
            small number of low-coefficient cells per block — visible
            as the bright clusters in `rich`/`real_rich`/`blocked`
            and the band structure in `block_dct_8`/`block_pca_8`.
            The `pca` panel's vertical purple band reflects
            rank-deficiency ($N - 1 = 499 < d = 1024$).]
)

#figure(
  image("figures/freq_recon_grid_img0.png", width: 100%),
  caption: [QuickDraw test image #0 reconstructed at four keep ratios
            $rho in {0.05, 0.10, 0.15, 0.20}$ (rows). Same column
            order as the freq-space figure above. PSNR (dB) annotated
            bottom-right of each cell. At $rho = 0.20$: trained `rich`
            $approx 34$ dB vs `block_dct_8` $approx 28$ dB ($+6$ dB)
            vs `dct` $approx 22$ dB ($+12$ dB).]
)

#pagebreak(weak: true)

== Image #2 — fish silhouette

#figure(
  image("figures/freq_recon_grid_img2.png", width: 100%),
  caption: [QuickDraw test image #2 reconstructed at the same four
            keep ratios. Same column layout as image #1. This image is
            among the strongest cases for trained tensor-network
            bases: at $rho = 0.20$, trained `rich` $approx 44$ dB vs
            `block_dct_8` $approx 32$ dB ($+12$ dB) vs `dct`
            $approx 25$ dB ($+19$ dB). Even at $rho = 0.05$ (top row)
            `rich` and `real_rich` already preserve the recognisable
            fish-stroke topology while `dct` and `fft` collapse to a
            blurred blob.]
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
    [`blocked`],        [Blocked QFT],       [$15$ ($m_"in"=3$)],   [§`block_wrapper`, Fig. (block_basis_circuits) bottom],
    [`rich`],           [RichBasis],         [$54$ ($m_"in"=3$)],   [Fig. (block_basis_circuits) middle],
    [`real_rich`],      [RealRichBasis],     [$21$ ($m_"in"=3$)],   [Fig. (block_basis_circuits) top — headline],
  ),
  caption: [Mapping from `pdft_benchmarks` basis name to the paper's
            naming and circuit figure. Param counts use the conventions
            in main.tex §`sec:qft_basis` and §`sec:block_wrapper`.]
)
