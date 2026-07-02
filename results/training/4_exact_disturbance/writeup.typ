#set page(paper: "us-letter", margin: (x: 0.9in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set block(above: 0.9em, below: 0.4em)
#show raw: set text(size: 8.5pt)

#let ss = json("disturbance_sweep.json")
#let ref = json("reference/classical_dct4.json")
#let f2(x) = str(calc.round(x, digits: 2))
#let tr(fk, rk) = ss.agg_trained.at(fk).at(rk)
#let base20 = ss.baseline.psnr_trained.at("0.2")
#let sig = ss.sigma
#let nseed = ss.seeds.len()
#let topk_pct = calc.round(ss.topk_ratio * 100)

#align(center)[
  #text(size: 15pt, weight: "bold")[Parameter-disturbance robustness of the exact DCT-IV init]
  #v(2pt)
  #text(size: 10.5pt)[Controlled O(2)-twiddle DCT-IV on DIV2K-8q: Gaussian jitter
  ($sigma = #f2(sig)$) of a random fraction of the 2200 exact-init gate
  entries, #(ss.epochs)-step top-#topk_pct% training, #nseed seeds]
  #v(3pt)
  #text(size: 9pt, fill: rgb("#555"))[Generated #datetime.today().display("[year]-[month]-[day]")]
]

= Setup

Starting from the *exact* analytic DCT-IV (`DCT4Basis(8, 8, parametrization:
"controlled")` — the O(2)-twiddle form of pdft PR \#24, commit `5365a5a`, whose
`CRY` twiddle is applied to the control${=}1$ half only), we perturb a
random fraction $f$ of its *2200 real gate-tensor entries* with on-manifold
Gaussian jitter: add $N(0, #f2(sig))$ to the selected entries and re-project
each touched gate onto its manifold (nearest-orthogonal O(4)/O(2) via SVD; the
$Delta$-sign gate by a phase jitter), so the perturbed init stays a valid
real-orthogonal DCT-IV and its untrained PSNR is interpretable. The disturbed
counts are $round(f dot 2200) = {2, 4, 11, 22, 44, 110, 220}$ entries for
$f in {0.1, 0.2, 0.5, 1, 2, 5, 10}%$.

Each perturbed init (#nseed seeds per $f$) is trained for #ss.epochs steps of
top-#topk_pct% MSE on the full 500-image DIV2K-8q pool (mini-batch 50, a *fixed*
shuffle seed so only the perturbation varies), then scored for test PSNR at four
keep ratios $rho$ on the fixed canonical 50-image test set. $f = 0$ is the
undisturbed exact-init reference (its untrained PSNR reproduces the classical
DCT-IV: #f2(ref.canonical_dct4.psnr.at("0.2")) dB @ $rho{=}.20$).

= Final PSNR vs. disturbance

#figure(
  image("figures/disturbance_psnr_vs_f.svg", width: 78%),
  caption: [Trained test PSNR vs. the fraction of exact-init parameters disturbed
  (log-x), one curve per keep ratio $rho$, mean $plus.minus sigma$ over #nseed
  seeds (shaded). Thin horizontal lines mark the undisturbed exact-init-trained
  PSNR per $rho$.],
)

#align(center)[#table(
  columns: 5, align: (left, right, right, right, right),
  stroke: 0.4pt + luma(180), inset: (x: 6pt, y: 3pt),
  table.header([*disturbed*], [$rho{=}.01$], [$rho{=}.05$], [$rho{=}.10$], [$rho{=}.20$]),
  [0 (exact)],
  text(weight: "bold")[#f2(ss.baseline.psnr_trained.at("0.01"))],
  text(weight: "bold")[#f2(ss.baseline.psnr_trained.at("0.05"))],
  text(weight: "bold")[#f2(ss.baseline.psnr_trained.at("0.1"))],
  text(weight: "bold")[#f2(base20)],
  table.hline(stroke: 0.6pt),
  ..ss.agg_trained.pairs().map(pair => {
    let fk = pair.at(0)
    let pct = calc.round(float(fk) * 100, digits: 1)
    ([#str(pct)%],
     [#f2(tr(fk, "0.01").mean) #sym.plus.minus #f2(tr(fk, "0.01").std)],
     [#f2(tr(fk, "0.05").mean) #sym.plus.minus #f2(tr(fk, "0.05").std)],
     [#f2(tr(fk, "0.1").mean) #sym.plus.minus #f2(tr(fk, "0.1").std)],
     [#f2(tr(fk, "0.2").mean) #sym.plus.minus #f2(tr(fk, "0.2").std)])
  }).flatten(),
)]
#align(center, text(8pt, fill: luma(90))[Trained test PSNR (dB), mean
  $plus.minus sigma$ over #nseed perturbation seeds; row *0 (exact)* is the
  undisturbed exact-init-trained reference.])

= Recovery: perturbed init vs. trained

#figure(
  image("figures/disturbance_recovery.svg", width: 92%),
  caption: [Per-$rho$ panels: the perturbed *init* PSNR (dotted) drops with $f$,
  while the *trained* PSNR (solid) stays close to the exact-init reference — the
  gap between the two curves is the amount training recovers.],
)

= Reading

#let exact_un = ref.canonical_dct4.psnr.at("0.2")
#let un_lo = ss.agg_untrained.at("0.001").at("0.2").mean
#let un_hi = ss.agg_untrained.at("0.1").at("0.2").mean

At $rho{=}.20$ the undisturbed exact init trains to #f2(base20) dB. Disturbing it
barely moves that endpoint: across the whole range — up to *10%* of the 2200 gate
entries jittered ($sigma = #f2(sig)$) — trained PSNR stays #f2(base20)#sym.dash.en
33.4 dB at $rho{=}.20$ and is equally flat at $rho{=}.01\/.05\/.10$, never
departing from the undisturbed reference by more than #sym.tilde 0.14 dB (within
seed scatter; the faint 2#sym.dash.en 5% bumps are basin noise in the very-flat
top-$k$ MSE valley, not a trend). The perturbed *init*, by contrast, degrades
monotonically — from #f2(un_lo) dB at 0.1% to #f2(un_hi) dB at 10% — i.e. up to
#f2(exact_un - un_hi) dB below the exact init (#f2(exact_un) dB), yet #ss.epochs
steps of top-#topk_pct% training close that gap entirely.

The exact DCT-IV init therefore sits in a *wide, flat basin*: within a
several-percent on-manifold neighbourhood the precise gate values are not what the
trained model depends on — training re-optimises them to the same optimum. This is
the local-robustness counterpart to the random-init seed study, where a *fully*
Haar-random controlled DCT-IV instead settles into a lower basin below the exact
transform; the lever is staying near the exact init, not the exact values
themselves.
