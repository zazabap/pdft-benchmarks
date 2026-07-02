#set page(paper: "us-letter", margin: (x: 0.9in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#show heading.where(level: 1): set block(above: 0.9em, below: 0.4em)
#show raw: set text(size: 8.5pt)

#let ss = json("disturbance_sweep.json")
#let ref = json("reference/classical_dct4.json")
#let f2(x) = str(calc.round(x, digits: 2))
#let f1(x) = str(calc.round(x, digits: 1))
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
`CRY` twiddle is applied to the control $=$ 1 half only), we perturb a
random fraction $f$ of its *2200 real gate-tensor entries* with on-manifold
Gaussian jitter: add $N(0, #f2(sig))$ to the selected entries and re-project
each touched gate onto its manifold (nearest-orthogonal O(4)/O(2) via SVD; the
$Delta$-sign gate by a phase jitter), so the perturbed init stays a valid
real-orthogonal DCT-IV and its untrained PSNR is interpretable. The disturbed
counts are $round(f dot 2200) = {2, 4, 11, 22, 44, 110, 220, 440, 1100, 2200}$
entries for $f in {0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100}%$; at $f = 100%$
*every* gate is jittered.

Each perturbed init (#nseed seeds per $f$) is trained for #ss.epochs steps of
top-#topk_pct% MSE on the full 500-image DIV2K-8q pool (mini-batch 50, a *fixed*
shuffle seed so only the perturbation varies), then scored for test PSNR at four
keep ratios $rho$ on the fixed canonical 50-image test set. $f = 0$ is the
undisturbed exact-init reference (its untrained PSNR reproduces the classical
DCT-IV: #f2(ref.canonical_dct4.psnr.at("0.2")) dB @ $rho{=}.20$).

= Disturbance procedure

The jitter treats the exact DCT-IV's 214 gate tensors as one flat vector of 2200
real entries and acts *per gate*. For disturbance rate $f$ and jitter scale
$sigma$:

```
select round(f * 2200) of the 2200 real gate entries, uniform, no replacement
for each gate G that owns >= 1 selected entry:
    add  N(0, sigma)  to G's selected entries only            # Gaussian jitter
    re-project the noised gate back onto its manifold:
        (2,2,2,2) mirror-U4    ->  nearest orthogonal, SVD polar U V^T   [O(4)]
        (2,2) Delta-sign gate  ->  phi <- pi + sigma*z ; controlled_phase_diag(phi)
        (2,2) rotation / H     ->  nearest orthogonal, SVD polar U V^T   [O(2)]
gates with no selected entry are copied unchanged             # f = 0 = identity
```

The noise is plain additive i.i.d. Gaussian $N(0, sigma)$ on the *selected raw
entries only*; the per-gate *re-projection* is what turns it into an on-manifold
perturbation. Each gate is real-orthogonal, so the nearest valid gate to the
noised one is its SVD polar factor $U V^T$ (drop the singular values) — this
keeps the operator a genuine real-orthogonal DCT-IV, so the perturbed init's
untrained PSNR is meaningful rather than an artefact of a non-orthogonal matrix.
The $Delta$-sign gate is not a rotation but a phase stub (its lower-right entry
is $e^(i phi)$, $phi = pi$ at init), so it is jittered in its phase $phi$ instead.
Because a gate moves only when one of its entries is selected, small $f$ nudges a
few gates while $f = 100%$ jitters *every* gate; the *magnitude* each touched gate
travels is set by $sigma$, not by $f$.

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

= Initialisation loss

#let il0 = ss.baseline.init_loss
#let il100 = ss.agg_init_loss.at("1").mean
#let fl100 = ss.agg_final_loss.at("1").mean

The same recovery shows up in loss space. The perturbed init's top-$k$ MSE loss
on the 500-image train pool — the value the optimiser sees at step 0 — rises with
the disturbance rate, from #f2(il0) at the exact init to #f2(il100) at
$f = 100%$ (a #f1(il100 / il0)#sym.times increase): flat below #sym.tilde 1%, then
climbing steeply, with a seed spread that widens as more gates are hit. After
#ss.epochs steps every perturbed init converges back to essentially the
exact-init training loss (#f2(fl100) at $f = 100%$ vs #f2(ss.baseline.final_loss)
at $f = 0$) — training erases the initialisation deficit in the loss just as it
does in test PSNR.

#figure(
  image("figures/disturbance_init_loss.svg", width: 72%),
  caption: [Top-$k$ MSE loss on the 500-image train pool vs. disturbance rate
  (log-x, linear y): at the perturbed *init* (rising, vermilion) and *after*
  #ss.epochs training steps (flat, green), mean $plus.minus sigma$ over #nseed
  seeds; the dotted line marks the exact-init loss. Training returns every
  perturbed init to the exact-init loss.],
)

= Reading

#let exact_un = ref.canonical_dct4.psnr.at("0.2")
#let un_lo = ss.agg_untrained.at("0.001").at("0.2").mean
#let un_100 = ss.agg_untrained.at("1").at("0.2").mean
#let tr_100 = ss.agg_trained.at("1").at("0.2").mean

At $rho{=}.20$ the undisturbed exact init trains to #f2(base20) dB, and *no amount
of disturbance moves that endpoint*: across the entire range — from 0.1% up to
*100%* of the 2200 gate entries jittered ($sigma = #f2(sig)$), i.e. every gate
perturbed — trained PSNR stays 33.2#sym.dash.en 33.4 dB at $rho{=}.20$ and is
equally flat at $rho{=}.01\/.05\/.10$, never leaving the undisturbed reference by
more than #sym.tilde 0.2 dB (within seed scatter). The perturbed *init*, by
contrast, collapses monotonically — from #f2(un_lo) dB at 0.1% to #f2(un_100) dB
at 100%, an #f2(exact_un - un_100) dB fall below the exact init (#f2(exact_un) dB)
that leaves the untrained operator barely above the $rho{=}.20$ floor — yet
#ss.epochs steps of top-#topk_pct% training recover it to #f2(tr_100) dB, the
exact-init endpoint.

So the exact DCT-IV init sits in a *very wide, flat basin*: a per-gate jitter of
$sigma = #f2(sig)$ applied to *all* its parameters still drains to the same
trained optimum. What the trained model depends on is the DCT-IV *topology* plus
the top-$k$ objective, not the precise gate values — provided each gate is only
locally perturbed. The lever that would break recovery is thus the jitter
*magnitude* $sigma$ (fixed here), not the disturbed *fraction*: contrast the
random-init seed study, where fully Haar-random gates (an $O(1)$, not
$sigma {=} #f2(sig)$, per-gate displacement) settle into a lower basin below the
exact transform.
