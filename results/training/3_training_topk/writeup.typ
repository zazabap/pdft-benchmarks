#set page(paper: "us-letter", margin: (x: 0.7in, y: 0.7in))
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em, first-line-indent: 0pt)
#show heading.where(level: 1): set block(above: 1.0em, below: 0.5em)
#show heading.where(level: 2): set block(above: 0.8em, below: 0.4em)
#show raw: set text(size: 8.5pt)

= Does the training top-k matter? Sweeping the QFT compression rate

*Question.* Every trained QFT cell in this repository optimises
`MSELoss(k)` — mean-squared reconstruction error after keeping the
top-$k$ transform coefficients by magnitude — with $k$ fixed at
$10%$ of the $2^(m+n)$ coefficients
($k = "round"(2^(m+n) dot 0.1)$). Evaluation, by contrast, reports
PSNR at four keep-ratios $rho in {0.05, 0.10, 0.15, 0.20}$. The
training objective therefore optimises one compression rate while we
deploy at several. Two questions follow: (i) is the fixed $10%$
training top-$k$ actually optimal, and (ii) is each eval rate $rho$
best served by a QFT *trained at a matching* top-$k$ ($k approx rho$)?

*Method.* We train the analytic-init `QFTBasis(m, n)` at four
training top-$k$ ratios $r in {0.05, 0.10, 0.15, 0.20}$ and evaluate
every trained operator at all four eval keep-ratios — the full
$4 times 4$ train-$k$ $times$ eval-$rho$ matrix. Each run is identical
to the headline `qft` cell except for the training top-$k$: analytic
QFT init, the `generalized` preset, $1008$ steps (`--epochs 112`, no
early stopping), seed $42$. Two datasets: DIV2K-8q ($m = n = 8$,
$256 times 256$) and QuickDraw-5q ($m = n = 5$, $32 times 32$).

*Sanity check.* The $r = 0.10$ DIV2K run reproduces the headline `qft`
cell *exactly* — $25.093$, $27.572$, $29.532$, $31.294$ dB at
$rho = 0.05, 0.10, 0.15, 0.20$, $Delta = 0.000$ dB at every
ratio — confirming the sweep faithfully replicates the headline
training pipeline and that the headline number was trained at $10%$.

== DIV2K-8q ($m = n = 8$)

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*train top-k*], $rho = 0.05$, $rho = 0.10$, $rho = 0.15$, $rho = 0.20$,
  ),
  [$r = 0.05$], [24.912], [27.304], [29.201], [30.906],
  [$r = 0.10$ (default)], [25.093], [27.572], [29.532], [31.294],
  [$r = 0.15$], [24.992], [27.456], [29.405], [31.151],
  [$r = 0.20$], [*25.235*], [*27.811*], [*29.841*], [*31.659*],
)

On DIV2K the *highest* training top-$k$ wins at every eval rate:
$r = 0.20$ beats the default $r = 0.10$ by $+0.14$, $+0.24$, $+0.31$,
$+0.37$ dB. The matching rule (ii) is wrong — a single training
top-$k$ ($0.20$) dominates the whole column set, including the
low-rate columns. Notably, $r = 0.20$ from the analytic init reaches
$31.659$ dB at $rho = 0.20$, essentially the same flat-valley floor
($approx 31.66$ dB) that identity-init QFT reaches at $r = 0.10$ — two
routes to the same endpoint.

#figure(
  image("figures/topk_div2k_8q.svg", width: 100%),
  caption: [DIV2K-8q. Left: test PSNR vs training top-$k$, one line
    per eval keep-ratio; the ringed point on each line is the
    *matching* run ($r = rho$). The rings are mostly *not* the peaks —
    each line keeps climbing past its matching point up to $r = 0.20$.
    Right: the full train-$k$ $times$ eval-$rho$ matrix; the
    best training top-$k$ per eval column is boxed in red (the entire
    $r = 0.20$ row). The small dip at $r = 0.15$ is within single-seed
    basin noise (see *Caveats*).]
)

== QuickDraw-5q ($m = n = 5$)

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  stroke: 0.5pt,
  table.header(
    [*train top-k*], $rho = 0.05$, $rho = 0.10$, $rho = 0.15$, $rho = 0.20$,
  ),
  [$r = 0.05$], [*16.198*], [*18.927*], [*21.268*], [23.443],
  [$r = 0.10$ (default)], [16.196], [18.921], [21.267], [*23.447*],
  [$r = 0.15$], [15.801], [18.372], [20.560], [22.567],
  [$r = 0.20$], [15.800], [18.370], [20.557], [22.568],
)

QuickDraw shows the *opposite* trend: the *lowest* training top-$k$
wins. $r = 0.05$ and $r = 0.10$ are tied and best at every rate;
$r = 0.15$ and $r = 0.20$ are uniformly $approx 0.7$–$0.9$ dB worse
(e.g. $23.447$ vs $22.568$ dB at $rho = 0.20$). Again the matching
rule fails — a single low training top-$k$ dominates, rather than
$k$ tracking $rho$.

#figure(
  image("figures/topk_quickdraw_5q.svg", width: 100%),
  caption: [QuickDraw-5q. Same panels as above. Here the lines
    *fall* with training top-$k$: the best operator is trained at
    $r in {0.05, 0.10}$, and the $r = 0.15$/$0.20$ runs sit a clear
    step lower at every eval rate.]
)

== What this shows

- *The fixed $10%$ training top-$k$ is not optimal.* It is too low on
  DIV2K (where $r = 0.20$ gains up to $+0.37$ dB) and at the high end
  of the good range on QuickDraw (where $r in {0.05, 0.10}$ tie and
  $r >= 0.15$ loses $approx 0.8$ dB). The training top-$k$ is a real
  hyperparameter, not a free choice.

- *"Train at the rate you deploy at" is the wrong rule.* On neither
  dataset does matching $k approx rho$ maximise PSNR at $rho$. A
  single, dataset-specific training top-$k$ is best across *all* eval
  rates simultaneously — high ($0.20$) for DIV2K, low ($<= 0.10$) for
  QuickDraw.

- *The optimum is dataset-dependent and opposite across scales.* The
  $m = 8$ natural-image dataset prefers a broader training objective;
  the $m = 5$ sparse-sketch dataset prefers a narrower one. A
  plausible reading: QuickDraw bitmaps concentrate energy in very few
  coefficients, so a wide top-$k$ spends gradient budget fitting
  near-zero coefficients; DIV2K photographs have heavier coefficient
  tails, so a wider top-$k$ supervises more of the signal that
  actually matters at $rho = 0.20$.

== Caveats

*Single seed.* Every cell is one run at seed $42$ on the headline
$1008$-step budget. The DIV2K $r = 0.15$ dip below $r = 0.10$ is the
clearest symptom: the cosine LR schedule is tied to the total epoch
count, and past step $approx 700$ the top-$k$ MSE valley is very flat,
so different runs can settle in slightly different basins (a caveat
inherited from the headline protocol). The *direction* of the effect
is consistent within each dataset across all four eval rates, but the
sub-$0.1$ dB orderings (e.g. $r = 0.05$ vs $r = 0.10$ on QuickDraw)
should not be over-read. A multi-seed repeat is the natural follow-up.

*Reproducibility.*

```
python experiments/qft_topk_sweep.py --gpu 0 --dataset div2k_8q
python experiments/qft_topk_sweep.py --gpu 0 --dataset quickdraw_5q
python tools/render_topk_sweep.py --dataset div2k_8q
python tools/render_topk_sweep.py --dataset quickdraw_5q
```

Training top-$k$ count: `experiment_utils.train_k_for(m, n, r)`
$= max(1, "round"(2^(m+n) dot r))$. Cells land at
`results/training/qft_topk_sweep/<dataset>/_runs/train_k<pct>/`; the
full matrix is in each dataset's `manifest.json`.
