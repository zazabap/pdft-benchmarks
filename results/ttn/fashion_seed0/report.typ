#set page(paper: "a4", margin: (x: 16mm, y: 15mm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.6em)
#set heading(numbering: none)
#let summary = json("summary.json")
#let results = summary.last_observed
#let pct(x) = str(calc.round(100 * x, digits: 2)) + "%"
#let names = ("adam", "radam_printed", "radam_corrected")
#let labels = ("Adam", "RADAM, printed", "RADAM, corrected")
#let refs = (
  link("https://arxiv.org/abs/2609.00870")[Willner et al., arXiv:2609.00870],
  link("https://arxiv.org/abs/1801.00315")[Stoudenmire, arXiv:1801.00315],
  link("https://arxiv.org/abs/2602.14656")[Javaloy and Vergari, arXiv:2602.14656],
  link("https://arxiv.org/abs/2604.25755")[Scharf et al., arXiv:2604.25755],
)
#text(size: 19pt, weight: "bold")[Riemannian TTN reproduction]
#v(2mm)
#text(fill: rgb("555555"))[Fashion-MNIST · Independent implementation · 3 October 2026]
#v(3mm)

This experiment tests the Adam–RADAM comparison in #refs.at(0), Figures 5–6.
The compression advantage is reproduced qualitatively. The reported equality
of final classification accuracy is not reproduced under the reconstruction
choices below. The experiment establishes a working manifold-optimization
baseline; it supplies no evidence yet about logical reasoning.

= Classification after #results.adam.epoch epochs
#table(
  columns: (1.8fr, 1fr, 1fr, 1fr),
  inset: 5pt,
  stroke: 0.4pt + rgb("cccccc"),
  table.header([*Optimizer*], [*Test accuracy*], [*Test loss*], [*Train time*]),
  ..names.enumerate().map(((i, name)) => {
    let row = results.at(name)
    (labels.at(i), pct(row.test.accuracy), str(calc.round(row.test.loss, digits: 4)),
      str(calc.round(row.training_seconds)) + " s")
  }).flatten(),
)
#v(2mm)
The paper reports approximately 89% accuracy for all optimizers. These are
single-seed results, with identical initial weights and minibatch ordering.
Training time excludes initialization and evaluation. Apple MPS timings are
not directly comparable with the paper's NVIDIA A30 timings.

#figure(image("figures/training.svg", width: 100%), caption: [Test accuracy,
online minibatch training loss, and test accuracy against training time.
The dotted reference is the paper's approximate final accuracy. Epoch zero
uses a full training-set loss, before optimization.])


The experiment uses the complete official Fashion-MNIST split, 16 × 16 images,
a balanced TTN with maximum bond 16, and the paper's training budget and learning
rates. Both second-moment formulas are retained because the printed pseudocode
and conventional Adam differ. The comparison is a single independent run with
explicit reconstruction choices, rather than a verification using author code.

The qualitative compression result is useful for selecting a baseline for the
proposed tensor logic layer. It does not establish better reasoning, and the
classification discrepancy remains unresolved.

#pagebreak()

= Compression and the precision control
#figure(image("figures/compression.svg", width: 100%), caption: [Accuracy after
Schmidt compression at epochs 3 and 4. Each point is a measured retained
parameter count; lines connect sampled tolerances. Float32 and float64 use the
same CPU backend. The grid includes zero-tolerance controls.])

The compression sweep follows the discarded squared-Schmidt-weight criterion
in #refs.at(3), Section 2.4.3. After QR canonicalization, a depth-first SVD sweep
visits the internal edges. Physical and class legs are fixed. The tolerance is
scaled by the original full-tensor norm squared. Norm calculation uses explicit
scaling to avoid avoidable overflow before the norm itself leaves the numeric
range. The paper does not give its exact sweep or tolerance grid.

At epoch 3, the full tensor norm is approximately $1.94 times 10^18$ for Adam
and $3.68 times 10^2$ for corrected RADAM. At epoch 4, Adam's norm is about
$6.99 times 10^20$. Zero-tolerance float64 canonicalization preserves Adam's
predictions. Its loss of accuracy under truncation nevertheless persists in
float64. In this run, numerical overflow alone cannot explain the compression
gap. The growing norm is also a change of the represented tensor: exact
internal gauge changes preserve that norm.


#pagebreak()
= Implementation and unresolved details
The balanced binary TTN has 256 input sites, local feature map
$phi(x) = (1,x)^T / sqrt(1+x^2)$, maximum bond dimension 16, and a free
10-class root. Its 274,944 parameters are trained on all 60,000 official
training images and evaluated on all 10,000 test images. Loss is the batch
mean of the sum of squared one-hot class errors. Batch size is 256; learning
rates are 0.001 for Adam and 0.03 for RADAM.

Nonroot cores satisfy $B^T B = I$. RADAM uses horizontal projection
$G -> G-B(B^T G)$, one scalar second moment per core, momentum transport by
projection, and the POGO correction $M -> M(1.5I - 0.5M^T M)$ from #refs.at(2).
Float32 training follows the paper; initialization uses float64.

Algorithm 1 prints $(1-beta_1)$ in the second-moment update, where conventional
Adam uses $(1-beta_2)$. Both interpretations are evaluated, without bias
correction. “Corrected” denotes our sensitivity variant, not an author-confirmed
erratum. We did not tune learning rates to match the test accuracy.

The initialization follows the reduced-covariance construction in #refs.at(1),
including the norms of every branch traced out. We choose bilinear antialiased
resizing, row-major pixels in $[0,1]$, maximum allowed ranks, and a ridge-fitted
root. The relative ridge coefficient is $10^(-6)$. The target paper does not
fully specify these choices. They limit the strength of a numerical comparison.


= Square-core numerical limitation
The reported run evaluates the horizontal projection literally in float32.
The bottom two levels have square cores, whose exact quotient space is a point.
Projection roundoff can nevertheless be amplified by the adaptive denominator.
Thus these measurements use approximate finite-precision geometry. They should
not be treated as an exact-manifold control.

The implementation now offers `--square-projection freeze`, which leaves these
cores unchanged and is covered by a regression test. The full training results
have not been repeated with that option. This distinction is explicit in the
CLI and run documentation; the original results and source snapshot are retained.

= Checks and implications for the proposed logic layer
Eleven targeted tests cover dense contraction and differentiation, tangent
projections, the POGO error order, the printed RADAM update, reduced covariances,
canonicalization, truncation error bounds, packed inference, large
representable float32 norms, frozen square cores, resume rejection, and failure logging. A separate end-to-end check found bitwise identical
parameters after continuous and resumed training for both Adam and printed
RADAM. All source papers were downloaded and inspected; PDF hashes are recorded.

The next study should preserve this compression control when replacing the
classification head with a tensor logic layer. Compare at matched predictive
accuracy or matched reasoning success before attributing a compression advantage
to improved reasoning. First resolve the root initialization and optimizer
conventions, then test length generalization and rule composition on the chosen
real-world benchmark. This reproduction does not support a reasoning claim by
itself.

#v(2mm)
#text(size: 8.5pt)[
*Reproduction entry point:* `experiments/ttn/README.md` in `pdft-benchmarks`.
Code, environment pins, downloaded-source hashes, raw JSON measurements,
checkpoints, and vector figures accompany this report. The run uses seed 0;
no uncertainty across seeds has been estimated.
]
