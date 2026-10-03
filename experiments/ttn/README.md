# Riemannian TTN reproduction

This experiment independently implements the Fashion-MNIST comparison in
Willner et al., [*Stochastic Optimization of Tree Tensor Networks*](https://arxiv.org/abs/2609.00870),
Figures 5–6. It compares unconstrained Adam with quotient-manifold RADAM and
measures accuracy after Schmidt compression. It does not test logical reasoning.

The completed run and its limitations are summarized in
[the three-page report](../../results/ttn/fashion_seed0/report.pdf).

## Run

From the repository root, with Python 3.12:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r experiments/ttn/requirements-lock.txt
.venv/bin/python -m pytest -q --confcutdir tests/ttn tests/ttn
.venv/bin/python -m experiments.ttn.train \
  --out results/ttn/fashion_seed0_rerun \
  --methods adam radam_printed radam_corrected
.venv/bin/python -m experiments.ttn.analyze --out results/ttn/fashion_seed0_rerun --sweep
```

The default device is MPS when available, otherwise CPU. CUDA users should pass
`--device cuda`. Data are downloaded with torchvision's Fashion-MNIST checksum
verification to `~/.cache/pdft-reproduction/data`. Checkpoints stay in the
ignored `results/**/checkpoints/` directory. JSON measurements and PDF/SVG figures
are separate, small artifacts. The experiment requires no PDFT/JAX installation;
`--confcutdir` avoids the unrelated JAX imports in the repository-wide conftest.

Use a new output directory for another seed or protocol. `--resume` continues
from the last complete checkpoint for each optimizer, requiring the initial and
all latest optimizer checkpoints; minibatch order is derived
from the seed and epoch. A smaller functional check is:

```sh
.venv/bin/python -m experiments.ttn.train --out results/ttn/smoke \
  --train-size 1024 --test-size 512 --epochs 1
```

## Protocol and reconstruction choices

| Item | Paper / implementation |
|---|---|
| Dataset | Official 60,000 training and 10,000 test Fashion-MNIST images |
| Image size | 16 × 16, as in the paper |
| Resampling | Bilinear with antialiasing, pixels divided by 255; exact kernel is not reported |
| Pixel ordering | Row-major, following the cited initialization paper; not specified by Willner et al. |
| Feature map | `(1, x) / sqrt(1 + x²)` |
| Tree | Balanced binary, maximum bond 16, ten-dimensional free root |
| Parameters | 274,944, including square isometries at the bottom two levels |
| Initialization | Training-only reduced-covariance eigenvectors, following Stoudenmire, Section III |
| Root initialization | Relative ridge least squares, ridge = 10⁻⁶ times mean Gram eigenvalue; an explicit choice because the root initialization is not reported |
| Loss | Mean over samples of the sum of ten squared class errors, one-hot labels |
| Budget | Batch 256, 30 epochs, no early stopping or learning-rate schedule |
| Adam | PyTorch Adam, lr = 0.001, betas = (0.9, 0.999), default epsilon 10⁻⁸ |
| RADAM | lr = 0.03, betas = (0.9, 0.999), one scalar second moment per core |
| Geometry | Horizontal projection `G − B(BᵀG)`; free root; projection transport of momentum |
| Retraction | One POGO polar correction, `M(1.5I − 0.5MᵀM)` |
| Precision | Float32 training; float64 covariance initialization |
| Hardware | Local Apple MPS, rather than the paper's NVIDIA A30 |
| Replicates | One paired seed; no statistical significance claim |

The initialization uses reduced covariances of the whole coarse-grained feature
vector. The covariance of a given pair is weighted by the product of squared
norms of **all other branches**. Dropping these factors changes the initialization
after truncation. Initializers, including the root, never use test data.

### Ambiguous second-moment formula

Algorithm 1 prints

```
v_t = beta2 * v_(t-1) + (1-beta1) * ||G_t||_F².
```

The `radam_printed` variant implements this literally. `radam_corrected` uses
`1-beta2`, as in conventional Adam. Neither adds bias correction, since the
printed algorithm does not include it. Both clamp the denominator below at
10⁻¹² to define the zero-gradient case. With zero initial second moments, the
corrected rule would produce steps ten times larger for the same gradient and
momentum history. Once training diverges, their histories also differ.

This is an apparent inconsistency, not a confirmed author correction. Both
variants are retained rather than selecting whichever matches the published
accuracy. No learning rates were selected using the test set.

### Square-core numerical control

The recorded full run evaluates the horizontal projection literally in float32,
including the square bottom cores. Their exact quotient space has zero dimension,
but subtraction roundoff can be amplified by RADAM's adaptive normalization.
This is a limitation of the literal numerical experiment. It does not represent
an additional mathematical degree of freedom.

`--square-projection freeze` skips updates and polar corrections for these cores.
A regression test verifies exact parameter preservation with arbitrary gradients.
The default `literal` retains the recorded protocol for reproducibility. The
30-epoch measurements in the report have **not** been repeated with `freeze`;
use it for the next geometry-controlled experiment in a new output directory.

### Compression

After bottom-up QR canonicalization, a depth-first SVD sweep moves the
orthogonality center through the tree. Each internal edge discards the smallest
Schmidt values whose squared sum is at most `fraction * ||W||_F²`, measured
relative to the original tensor norm. The physical and class legs are fixed.
The reported retention is the actual number of stored core entries after rank
reduction divided by the original parameter count. Padding used for fast
inference does not change that mathematical count.

This implements the local discarded-weight criterion of Scharf et al., Section
2.4.3. Willner et al. do not report their exact sweep order, tolerance grid, or
compression implementation. Our curves therefore compare the same operation,
but do not claim identical truncation choices. Zero tolerance supplies a
prediction-preservation control. Both float32 and float64 sweeps use CPU linear
algebra to hold the backend fixed. Norms are explicitly scaled before squaring
to avoid preventable overflow in the norm calculation itself. Numerical failures
are recorded as failures, never plotted as zero accuracy.

The full weight-tensor norm is invariant under exact internal gauge changes.
Its growth under Adam is therefore a change of the learned tensor, not merely
a rescaling of its factors. It can affect compression even when high-precision
canonicalization preserves predictions. Float64 compression distinguishes that
effect from finite-precision failures, within the precision it can resolve.

## Files and validation

- `model.py`: vectorized TTN contractions, data initialization, and RADAM.
- `compression.py`: canonicalization, Schmidt truncation, and packed inference.
- `train.py`: paired training, checkpointing, configuration, and epoch metrics.
- `analyze.py`: resumable compression sweeps and PDF/SVG plotting.
- `sources.json`: downloaded source PDF hashes and implementation pointers.
- `requirements-lock.txt`: the versions used for this run.

Eleven targeted tests compare a small TTN against its explicit dense tensor, check autograd,
horizontal and Stiefel tangent projections, the POGO orthogonality-error order,
the literal optimizer update, reduced covariances against explicit partial
traces, gauge-invariant predictions, truncation error, and packed inference.
They also test a large float32 norm that is representable although its squared
value is not.

Training configuration records software versions, seed, source-file hashes,
and Git base revision. The initial implementation snapshot used by the local
run is also saved under its ignored checkpoint directory. Training times exclude
initialization, evaluation, and compression. Concurrent CPU compression can
change local timings; they are not an A30 performance reproduction.

## Sources inspected as downloaded PDFs

1. [Willner et al., 2609.00870](https://arxiv.org/abs/2609.00870): Sections 2,
   4.1 and 5.1; Algorithm 1; Figures 5–6. Target experiment and geometry.
2. [Stoudenmire, 1801.00315](https://arxiv.org/abs/1801.00315): Sections III–IV.
   Reduced-covariance initialization and supervised root fitting. The available
   downloaded source is v1; the journal publication appeared in 2018.
3. [Javaloy and Vergari, 2602.14656](https://arxiv.org/abs/2602.14656): polar
   correction used by POGO. Our implementation transposes the paper's row
   convention to column isometries and uses only its retraction step.
4. [Scharf et al., 2604.25755](https://arxiv.org/abs/2604.25755): Section 2.4.3,
   Equation 6. Schmidt compression, not the paper's SAR benchmark.

No author-supplied experiment implementation was located or used. This is an
independent reproduction with the reconstruction choices above, not a claim
of bitwise agreement with the original framework.
