# DIV2K-8q independent baseline rerun

- seed: 42
- n_train: 500, n_test: 50
- img_size: 256
- baselines: ['block_dct_8', 'block_fft_8', 'block_pca_8', 'dct', 'fft', 'pca']

## PSNR (mean over test set, keep ratios = 0.05/0.10/0.15/0.20)

| baseline | 0.05 | 0.10 | 0.15 | 0.20 |
|---|---|---|---|---|
| block_dct_8 | 26.11 | 29.41 | 31.86 | 34.01 |
| block_fft_8 | 24.47 | 27.10 | 29.06 | 30.79 |
| block_pca_8 | 25.93 | 29.05 | 31.42 | 33.51 |
| dct | 25.36 | 27.61 | 29.33 | 30.85 |
| fft | 24.50 | 26.54 | 28.07 | 29.39 |
| pca | 18.15 | 18.15 | 18.15 | 18.15 |
