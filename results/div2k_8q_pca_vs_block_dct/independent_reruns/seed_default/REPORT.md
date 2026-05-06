# DIV2K-8q independent baseline rerun

- seed: 42
- n_train: 500, n_test: 50
- img_size: 256
- baselines: ['bd_pca', 'block_bd_pca_8', 'block_dct_8', 'block_dct_8_rank', 'block_fft_8', 'block_pca_8', 'block_pca_8_rank', 'dct', 'dct_rank', 'fft', 'pca', 'pca_rank']

## PSNR (mean over test set, keep ratios = 0.05/0.10/0.15/0.20)

| baseline | 0.05 | 0.10 | 0.15 | 0.20 |
|---|---|---|---|---|
| bd_pca | 25.44 | 27.74 | 29.51 | 31.07 |
| block_bd_pca_8 | 26.13 | 29.30 | 31.68 | 33.77 |
| block_dct_8 | 26.11 | 29.41 | 31.86 | 34.01 |
| block_dct_8_rank | 22.35 | 24.11 | 25.20 | 26.30 |
| block_fft_8 | 24.47 | 27.10 | 29.06 | 30.79 |
| block_pca_8 | 25.93 | 29.05 | 31.42 | 33.51 |
| block_pca_8_rank | 22.38 | 24.17 | 25.35 | 26.40 |
| dct | 25.36 | 27.61 | 29.33 | 30.85 |
| dct_rank | 23.43 | 25.06 | 26.29 | 27.39 |
| fft | 24.50 | 26.54 | 28.07 | 29.39 |
| pca | 16.77 | 17.18 | 17.42 | 17.58 |
| pca_rank | 16.36 | 16.81 | 17.05 | 17.22 |
