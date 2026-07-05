# Seed-98 full-parameter block-size sweep — per-family max

Init `family_random_basis(family,m,n,98)`, full-param top-10% MSE, fixed seed-42 test split.
Per family the value is the **max over {full-image, 4×4, 8×8, 16×16 blocks}** — the paper's protocol.
`paper` = current Table 2 (`tab:div2k_repr`/`tab:quickdraw_repr`).


## div2k_8q  (PSNR dB; ρ=0.01/0.05/0.10/0.20)

| family | max-over-sweep | argmax block | paper |
|---|---|---|---|
| qft | 21.22/25.24/28.09/32.26 | 8x8 | 21.21/25.28/28.09/32.26 |
| entangled_qft | 21.21/25.24/28.09/32.26 | 8x8 | 21.21/25.28/28.09/32.26 |
| tebd | 21.34/25.23/28.09/32.26 | 8x8 | 21.21/25.28/28.09/32.26 |
| mera | 21.34/25.23/27.81/32.05 | 4x4 | 21.21/25.28/27.98/32.05 |
| rich | 21.89/26.15/29.05/33.56 | 8x8 | 21.76/26.35/29.27/33.66 |

## quickdraw_5q  (PSNR dB; ρ=0.01/0.05/0.10/0.20)

| family | max-over-sweep | argmax block | paper |
|---|---|---|---|
| qft | 12.32/17.43/21.63/28.92 | 16x16 | 11.32/17.25/23.35/49.62 |
| entangled_qft | 12.67/17.43/21.62/28.91 | 16x16 | 11.32/17.25/23.35/39.81 |
| tebd | 12.66/17.43/21.62/28.90 | 8x8 | 11.32/17.25/23.35/39.81 |
| mera | 12.32/17.43/21.62/28.90 | 16x16 | 11.32/17.25/23.35/39.81 |
| rich | 12.97/18.39/23.65/34.32 | 4x4 | 11.51/17.84/24.07/49.62 |
