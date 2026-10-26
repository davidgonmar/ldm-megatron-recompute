# Runs

4x B200, one node per run; steady step times in seconds from step 3 on; memory on rank 0 after
step 2 (allocated / peak allocated / peak reserved, GB).

| arm | node | s / step | memory |
|---|---|---|---|
| plain | i302 | 12.93 12.96 12.96 12.98 12.99 13.11 13.47 13.02 14.89 13.44 13.18 13.24 13.28 | 44.1 / 60.2 / 62.6 |
| mHC-4 | i405 | 17.08 17.07 17.06 17.07 17.09 17.23 17.32 17.18 19.49 17.31 17.40 17.52 17.26 | 44.2 / 64.1 / 66.9 |
| mHC-4 | i401 | 16.86 16.86 16.87 16.86 16.91 17.16 17.18 16.88 | 44.2 / 64.1 |
| LDM, PyTorch unwind | i302 | 18.88 18.80 18.84 18.81 18.85 | 55.4 / 69.3 / 71.7 |
| LDM, CuTe segments | i302 | 39.12 36.83 35.53 37.59 38.29 35.71 | 58.7 / 90.3 |
| LDM, CuTe under recompute | i302 | 17.08 17.04 16.99 17.01 | 53.5 / 75.2 / 77.5 |
