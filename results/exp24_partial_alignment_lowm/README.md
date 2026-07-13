# Experiment 24 low-anchor extension

This run adds `m={4,8,16}` and repeats the grid through 8,192 with the same
gallery, targets, seeds and bootstrap as the primary exp24 run. It establishes
that minimum-norm OLS crosses R@1=0.5 at global budgets 16, 256, 1,024 and
4,096 for `C=1,16,64,256`; the corresponding transition is gradual and occurs
without any full-rank target cell.
