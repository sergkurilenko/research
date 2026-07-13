# Experiment 24: partial alignment without a full-rank gate

This is the primary alignment result used by the revised manuscript.

Configuration: e5-small, 100,000-record pool, `d_pub=96`, `d_priv=288`,
`C={1,16,64,256}`, a 10,000-record native residual gallery, 256 held-out
targets, three nested disclosure seeds and a 500-repetition hierarchical
seed-then-target bootstrap.

Every method is fitted whenever a target cell has at least one known pair:
rank-deficient Procrustes, minimum-norm OLS, ridge with zero/CV-selected
regularisation, and the polar projection of OLS.

The strongest method is minimum-norm OLS. It reaches R@1 above 0.9 at global
budgets 32, 512, 2,048 and 8,192 for `C=1,16,64,256`, respectively. These
points contain about 32--36 pairs in the average target cell, while full-rank
coverage is zero. Thus `C` compartmentalises diffuse disclosure, but
`d_priv` is an exact-map identifiability condition rather than a
re-identification threshold.

Reproduction command and environment are in `config.json`; complete outputs
are `summary.csv/json`, `metrics_per_target.csv`, `cell_fits.csv`,
`manifest.json` and `run.log`.
