# Experiment 27: formally calibrated Gaussian release baseline

This directory replaces the earlier pseudo-DP noise sweep with a one-shot
mechanism that has an explicit adjacency relation, clipping bound, global
sensitivity, and exact analytic Gaussian calibration.

## Main finding

At `epsilon=1`, nDCG@10 is at most 0.011 across the 8 evaluated cases. At
`epsilon=512`, native-gallery R@1 is already 0.962--0.995, while retrieval
remains far below the corrected SHARD target.

Only 3/8 strict utility matches occur on the finite grid, all at
`epsilon=32768`; their linkage R@1 is at least 0.995. The other cases do not
satisfy both utility tolerances even at that extremely weak privacy parameter.
Thus this formal per-record Gaussian release has no evaluated high-utility
point that also resists native clean-reference linkage. This does not imply
that SHARD is unlinkable or DP.

## Privacy statement

- Adjacency: fixed-size bounded/replacement adjacency; databases differ in one row.
- Public information: row participation/identifiers, encoder, dimension and release size.
- Clipping: `clip_B(x)` with `B=1.000001`.
  The bound is fixed from the e5 unit-normalisation contract with a float32
  safety margin; it is not estimated from the evaluated corpus.
- Global L2 sensitivity: `Delta_2=2B=2.000002` for the concatenated database release.
- Mechanism: clipped rows plus iid Gaussian noise; row-wise unit normalisation is DP-preserving post-processing.
- Accounting: exact Gaussian privacy-loss equation at `delta=1e-06`.

This is a **content privacy conditional on public participation** guarantee, not
an add/remove membership guarantee. A trusted curator gives the central-DP
interpretation. If each single-record owner clips and perturbs locally, the same
calibration gives approximate local DP for that vector. Independent repeated
releases compose; the elementary bound is `(k epsilon, k delta)`. The one-shot
joint database publication is not charged N times because replacement
adjacency changes only one row. Public or independently trained model
parameters are assumed. Fitting a centroid/PCA on the private release set
would require a separate DP mechanism and composition.

The mechanism protects document-vector content only; it makes no query-privacy
claim. Although identifiers are public in the formal adjacency model, the
linkage diagnostic removes them before matching so that R@1 measures geometric
linkability rather than the tautological identifier channel. It is an attack
diagnostic, not an empirical test or replacement for the DP theorem.

## Calibration

| epsilon | sigma | noise multiplier sigma/Delta | achieved delta |
|---:|---:|---:|---:|
| 0.5 | 16.115253 | 8.0576185 | 1e-06 |
| 1 | 8.4493662 | 4.2246789 | 1e-06 |
| 2 | 4.460957 | 2.2304763 | 1e-06 |
| 4 | 2.3870396 | 1.1935186 | 1e-06 |
| 8 | 1.3058721 | 0.65293538 | 1e-06 |
| 16 | 0.73722405 | 0.36861166 | 1e-06 |
| 32 | 0.43237493 | 0.21618725 | 1e-06 |
| 64 | 0.26380843 | 0.13190408 | 1e-06 |
| 128 | 0.16683265 | 0.083416239 | 1e-06 |
| 256 | 0.10865853 | 0.05432921 | 1e-06 |
| 512 | 0.072395124 | 0.036197526 | 1e-06 |
| 1024 | 0.049054349 | 0.02452715 | 1e-06 |
| 2048 | 0.03364863 | 0.016824298 | 1e-06 |
| 4096 | 0.023285156 | 0.011642566 | 1e-06 |
| 8192 | 0.016215032 | 0.0081075077 | 1e-06 |
| 16384 | 0.011342139 | 0.0056710638 | 1e-06 |
| 32768 | 0.0079587951 | 0.0039793936 | 1e-06 |

## Retrieval utility and native-gallery linkage

The table reports raw utility, the strongest conventional privacy point
(`epsilon=1`), and the largest/weakest evaluated finite epsilon. Linkage uses
256 released targets against a clean native gallery of up to 5,000 documents
after row identifiers have been removed. CIs are hierarchical bootstrap CIs
over noise seeds and queries/targets.

| suite / encoder / data | eps | nDCG@10 | R@100 | score r | link R@1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| beir/multilingual-e5-base/arguana | inf | 0.346 | 0.955 | 1.000 | - | - |
| beir/multilingual-e5-base/arguana | 1.0 | 0.000 | 0.013 | -0.001 | 0.000 | 0.522 |
| beir/multilingual-e5-base/arguana | 32768.0 | 0.333 | 0.942 | 0.971 | 0.996 | 1.000 |
| beir/multilingual-e5-base/nfcorpus | inf | 0.327 | 0.287 | 1.000 | - | - |
| beir/multilingual-e5-base/nfcorpus | 1.0 | 0.009 | 0.022 | 0.006 | 0.000 | 0.506 |
| beir/multilingual-e5-base/nfcorpus | 32768.0 | 0.321 | 0.278 | 0.951 | 0.999 | 1.000 |
| beir/multilingual-e5-base/scifact | inf | 0.637 | 0.915 | 1.000 | - | - |
| beir/multilingual-e5-base/scifact | 1.0 | 0.003 | 0.023 | 0.011 | 0.000 | 0.511 |
| beir/multilingual-e5-base/scifact | 32768.0 | 0.625 | 0.899 | 0.961 | 1.000 | 1.000 |
| beir/multilingual-e5-small/arguana | inf | 0.358 | 0.963 | 1.000 | - | - |
| beir/multilingual-e5-small/arguana | 1.0 | 0.000 | 0.015 | 0.004 | 0.000 | 0.511 |
| beir/multilingual-e5-small/arguana | 32768.0 | 0.346 | 0.956 | 0.971 | 0.996 | 1.000 |
| beir/multilingual-e5-small/nfcorpus | inf | 0.302 | 0.248 | 1.000 | - | - |
| beir/multilingual-e5-small/nfcorpus | 1.0 | 0.010 | 0.028 | 0.002 | 0.000 | 0.522 |
| beir/multilingual-e5-small/nfcorpus | 32768.0 | 0.297 | 0.250 | 0.958 | 0.995 | 1.000 |
| beir/multilingual-e5-small/scifact | inf | 0.598 | 0.901 | 1.000 | - | - |
| beir/multilingual-e5-small/scifact | 1.0 | 0.000 | 0.014 | -0.000 | 0.000 | 0.502 |
| beir/multilingual-e5-small/scifact | 32768.0 | 0.581 | 0.895 | 0.966 | 1.000 | 1.000 |
| miracl/multilingual-e5-small/bn | inf | 0.688 | 0.963 | 1.000 | - | - |
| miracl/multilingual-e5-small/bn | 1.0 | 0.000 | 0.000 | 0.004 | 0.000 | 0.535 |
| miracl/multilingual-e5-small/bn | 32768.0 | 0.675 | 0.964 | 0.978 | 1.000 | 1.000 |
| miracl/multilingual-e5-small/sw | inf | 0.680 | 0.946 | 1.000 | - | - |
| miracl/multilingual-e5-small/sw | 1.0 | 0.000 | 0.000 | 0.004 | 0.000 | 0.512 |
| miracl/multilingual-e5-small/sw | 32768.0 | 0.669 | 0.945 | 0.976 | 1.000 | 1.000 |

## Matched utility against corrected SHARD

A match requires `|Delta nDCG@10| <= 0.01` and
`|Delta Recall@100| <= 0.02` simultaneously.
No interpolation is used; a match is claimed only at an actually evaluated point.

| suite / encoder / data | matched | strongest matching epsilon | note |
|---|---:|---:|---|
| beir/multilingual-e5-base/arguana | false | - | no finite epsilon on the evaluated grid meets both utility tolerances |
| beir/multilingual-e5-base/nfcorpus | true | 32768.0 | finite grid match |
| beir/multilingual-e5-base/scifact | false | - | no finite epsilon on the evaluated grid meets both utility tolerances |
| beir/multilingual-e5-small/arguana | true | 32768.0 | finite grid match |
| beir/multilingual-e5-small/nfcorpus | true | 32768.0 | finite grid match |
| beir/multilingual-e5-small/scifact | false | - | no finite epsilon on the evaluated grid meets both utility tolerances |
| miracl/multilingual-e5-small/bn | false | - | no finite epsilon on the evaluated grid meets both utility tolerances |
| miracl/multilingual-e5-small/sw | false | - | no finite epsilon on the evaluated grid meets both utility tolerances |

The comparison is deliberately limited to retrieval utility. The Gaussian
baseline also uses an exact full-corpus scan, whereas the exp23 SHARD target
uses two-stage K=200 routing. Therefore the match controls quality, not
latency, traffic or compute. SHARD and the Gaussian mechanism have different
guarantees and attack surfaces; a native-gallery linkage rate must not be
relabelled as an epsilon value, and SHARD must not be described as DP.

## Files

- `config.json`: complete mechanism and run configuration.
- `calibration.csv/json`: exact sigma values and achieved deltas.
- `calibration_tests.json`: executable formula/clipping unit tests.
- `per_query.csv`: raw seed/query measurements.
- `per_target.csv`: raw seed/target linkage measurements.
- `per_seed_summary.csv/json`: compact per-seed aggregates of both raw tables.
- `utility_summary.csv/json`: bootstrap summaries.
- `attack_summary.csv/json`: linkage bootstrap summaries.
- `matched_utility.csv/json`: explicit corrected-SHARD matching decisions.
- `case_diagnostics.json`, `run_info.json`, and `run.log`: audit trail.

Elapsed time: 69.6 seconds.
