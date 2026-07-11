# IR expansion for the journal revision

The result schemas retain the historical key `strict_confirmatory` for stable
reproduction. Because the full encrypted replay predated this split analysis,
the manuscript describes these rows more precisely as a **frozen
post-exploratory revision-analysis subset**. They are disjoint from validation
but are not a prospectively preregistered or independently rerun experiment.

This document describes the additional retrieval experiments requested after
the acceptance audit.  The manuscript source is deliberately not modified by
the experiment harness; tables and prose are incorporated only after the
result manifests are complete and audited.

## Frozen analysis policy

The frozen revision-analysis policy is fixed in
`docs/confirmatory_analysis_plan.md`.
Official development qrels (or training qrels when development is absent) are
used for validation.  ArguAna, SciDocs, and TREC-COVID have test qrels only,
so seed 2026 defines a deterministic SHA-256-ordered 20/80 validation and
revision-subset partition (stored under the schema key
`strict_confirmatory`). Query ID files and membership hashes are in
`results/system_revision/ir_expansion/splits/`.

The primary endpoint is the paired per-query nDCG@10 change between actual
CKKS and plaintext reranking of the same candidate IDs.  The analysis uses
10,000 paired bootstrap samples with seed 2026 and a two-sided 95% percentile
interval.  The operational margin is 0.002.  Non-inferiority is reported only
if the lower endpoint is greater than -0.002; equivalence is reported only if
the *complete* interval lies inside [-0.002, 0.002].  Covering zero alone is
never called equivalence.

The official ArguAna ZIP contains 1,406 queries and qrels but omits the judged
positive corpus document for five of them. After the separate 282-query
validation split, the primary frozen revision denominator is 1,124 queries,
with deterministic zero metric contributions for those five impossible
queries. A 1,119-evaluable-query sensitivity result is kept separately.
Missing documents are not reconstructed or invented.

## Reproduction commands

Use the same pinned CUDA environment as the original graded-IR runs:

```powershell
cd <repository-root>
$py = 'python'

& $py -m pytest tests\test_ir_expansion.py tests\test_graded_ir_bench.py -q
& $py system\ir_expansion.py splits
& $py system\ir_expansion.py controls
& $py system\ir_expansion.py pareto
& $py system\ir_expansion.py pq
& $py system\validate_ir_expansion.py
```

Large PQ indices are cached under the user-selected experiment cache; result
JSON/CSV files remain under
`results/system_revision/ir_expansion`.  Every index record includes the
projection/corpus fingerprints, serialized byte count, file SHA-256, M, seed,
training size, and iteration count.  All stages checkpoint atomically.

## Experiment design

The pure projection ablation uses nested prefixes of the already frozen
corpus-only SVD basis at dimensions 192, 256, 384, 512, and 672.  The
768-dimensional row is the full-rank centred geometry.  It reports retained
fit-sample variance, exhaustive projected nDCG@10 and Recall@100, paired
deltas, and exact-search timing.  Because the first 672 columns come from one
frozen ordered basis, differences across these rows change only retained SVD
dimension.

The accompanying deployment Pareto curve uses M=d/8 with 8-bit
subquantizers, fixing an eight-dimensional PQ subvector width.  This is
explicitly a joint SVD/PQ/storage co-design curve rather than a pure SVD
effect.  The published d=672, M=96, K=100 index remains a separate deployment
point.  M sensitivity uses M in {32, 48, 84, 96}; K sensitivity uses K in
{20, 50, 100, 200}.  For K below 100, Recall@100 is evaluated with at most K
returned documents and is numerically the candidate Recall@K.  Both labels
are stored to prevent misinterpretation.  FiQA and TREC-COVID additionally use
PQ seeds 17, 42, and 2026 for M=84 and M=96; dispersion is reported without
choosing the best seed.

Projection controls use a seeded Gaussian orthoprojector and coordinate
truncation at dimensions 384 and 672.  Random projections are nested prefixes
of the same 672-column orthoprojector.  Documents are centred using the frozen
passage mean; online queries are never centred.

Every utility table contains disjoint `validation` and
`strict_confirmatory` branches.  Validation may guide operating-point
discussion. Frozen revision-subset rows are reported only after settings are
fixed and are never used for selection.

## Completed frozen revision-subset CKKS analysis

The strict query-subset calculation is a mathematically exact reuse of the
existing query-independent CKKS records: each JSONL row is one complete
request, with no cross-query state, training, aggregation, or cryptographic
dependency.  A replay would change latency noise but cannot change which
records belong to a frozen query subset.

| Dataset | Revision-subset queries | Mean CKKS minus plaintext nDCG@10 | Paired 95% interval | Decision at 0.002 |
|---|---:|---:|---:|---|
| ArguAna | 1,124 canonical / 1,119 evaluable | -0.000489 | [-0.002199, 0.001164] | inconclusive |
| FiQA | 648 | 0.000000 | [0.000000, 0.000000] | non-inferior and equivalent |
| NFCorpus | 323 | -0.000041 | [-0.000111, 0.000000] | non-inferior and equivalent |
| SciDocs | 800 | 0.000000 | [0.000000, 0.000000] | non-inferior and equivalent |
| SciFact | 300 | 0.000000 | [0.000000, 0.000000] | non-inferior and equivalent |
| TREC-COVID | 40 | 0.000079 | [0.000000, 0.000238] | non-inferior and equivalent |

ArguAna therefore must not be described as equivalent or non-inferior under
the deliberately strict 0.002 rule; its result is inconclusive.  The other
five datasets satisfy the frozen criterion.

## Projection-control utility (completed)

The first control timing pass overlapped a separate systems run and was
discarded.  A clean exclusive rerun replaced it on 2026-07-11 from 14:03:06
to 14:03:31 UTC.  Both utility and timing fields in the final manifest now
come from that uncontaminated run.

At d=384, SVD has the highest strict-confirmatory nDCG@10 on every dataset.
Examples are FiQA 0.3687 versus 0.2188 coordinate and 0.1926 random; SciFact
0.6885 versus 0.5324 and 0.5530; and TREC-COVID 0.6722 versus 0.4302 and
0.4899.  At d=672 the gaps narrow, but SVD remains strongest on all six:
ArguAna 0.4418, FiQA 0.3801, NFCorpus 0.3270, SciDocs 0.1661, SciFact 0.6968,
and TREC-COVID 0.6867.

## Dimension and storage Pareto results

The table below is a six-dataset unweighted macro-average.  Explained
variance is measured on each frozen corpus-only fit sample.  `Exact val` is
the disjoint validation branch; all other utility columns shown here are
strict-confirmatory.  The PQ curve fixes eight dimensions per subquantizer,
so its code bytes increase with dimension.

| d | Mean retained variance | Exact val nDCG@10 | Exact confirm nDCG@10 | PQ+rerank confirm nDCG@10 | Candidate recall@100 | PQ code bytes/doc |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 0.7828 | 0.4049 | 0.4081 | 0.4092 | 0.4917 | 24 |
| 256 | 0.8578 | 0.4306 | 0.4270 | 0.4281 | 0.5034 | 32 |
| 384 | 0.9499 | 0.4517 | 0.4411 | 0.4435 | 0.5127 | 48 |
| 512 | 0.9881 | 0.4560 | 0.4470 | 0.4472 | 0.5104 | 64 |
| 672 | 0.9986 | 0.4572 | 0.4498 | 0.4490 | 0.5113 | 84 |
| 768 | 1.0000 | 0.4583 | 0.4512 | 0.4478 | 0.5140 | 96 |

The important systems conclusion is a plateau rather than a universally best
dimension.  Relative to d=672, d=512 loses 0.00122 validation macro nDCG in
the pure SVD row and 0.00078 in the joint PQ row, while reducing the PQ code
from 84 to 64 bytes per document and moving CKKS from the 1,024-padded layout
to the 512-padded layout.  The paper should therefore describe d=512 as the
latency/storage-favouring point and the already frozen d=672 setting as the
utility-favouring deployment point.  Strict-confirmatory results are reported
for completeness but were not used to change that frozen setting.

## K and M sensitivity

For the published d=672, M=96 index, increasing K primarily improves candidate
coverage.  Recall@100 with K below 100 treats unfilled ranks as misses.  At
K=200, candidate recall is Recall@200 while the separate reranked column still
evaluates the first 100 returned results.

| K | nDCG@10 | MRR@10 | Candidate recall@K | Reranked recall@100 |
|---:|---:|---:|---:|---:|
| 20 | 0.4352 | 0.5273 | 0.3962 | 0.3962 |
| 50 | 0.4453 | 0.5335 | 0.4643 | 0.4643 |
| 100 | 0.4506 | 0.5363 | 0.5148 | 0.5148 |
| 200 | 0.4516 | 0.5349 | 0.5735 | 0.5466 |

At K=100, reducing M gives the expected quality/storage trade-off:

| M (bytes/doc) | Strict-confirmatory macro nDCG@10 | Candidate recall@100 | Serialized bytes, six indices |
|---:|---:|---:|---:|
| 32 | 0.4166 | 0.4397 | 12.84 MB |
| 48 | 0.4373 | 0.4722 | 17.19 MB |
| 84 | 0.4490 | 0.5113 | 26.99 MB |
| 96 | 0.4506 | 0.5148 | 30.25 MB |

The d=672, M=96, K=100 per-dataset rows are:

| Dataset | nDCG@10 | MRR@10 | Candidate recall@100 |
|---|---:|---:|---:|
| ArguAna | 0.4410 | 0.3634 | 0.9048 |
| FiQA | 0.3777 | 0.4552 | 0.5896 |
| NFCorpus | 0.3272 | 0.5224 | 0.2647 |
| SciDocs | 0.1649 | 0.3049 | 0.3250 |
| SciFact | 0.6970 | 0.6669 | 0.9120 |
| TREC-COVID | 0.6959 | 0.9050 | 0.0929 |

PQ seed sensitivity is small on FiQA: at K=100 the across-seed sample standard
deviation is 0.00136 nDCG for M=84 and 0.00208 for M=96.  TREC-COVID has only
40 strict-confirmatory queries and correspondingly larger nDCG dispersion
(0.01094 and 0.01196), although its candidate-recall standard deviations stay
below 0.0016.  No best seed is selected or substituted.

## Artifact validation

`system/validate_ir_expansion.py` parses every JSON with non-finite constants
forbidden, verifies disjoint query membership, complete d/M/K/seed grids,
monotone candidate recall, cross-manifest metric identities, serialized byte
counts, and SHA-256 for every index.  The final report passes 614 checks over
62 unique serialized indices.  Table-ready files are:

- `confirmatory_ckks.csv` (6 rows),
- `projection_controls.csv` (72 rows),
- `svd_pareto.csv` (72 rows), and
- `pq_sensitivity.csv` (256 rows).

The machine-readable hashes and individual check records are in
`results/system_revision/ir_expansion/validation_report.json`.
