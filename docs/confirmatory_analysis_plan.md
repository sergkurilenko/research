# Confirmatory analysis plan for the journal revision

**Terminology note.** This plan was frozen before calculating the subset
statistics, but after the earlier full-test replay had been inspected. It is
therefore a post-exploratory revision analysis, not a prospective
preregistration or an independent second replay. Machine-readable keys and
filenames retain `confirmatory` / `strict_confirmatory` for schema stability;
the manuscript uses “frozen post-exploratory revision analysis.”

Created on 2026-07-11 before calculating the frozen revision-subset tables.
The earlier full-official-test results already existed, so this document is a
revision analysis plan rather than a claim of prospective preregistration.

## Query sets

- When an official development or training query set was used for validation,
  the complete official test set remains the frozen revision set.
- When validation was formed from the official test query IDs, the
  revision set is only the deterministic remaining 80% under seed 2026.
- Query IDs and membership hashes must be archived. No validation query may be
  reintroduced into a revision-subset mean or confidence interval.
- ArguAna keeps the canonical 1,124-query revision-subset denominator in the
  primary table after its separate 282-query validation split.
  The five queries whose judged-positive document IDs are absent from the
  pinned corpus receive zero contribution. A 1,401-evaluable-query sensitivity
  analysis is reported separately over 1,119 evaluable queries.

## Primary comparison

The primary endpoint is the paired per-query change in nDCG@10 between actual
CKKS reranking and plaintext projected reranking of exactly the same frozen
candidate IDs. A 95% percentile interval is calculated from 10,000 paired
query-bootstrap samples with seed 2026.

An absolute degradation margin of 0.002 nDCG@10 is used for a deliberately
strict operational non-inferiority check. Non-inferiority is reported only
when the lower endpoint of the paired 95% interval is above -0.002. Two-sided
equivalence is reported only when the complete interval lies inside
[-0.002, 0.002]. A confidence interval containing zero is never described as
evidence of equivalence. Datasets that do not satisfy a criterion are labeled
inconclusive rather than equivalent.

## Candidate and projection analyses

- Candidate quality is reported with Recall@100, nDCG@10, MRR@10, and the
  exact projected-reranking delta for K in {20, 50, 100, 200}.
- The fixed dimension grid is d' in {192, 256, 384, 512, 672, 768}.
- Projection controls are a corpus-fitted truncated SVD, a seeded random
  orthoprojector, and coordinate truncation where those controls are run.
- Each dimension row reports retained variance, projected exact utility,
  candidate utility, reranked utility, CKKS layout/latency, and serialized
  storage. The system operating point is discussed as a Pareto choice, not as
  a universally optimal dimension.
- PQ robustness uses fixed published seeds and reports seed dispersion rather
  than selecting the best seed.

All exploratory full-test tables remain archived for provenance but are not
used as frozen revision-subset evidence when they overlap validation queries.
