# Delta from the earlier arXiv version

The journal revision keeps the arXiv title and author name **Sergey Kurilenko**.
The following material changes must be disclosed in the arXiv replacement note
and the journal cover letter.

- The threat model is narrowed to numerical query-value confidentiality from
  an honest-but-curious evaluator, conditional on the exposed candidate set.
  SVD and PQ are described as public compression, not document protection.
- The CKKS implementation now includes a tested block-SIMD one-response kernel,
  a public-only spawned provider process, exact object counts, and real
  ciphertext serialization.
- Full actual-CKKS replay replaces plaintext arithmetic surrogates. The
  confirmatory analysis excludes all validation-query overlap and retains the
  canonical ArguAna denominator with an evaluable-query sensitivity analysis.
- New system evidence covers concurrency, throughput, tail latency, worker RSS,
  cold/warm sessions, key-material amortization, and measured TCP/TLS loopback
  transport.
- New utility evidence covers a dimension/quality/latency Pareto analysis,
  random-projection and coordinate-truncation controls, candidate Recall@100,
  shortlist-size and PQ-code sensitivity, and seed dispersion.
- New negative evidence quantifies public-PQ reconstruction, semantic
  information retained by disclosed candidate IDs, cross-request linkability,
  adaptive score-oracle extraction, and the absence of a circuit-privacy
  guarantee.
- The exact SEAL modulus primes, secret/error distributions, evaluation-key
  assumptions, and an independently executed pinned LWE-estimator transcript
  are archived.
- Strong float32, float16, and int8 vector-return baselines remain visible even
  where they are substantially faster than encrypted reranking.

The replacement note should state that these changes correct and substantially
extend the evaluation while preserving the same core public-PQ plus encrypted
reranking research question. No claim from the earlier version should be used
when it conflicts with the restricted boundary above.
