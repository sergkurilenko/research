# arXiv metadata draft

## Title

Hybrid privacy-aware semantic search: SVD-truncated document geometry and CKKS-encrypted query reranking under a restricted threat model

## Authors

Sergey Kurilenko

Affiliation: Moscow Institute of Physics and Technology, Dolgoprudny, Moscow Region, Russia

## Abstract

Use the plain-text contents of `submission/arxiv_abstract.txt`.

## Comments

37 pages, 6 figures, 20 tables. Journal-oriented revision with validation-disjoint BEIR evaluation, actual CKKS reranking, systems scaling, projection and PQ controls, and explicit leakage audits. Reproducibility artifacts and per-query records are included in the companion repository.

## Subjects

- Primary: Information Retrieval (`cs.IR`)
- Cross-list: Cryptography and Security (`cs.CR`)
- Cross-list: Distributed, Parallel, and Cluster Computing (`cs.DC`)
- Cross-list: Machine Learning (`cs.LG`)

## Keywords

homomorphic encryption; dense retrieval; query confidentiality; SVD truncation; product quantization; leakage evaluation

## Replacement note

This version replaces the earlier arXiv manuscript under the same title. It materially narrows and clarifies the security claim: SVD and PQ are treated as public compression, candidate identifiers and the access pattern are disclosed, and no document, circuit, unlinkability, or access-pattern privacy is claimed. The earlier secret-rotation argument has been removed.

The revision adds the block-SIMD one-response CKKS implementation, a public-only provider process, strong vector-return baselines, validation-separated BEIR and actual-CKKS analysis, SVD/random/coordinate controls, K/M/PQ-seed sensitivity, concurrency and TCP/TLS measurements, a same-host Windows/WSL portability check, candidate-ID linkability and neighbourhood-reproduction audits, an adaptive score-oracle audit, and an independently executable LWE-estimator transcript. The title is unchanged to preserve continuity with the existing record.
