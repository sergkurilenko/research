Online Resource 1

Journal: The Journal of Supercomputing

Article title: Hybrid privacy-aware semantic search: SVD-truncated document
geometry and CKKS-encrypted query reranking under a restricted threat model

Author: Sergey Kurilenko
Affiliation: Moscow Institute of Physics and Technology, Dolgoprudny,
Moscow Region, Russia
Contact: sergkurilenko@gmail.com

Description
-----------
This archive is the reproducibility artifact for the article. It contains the
Springer manuscript source, experiment harnesses, unit tests, pinned commands,
environment and data manifests, compact projection artifacts, aggregate
result files, and per-query logs used for the reported statistical analyses.

Large raw corpora, model weights, and million-row embedding matrices are not
redistributed. Their public sources, immutable revisions or checksums, cache
keys, preprocessing recipes, and rebuild commands are recorded in the archive.
The million-vector Wikipedia track is a systems stress diagnostic; the pinned
BEIR collections provide the graded relevance evaluation.

The archive includes negative and boundary results. In particular, public-PQ
reconstruction, semantic information retained by disclosed candidate IDs,
adaptive score-oracle extraction, the absence of a circuit-privacy guarantee,
and all inconclusive non-inferiority checks are retained rather than filtered.

Start with README.md, docs/reproduce_system_revision.md, and
docs/results_manifest.md. Run the test suite before reproducing experiments.
