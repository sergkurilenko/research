# Evidence manifest for the revised manuscript

This manifest maps each empirical claim in paper_revised/main.tex to the
machine-readable evidence under results/system_revision/. It is intended for
review of the rewritten public-PQ/CKKS study, not the superseded
random-projection paper in paper/.

## Primary evidence

| Manuscript material | Result summary | Per-request evidence |
|---|---|---|
| Frozen post-exploratory revision-subset membership and inference rule | ir_expansion/confirmatory_splits.json | ir_expansion/splits/*.txt |
| SVD dimension/PQ storage Pareto curve | ir_expansion/svd_pareto.json and .csv | checkpoints and hash-addressed Faiss metadata recorded in the JSON |
| SVD versus random/coordinate controls | ir_expansion/projection_controls.json | deterministic validation and frozen revision branches (`strict_confirmatory` schema key) in the JSON |
| M, K, and PQ-seed sensitivity | ir_expansion/pq_sensitivity.json | every dataset/M/K/seed point retained in the JSON |
| CKKS microbenchmark (Table `tab:microbenchmark`) | ckks_micro_d672_K100.json | none; 20 repeated kernel measurements are embedded in the summary |
| CKKS dimension and shortlist sweep (Table `tab:ckks-scaling`; Fig. `fig:kernel-scaling`) | ckks_sweep.json | none; repeated timing records are embedded in the summary |
| Million-vector utility (Table `tab:heldout-utility`) | retrieval_e5base_k672_K100_test.json | retrieval_e5base_k672_K100_test.jsonl |
| Unified spawned-process request path (Table `tab:unified-latency`) | unified_ckks_e5base_k672_K100_test.json | unified_ckks_e5base_k672_K100_test.jsonl |
| Five restart check | restarts/unified_validation_restarts_aggregate.json | restarts/unified_validation_restart_*.jsonl |
| Provider saturation, sessions, TCP/TLS loopback | systems_expansion/windows_full.json and validation JSON | all restart/condition/request summaries embedded in the full JSON |
| Same-CPU Windows/WSL CKKS control | systems_expansion/windows_wsl_runtime_pair.json and .csv | raw platform records in windows_pair_ckks_micro.json and wsl_linux_ckks_micro.json |
| Vector-return baselines (Table `tab:return-vector`; Fig. `fig:tradeoff`) | vector_return_baselines_e5base_K100_test.json | vector_return_baselines_e5base_K100_test.jsonl |
| PQ scan scaling (Table `tab:pq-scaling`) | pq_scaling_e5base_k672_K100.json | timings embedded in the summary |
| Ideal-link planning model (Fig. `fig:tradeoff`) | network_tradeoff.json | derived only from logged payloads and declared link assumptions |
| Public-PQ reconstruction (Table `tab:pq-leakage`; Fig. `fig:leakage`) | pq_leakage_e5base_k672_sample100k.json | deterministic 100,000-row sample information is embedded in the summary |
| Candidate-ID semantic leakage (Table `tab:candidate-leakage`) | security_expansion/candidate_id_million.json and candidate_id_beir_confirmatory.json | matching JSONL files retain query-level observations |
| Adaptive score-oracle audit (Fig. `fig:leakage`) | score_oracle_extraction_d672_3docs.json | encrypted-request measurements are embedded in the summary |
| Circuit-privacy boundary | security_expansion/circuit_privacy_boundary.json | 20 fixed-request and 20 fresh-encryption diagnostics in the JSON |
| Exact CKKS parameters and independent attack costs | security_expansion/lwe_security/security_parameters.json | pinned Sage input plus raw stdout/stderr and hashes |
| Frozen BEIR retrieval (Table `tab:graded-official`; Fig. `fig:beir-utility`) | graded_ir_official_test.json | graded_ir_official_test_*.jsonl |
| Frozen actual-CKKS decisions (Table `tab:graded-ckks`) | ir_expansion/confirmatory_splits.json plus graded_ckks_replay_*_full.json | frozen query-ID lists and records selected from the earlier full encrypted replay |

The six BEIR full-replay summaries are
graded_ckks_replay_scifact_full.json,
graded_ckks_replay_nfcorpus_full.json,
graded_ckks_replay_arguana_full.json,
graded_ckks_replay_scidocs_full.json,
graded_ckks_replay_fiqa_full.json, and
graded_ckks_replay_trec-covid_full.json.

## Artifact boundary

The reproducibility archive includes source code, tests, result summaries,
query-level JSONL logs, the 672-dimensional projection basis, and the hardware
manifest. It intentionally excludes
cache/E_docs_e5base_proj672_1000000.npy (about 2.69 GB), because it is a
derived cache rebuildable from the separately identified local million-vector
input cache and the included projection basis. Its companion manifest records
the input, basis, shape, and build recipe.

Unless a row explicitly says otherwise, logged byte counts are application
payloads. The dedicated socket experiment additionally measures real TCP and
TLS 1.3 framing over persistent loopback connections. It is labeled loopback
throughout and is not presented as WAN/LAN latency.
