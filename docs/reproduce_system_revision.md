# Reproducing the journal revision

This guide routes the complete artifact for **Hybrid privacy-aware semantic
search: SVD-truncated document geometry and CKKS-encrypted query reranking
under a restricted threat model**. It does not reproduce the superseded
random-rotation study retained under `paper/` and `baseline/`.

The implemented boundary is intentionally narrow. The client searches a
public SVD/PQ artifact and discloses candidate IDs. CKKS protects the numerical
query slots from a public-only honest-but-curious provider process, conditional
on that disclosed transcript. The artifact does not claim semantic-query,
document, database, access-pattern, circuit, or malicious-client privacy.

## 1. Environment and first checks

The measured Windows host has an Intel Core i5-14400F, 32 GB RAM, and an
NVIDIA RTX 5060 with 8 GB. CKKS is CPU-only; CUDA is used for E5 encoding and
exact-reference retrieval. Set the interpreter for the reproduction environment:

```powershell
$py = 'python'
```

The result manifests pin Python, TenSEAL/Microsoft SEAL, NumPy, Faiss,
PyTorch/CUDA, Transformers, Datasets, and scikit-learn versions. Start with:

```powershell
cd <repository-root>
& $py -m pytest tests -q
& $py -m system.validate_systems_expansion `
  --input results\system_revision\systems_expansion\windows_full.json `
  --output results\system_revision\systems_expansion\windows_full_validation.json
```

`docs/results_manifest.md` maps manuscript claims to authoritative JSON/JSONL
files. Never copy rounded values out of the PDF when a machine-readable record
is present.

## 2. Data and cache boundary

Six graded collections are fetched from the pinned BEIR/Hugging Face sources
recorded in `graded_ir_official_test.json`. The E5 model revision, literal
`query:`/`passage:` prefixes, pooling, normalization, maximum length, corpus
and qrel hashes, and all cache keys are frozen in the result files.

The million-vector stress track uses local derived caches that are too large
for the archive:

```text
cache\million_vector\E_docs_e5base_1000000.npy
cache\million_vector\E_queries_e5base_self_q500.npy
cache\million_vector\index_cache\v2_indexpq_e5base_proj672_M96_b8.faiss
```

It is a scale/self-retrieval diagnostic, not a graded relevance benchmark.
The public projection basis and rebuild manifest are included. Rebuild the
provider memmap with:

```powershell
& $py system\build_projected_cache.py `
  --input cache\million_vector\E_docs_e5base_1000000.npy `
  --basis results\system_revision\cache\e5base_k672_basis.npz `
  --output results\system_revision\cache\E_docs_e5base_proj672_1000000.npy `
  --backend cuda --chunk-size 8192
```

Documents use `(d - passage_mean) @ V`; online queries use `q @ V`. No test
query statistic enters projection fitting or PQ training.

## 3. Base kernel and request-path evidence

The three CKKS kernels share degree 8192, coefficient-modulus bit sizes
`[60,40,60]`, scale `2^40`, and the same selected Galois keys:

```powershell
& $py system\packed_ckks.py `
  --dimension 672 --candidates 100 --repeats 20 --warmups 2 `
  --output results\system_revision\ckks_micro_d672_K100.json

& $py system\run_ckks_sweep.py `
  --repeats 7 --warmups 2 `
  --output results\system_revision\ckks_sweep.json
```

Run Windows spawned-process experiments as modules so that `spawn` can import
the package. The exact complete commands, input paths, and hashes are embedded
under `command`/`config` in their result JSONs. The relevant harnesses are:

```text
system.unified_ckks_bench
system.vector_return_baselines
system.pq_scaling_bench
system.pq_leakage_audit
system.score_oracle_extraction
system.candidate_id_leakage
system.circuit_privacy_audit
```

The provider worker receives a serialized public context, a read-only exact
matrix path, an encrypted query, and uint64 candidate IDs. Its attestation is
a structural negative-capability test (`has_secret_key=false`, no decryptor),
not remote attestation or host-compromise protection.

## 4. Frozen revision analysis and IR/PQ controls

The revision analysis rule was written before calculating the frozen subsets
but after the earlier exploratory full-test analysis; it is not presented as a
prospective preregistration. The primary paired endpoint is CKKS minus
plaintext-shortlist nDCG@10, with 10,000 bootstrap samples, seed 2026, and a
symmetric margin of 0.002. Non-inferiority requires the lower endpoint to be
greater than -0.002. Equivalence requires the *entire* interval inside
[-0.002, 0.002]; merely covering zero is not equivalence.

The split ID files and hashes are immutable. ArguAna retains five queries whose
positive documents are absent from the pinned corpus as zero rows in the
canonical denominator; the 1,119-evaluable sensitivity result is separate.

Run the expansion stages only when no other performance experiment is active:

```powershell
& $py -m pytest tests\test_ir_expansion.py tests\test_graded_ir_bench.py -q
& $py system\ir_expansion.py splits
& $py system\ir_expansion.py controls
& $py system\ir_expansion.py pareto
& $py system\ir_expansion.py pq
```

The Pareto and PQ stages checkpoint atomically after every completed point and
verify cached Faiss files by metadata and hash. Details of dimensions,
M/K/seeds, validation separation, and output schemas are in
`docs/ir_expansion.md`.

## 5. Provider scaling and transport

The controlled provider run fixes OpenMP/BLAS-family thread pools to one and
uses one independently spawned public-only worker per closed-loop client. It
measures packed concurrency 1, 2, 4, 8, and 16; naive controls at 1 and 8;
cold/warm sessions; key-material amortization; process RSS/CPU; and persistent
TCP and TLS 1.3 loopback sockets. Encryption/decryption are excluded only from
the saturation timer and remain present in the full-client session records.

The full command and interpretation limits are in
`docs/systems_expansion.md`. `windows_full.json` contains every condition and
restart, while compact CSV/JSON summaries are derived from it. The same-CPU
Windows/WSL microbenchmark is a portability diagnostic, not independent
hardware replication.

Loopback is a real serialized socket implementation but not a physical
network. RTT, loss, congestion, certificate deployment, and final payload
fetch remain outside the measurement. The separate ideal-link calculation is
explicitly a planning model.

## 6. CKKS parameter and security transcript

`system.ckks_security_transcript` extracts the exact three SEAL primes through
the implementation API, records the binary/source hashes, secret/error
distributions, evaluation-key assumptions, and generates Sage input for a
pinned malb/lattice-estimator commit. The primary input uses uniform ternary
secret, exact `CenteredBinomial(21)` error, and an unbounded-sample conservative
sensitivity model; a `DiscreteGaussian(3.2)` model is retained separately.

The Sage runtime and raw output are archived under:

```text
results/system_revision/security_expansion/lwe_security/
```

If the upstream batch reaches its timeout before emitting every attack,
`system.ckks_security_individual` runs the same rough-cost attacks in isolated
Sage processes with atomic, resume-aware checkpoints and preserves timed-out
points. It never deletes or relabels the primary batch record. See
`docs/security_expansion.md` for commands and the circuit-privacy boundary.

The estimator supplies cost estimates, not a proof of the ring reduction,
circular/KDM security of Galois material, circuit privacy, malicious-client
security, or implementation side-channel resistance.

## 7. Figures and manuscript

All charts are regenerated from result JSON, not manually entered numbers:

```powershell
& $py system\make_revision_figures.py
& $py system\make_expansion_figures.py
```

Build the Springer Nature source without requiring Perl/`latexmk`:

```powershell
cd paper_revised
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The final workflow renders every page to PNG, inspects the pages visually,
checks fonts and metadata, compiles the flat source archive in an independent
temporary directory, and then builds the online-resource ZIP. Generated PDFs
and delivery archives are not authoritative substitutes for source/result
files and their hashes.
