# Security-expansion experiments

This addendum evaluates two boundaries of the revised protocol: semantic
information already exposed by the ordered public-PQ candidate identifiers,
and the cryptographic assumptions behind the exact CKKS parameter set.  It is
an audit of an authorised local prototype.  It does not probe an external
system, decrypt server traffic, or attempt to obtain a secret key.

## Candidate-ID semantic leakage

`system/candidate_id_leakage.py` assumes only what the protocol already gives
the provider: the ordered candidate identifiers and the provider-owned exact
vectors for those identifiers.  It deliberately does not use PQ distances,
CKKS ciphertext contents, plaintext scores, a decryptor, or a secret key.

The estimators separate set leakage from order leakage:

* `centroid` is set-only: it ignores the order of the exposed identifiers.
* `log_rank` uses the public ordering through a logarithmic rank discount.
* `ridge_rank_ls` fits the exact candidate vectors to fixed normal-score rank
  targets with a log-rank prior.  It never observes the actual PQ scores.

Direction cosine is measured against the true projected query.  Retrieval
leakage is the overlap between exhaustive top-10/top-100 results obtained with
the true and reconstructed directions.  The topic proxy is hit rate and
reciprocal rank for the held-out self-document (million-vector track) or
positive qrels (BEIR).  Linkability uses two non-overlapping views formed from
alternating candidate ranks; positive pairs belong to the same query and
negative pairs are deterministically reassigned to other queries.

The primary million-vector command uses all 400 held-out test queries:

```powershell
python -m system.candidate_id_leakage million `
  --projected-docs results\system_revision\cache\E_docs_e5base_proj672_1000000.npy `
  --queries cache\million_vector\E_queries_e5base_self_q500.npy `
  --basis results\system_revision\cache\e5base_k672_basis.npz `
  --pq-index cache\million_vector\index_cache\v2_indexpq_e5base_proj672_M96_b8.faiss `
  --split test --validation-size 100 --split-seed 2026 --qrel-seed 42 `
  --max-queries 400 --k 20 50 100 200 --exact-backend cuda `
  --output results\system_revision\security_expansion\candidate_id_million.json `
  --per-query-output results\system_revision\security_expansion\candidate_id_million.jsonl
```

BEIR evaluations must use only the immutable frozen revision-subset ID files produced by
the IR expansion, never the validation or full official-test rows.  The primary
run uses every available revision-subset query:

```powershell
python -m system.candidate_id_leakage beir `
  --graded-summary results\system_revision\graded_ir_official_test.json `
  --confirmatory-split-dir results\system_revision\ir_expansion\splits `
  --dataset arguana fiqa nfcorpus scidocs scifact trec-covid `
  --k 20 50 100 200 --exact-backend cuda `
  --output results\system_revision\security_expansion\candidate_id_beir_confirmatory.json `
  --per-query-output results\system_revision\security_expansion\candidate_id_beir_confirmatory.jsonl
```

The JSON records the confirmatory split path and SHA-256 for every collection,
including any canonical query ID unavailable in the cached evaluable ArguAna
subset.

The run completed over 400 held-out million-vector queries and 3,230 evaluable
strict-confirmatory BEIR queries.  ArguAna contributed 1,119 of 1,124 requested
IDs; the five canonical qrels whose positive document is absent from the cache
remain listed in the JSON rather than being silently discarded.  The other
query counts were FiQA 648, NFCorpus 323, SciDocs 800, SciFact 300, and
TREC-COVID 40.

At the deployed `K=100`, the million-vector set-only centroid reconstructed the
query with mean cosine 0.3897; the order-aware log-rank estimate reached 0.4001.
Their mean exhaustive top-10 overlaps were 0.2130 and 0.2755.  The log-rank
estimate retrieved the held-out self-document in its top 10 for 52.0% of
queries, against 92.75% for the true projected query.  Linkability AUC from two
non-overlapping candidate-ID views was 0.999997 and 0.999996, respectively.

The smallest candidate list was usually the most revealing for exact top-10
replication: million-vector log-rank overlap was 0.3925 at `K=20`, 0.3248 at
`K=50`, 0.2755 at `K=100`, and 0.2243 at `K=200`.  Larger lists make the
centroid less query-specific even though their split-view linkability remains
near perfect.

The same signal appears in every confirmatory BEIR collection.  For log-rank
at `K=20`, mean direction cosine / top-10 overlap / linkability AUC were:

| Collection | Cosine | Top-10 overlap | Linkability AUC |
|---|---:|---:|---:|
| ArguAna | 0.3078 | 0.5985 | 0.9962 |
| FiQA | 0.3366 | 0.4015 | 0.9979 |
| NFCorpus | 0.2813 | 0.5285 | 0.9910 |
| SciDocs | 0.3397 | 0.4885 | 0.9953 |
| SciFact | 0.2940 | 0.5530 | 0.9936 |
| TREC-COVID | 0.3377 | 0.4125 | 0.9993 |

The predeclared ridge rank least-squares estimator is retained despite its
lower direction cosine (0.177--0.193 on the million-vector track).  This is not
a selected-winner analysis: ridge sometimes reproduces local retrieval better
despite its poor global direction, for example top-10 overlap 0.6134 on
ArguAna at `K=100` and 0.5473 on SciFact at `K=200`.  The mixed result helps
separate direction recovery from neighbourhood/ranking leakage.

## Exact CKKS parameter transcript

The prototype uses Microsoft SEAL through TenSEAL 0.3.16.  Calling the same
`CoeffModulus.Create` API as the implementation gives the exact key-context
primes

```text
1152921504606748673
1099511480321
1152921504606830593
```

Their product is
`1461501441329443417981674104123436284398337261569`, or 159.9999998065
bits.  The key context therefore has 160 coefficient-modulus bits; the first
and last data levels have 100 and 60 bits.  SEAL's `tc128` guard permits at
most 218 bits for `N=8192`, leaving 58 bits of table headroom.  The scale is
`2^40` and there are 4096 complex slots.

The SEAL 4.1.1 source samples independent uniform ternary secret coefficients.
Its default error sampler is an exact centred binomial with 21 Bernoulli bits
per side, support `[-21,21]`, and standard deviation `sqrt(21/2) = 3.24037`;
SEAL retains 3.2 as its nominal configured standard deviation.  The archived
estimator input therefore evaluates both `CenteredBinomial(21)` and a
`DiscreteGaussian(3.2)` sensitivity model, with uniform ternary secret and
unbounded samples.

The input is pinned to malb/lattice-estimator commit
`3e48ef421ec256afddb3e7d2249a77eab6e9ba12`.  Generate the exact parameter
record and Sage input with:

```powershell
python -m system.ckks_security_transcript `
  --estimator-dir tmp\lattice-estimator `
  --output-dir results\system_revision\security_expansion\lwe_security `
  --profile rough
```

Docker execution was attempted first, but the local Docker backend reported a
prior containerd ingest I/O failure and became unresponsive.  No user images or
volumes were pruned or relocated.  The estimator was therefore executed in an
official Ubuntu 22.04 WSL distribution without changing the default
distribution.
Ubuntu's pinned `sagemath=9.5-4` package matches the Sage 9.5 base named by the
estimator's own development Dockerfile. Estimator inputs and raw attack
stdout/stderr are archived; machine-install transcripts are not part of the
scientific result bundle.

The executed primary command was:

```powershell
python -m system.ckks_security_transcript `
  --estimator-dir tmp\lattice-estimator `
  --output-dir results\system_revision\security_expansion\lwe_security `
  --profile rough --run-estimator --execution-backend wsl `
  --wsl-distro CodexSageEstimatorUbuntu --estimator-timeout-seconds 3600
```

The upstream rough batch remained CPU-bound in the exact bounded-CBD model and
hit the predeclared 3600-second hard timeout before returning any attack cost.
The authoritative status is `timed_out_partial_output_archived`, exit code 124,
elapsed time 3600.068 seconds.  Its ten-line stdout contains the exact
parameters and `MODEL_BEGIN seal_exact_cbd21`, but no completed attack line;
therefore it provides **no numerical security-bit estimate** and must not be
quoted as if it did.  The stdout SHA-256 is
`dfa27318aa31994b37befc1a913916ac4d0fc7552279099ef79757decd3632cc`.
The stderr SHA-256 is
`eff53f45a7f54eb3a6641b4089c6930da26174040c718b62b9503ba51275c4db`.

The duration is explained by upstream behaviour rather than a stalled process:
with bounded `CenteredBinomial(21)` and conservative unbounded samples, rough
adds Arora--GB to uSVP and dual-hybrid, then emits result lines only after the
whole batch returns.  At the final diagnostic the Sage process remained in
Linux state `R+` at about 108% CPU and 2.32 GiB RSS.  Individual attack
transcripts are consequently retained alongside, with one process/timeout per
attack and atomic resume-aware checkpoints; they supplement rather than erase
the failed primary batch.

The individually instrumented rough-model attacks use exactly the same
ADPS16 reduction-cost model and GSA shape assumption as upstream rough.  Both
noise models completed with the following costs:

| Rough attack | Exact CBD21 | Gaussian 3.2 sensitivity |
|---|---:|---:|
| uSVP | `2^147.8` | `2^147.8` |
| plain dual | `2^148.3` | `2^148.3` |
| dual-hybrid | `2^146.9` | `2^146.9` |

The smallest completed rough-model estimate is therefore `2^146.9`.  Exact
CBD21 Arora--GB was run in its own process, reached its predeclared
1800-second timeout (1800.060 seconds observed), and returned no cost.  It was
not rerun.  Thus `2^146.9` is the minimum of the completed core lattice attacks,
not a claim that every supported estimator attack completed.

For sensitivity to the estimator's default cost model, all non-Arora extended
attacks were run separately under MATZOV/GSA with an overall 7200-second budget
and 900 seconds per point:

| Extended/default attack | Exact CBD21 | Gaussian 3.2 sensitivity |
|---|---:|---:|
| uSVP | `2^176.3` | `2^176.3` |
| plain dual | `2^177.8` | `2^177.8` |
| dual-hybrid | `2^176.3` | `2^176.3` |
| BDD | `2^175.8` | `2^175.8` |
| BDD hybrid | `2^175.8` | `2^175.8` |
| BDD MITM hybrid | `2^247.6` | `2^247.0` |
| coded BKW | timeout, no cost | timeout, no cost |

The extended session completed in 2363 seconds, below its overall budget.
The two BKW processes timed out independently after 900.058 and 900.057
seconds.  The smallest completed default-model estimate is `2^175.8` for both
noise models.  These numbers must not be combined numerically with the
ADPS16/GSA rough exponents: they encode different lattice-reduction cost
models.  The conservative statement supported by the archive is that the exact
parameter set passes SEAL's `tc128` modulus guard and the completed core
lattice attacks exceed 128 bits under both stated models; the archive does not
establish an exhaustive all-attack minimum because Arora--GB/BKW did not
complete.

Reproduce the individually durable points with:

```powershell
python -m system.ckks_security_individual `
  --estimator-dir tmp\lattice-estimator `
  --output-dir results\system_revision\security_expansion\lwe_security `
  --wsl-distro CodexSageEstimatorUbuntu --timeout-per-attack-seconds 1800

python -m system.ckks_security_extended `
  --estimator-dir tmp\lattice-estimator `
  --output-dir results\system_revision\security_expansion\lwe_security `
  --wsl-distro CodexSageEstimatorUbuntu `
  --timeout-per-attack-seconds 900 --overall-timeout-seconds 7200
```

Both runners checkpoint through temporary-file `fsync` plus atomic replace and
skip every recorded status on resume.  A timed-out point is preserved unless a
caller explicitly requests its rerun; the reported runs did not request one.
The consolidated machine-readable record is
`results/system_revision/security_expansion/security_summary.json`.

An LWE cost estimate does not validate the circular/KDM assumption introduced
by public Galois key-switching material.  The prototype supplies only the ten
rotation keys required for dimensions up to 1024 and does not supply
relinearisation keys, but the remaining evaluation keys are still
encryptions/functions of the secret key under the usual circular-security
assumption.

## Circuit-privacy boundary

`system/circuit_privacy_audit.py` is a non-extractive diagnostic.  It replays a
fixed encrypted request and compares it with fresh client encryptions while
recording the implemented evaluator operations.  Run it with:

```powershell
python -m system.circuit_privacy_audit `
  --dimension 32 --candidates 8 --repeats 20 --seed 2026 `
  --output results\system_revision\security_expansion\circuit_privacy_boundary.json
```

The server performs deterministic ciphertext-plaintext multiplication,
rescaling, rotations, additions, masking, and serialization.  There is no
fresh encryption of zero, output re-randomisation, bootstrapping,
sanitisation, or calibrated noise flooding.  Consequently, the manuscript can
claim numerical query-slot confidentiality only under its restricted
honest-client model; it cannot claim provider circuit/operand privacy,
q-IND-CPA-D/IND-CPA-D security, or malicious-client security.  Diversity of
responses after fresh client encryption and small decoding error are
correctness observations, not a circuit-privacy proof.

Empirically, twenty replays of one fixed encrypted request produced one unique
serialized response hash, whereas twenty fresh encryptions of the same query
produced twenty unique response hashes.  The maximum observed score error was
`2.97e-5`.  Source inspection found the expected multiplication, rescaling,
rotation and addition operations, but no output sanitisation.  These facts
support the boundary statement above; they do not substitute for a security
proof.

Naive noise flooding is intentionally not added.  Its required magnitude is a
formal worst-case parameter question, and implementation-specific or
average-case flooding can be unsafe.  Relevant primary sources are:

* Li and Micciancio, [On the Security of Homomorphic Encryption on Approximate
  Numbers](https://eprint.iacr.org/2020/1533).
* Kluczniak and Schild, [Circuit Privacy for FHEW/TFHE-Style Fully Homomorphic
  Encryption in Practice](https://eprint.iacr.org/2022/1459).
* Döttling and Dujmovic, [Maliciously Circuit-Private FHE from
  Information-Theoretic Principles](https://eprint.iacr.org/2022/495).
* Albrecht et al., [Homomorphic Encryption Security
  Standard](https://homomorphicencryption.org/standard/).

## Tests

```powershell
python -m pytest `
  tests\test_candidate_id_leakage.py `
  tests\test_ckks_security_transcript.py `
  tests\test_circuit_privacy_audit.py -q
```

All result files belong under
`results/system_revision/security_expansion/`.  The raw JSON/JSONL and Sage
transcripts, rather than rounded manuscript values, are the authoritative
records.
