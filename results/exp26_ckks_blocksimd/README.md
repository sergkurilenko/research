# Experiment 26: measured CKKS block-SIMD reranking

This directory contains **measured Microsoft SEAL/TenSEAL CKKS runs**, not a
latency or ciphertext-size model.  The server context contains a public key and
Galois keys but no secret key.  Cached real SciFact document/query embeddings
from multilingual-e5-small and multilingual-e5-base are used throughout.

## Protocol

Documents use `(x-mu) @ V`; final query scoring uses `q @ V`, while shortlist
routing uses `(q-mu) @ V`.  The private residual occupies 3/4 of the embedding.
For width 1, each active cell-specific `H_c r_q` is encrypted separately.  For
width B, B such vectors are concatenated in one ciphertext.  For a candidate
in block b, the plaintext contains `H_c r_i` only in b and zeros in every other
block; a real ciphertext-plaintext `dot` therefore returns the residual score.
The public-prefix score is then added to that ciphertext before it is returned.

The online p50/p95/p99 measurements include client transformation, packing,
encryption and serialization; server query deserialization, mask construction,
ct-pt evaluation and response serialization; and client response deserialization
and decryption.  PCA, k-means, dense key generation, keyed-document preparation
and CKKS context/key generation are measured separately and excluded online.

## Main measured results

CKKS parameters: N=8192, slots=4096, 
coefficient bits=[60, 40, 40, 60], scale=2^40.
The provisioned public server context is 34.08 MB; 
it is a one-time setup object and is not counted as per-query traffic.

* `multilingual-e5-small`, 128 candidates: width 1 used a median of 
  30 query ciphertexts and 
  10.03 MB upload; its measured 
  end-to-end p50/p95/p99 was 2414.9/
  2504.0/2505.1 ms.
  Width 8 reduced query upload to 
  1.34 MB 
  (86.7% reduction) 
  and client encryption time by 87.6%, but its p50 
  end-to-end latency was 3049.3 ms 
  (+26.3%).  The encrypted scalar responses occupied 
  30.13 MB in either layout.
* `multilingual-e5-base`, 128 candidates: width 1 used a median of 
  31 query ciphertexts and 
  10.36 MB upload; its measured 
  end-to-end p50/p95/p99 was 2654.7/
  2740.8/2785.0 ms.
  Width 4 reduced query upload to 
  2.67 MB 
  (74.2% reduction) 
  and client encryption time by 74.4%, but its p50 
  end-to-end latency was 3040.4 ms 
  (+14.5%).  The encrypted scalar responses occupied 
  30.13 MB in either layout.

Across all measured trials, maximum absolute score error was 2.290e-06, 
minimum top-10 overlap was 1.000, total top-1 flips were 
0, and minimum Kendall tau was 0.999754.

See `summary.csv` for every encoder/candidate-count/width cell and
`raw_measurements.csv` for individual seed/repetition measurements.

## Interpretation and limitations

Block-SIMD here reduces the number of uploaded query ciphertexts.  Each candidate
still produces one encrypted scalar response, and TenSEAL's `dot` reduces the
entire logical packed vector.  Consequently, larger packing widths can make each
candidate evaluation slower even while query upload and encryption become smaller.
This experiment does not claim batched multi-score output, network RTT, TLS, ANN
index latency, multi-client concurrency, GPU acceleration, or production service
throughput.  Point-sampled RSS is not an exact high-frequency peak-memory trace.
TenSEAL/SEAL executes this CKKS path on the CPU; the installed RTX GPU is not used.
Trials were pinned to logical CPUs [0, 2, 4, 6, 8, 10] 
(one hardware thread per i5-14400F P-core) to prevent P/E-core migration;
the deterministic trial order was shuffled within each comparison cell.

## Reproduction

From the repository root:

```powershell
D:\PHD\research\RES\experiments\.venv\Scripts\python.exe shard\exp26_ckks_blocksimd.py --force
```

Dependencies added to that environment without upgrading other packages:

```powershell
python -m pip install --no-deps tenseal==0.3.16 psutil==7.2.2
```

Unit/algebra checks passed: `True`. Total wall time: 627.5 s.

## Files

* `config.json`: frozen protocol and CLI configuration.
* `run_info.json`: software/hardware, hashes and CKKS context metadata.
* `unit_checks.json`: orthogonality, keyed-dot and real encrypted-mask checks.
* `geometry.json`: dataset/PCA/cell preparation diagnostics.
* `raw_measurements.csv` / `.json`: every actual timed trial.
* `summary.csv` / `.json`: aggregate p50/p95/p99 and measured width-1 ratios.
* `run.log`: timestamped execution log.
