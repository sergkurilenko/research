"""Portable path resolution for the SHARD experiments.

The cached embeddings (~17 GB) are NOT in the repo. Either point the
environment variable SHARD_DATA at a local copy, or regenerate them and
place them under <repo>/data/corpus_cache (see docs/reproduce.md). Results
go to <repo>/results/ and figures to <repo>/paper/figs/ by default; override
with SHARD_RESULTS / SHARD_FIGS.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_data_candidates = [REPO_ROOT / "data" / "corpus_cache",
                    REPO_ROOT / "notebooks" / "_corpus_cache"]  # legacy dev location
_default_data = next((c for c in _data_candidates if c.exists()), _data_candidates[0])

DATA = Path(os.environ.get("SHARD_DATA", str(_default_data)))
RESULTS = Path(os.environ.get("SHARD_RESULTS", str(REPO_ROOT / "results")))
FIGS = Path(os.environ.get("SHARD_FIGS", str(REPO_ROOT / "paper" / "figs")))
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
