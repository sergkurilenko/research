# Third-party data and software notices

The repository's original experiment code and documentation are distributed
under the root `LICENSE`. That license does not replace the licenses of the
datasets, model weights, libraries, or Springer template used by the study.

The online resource does not redistribute raw BEIR corpora, Hugging Face
snapshots, E5 model weights, the million-row Wikipedia embedding cache, or
trained Faiss indexes whose source licenses or size make redistribution
inappropriate. Result manifests record their source identifiers, pinned
revisions or content hashes, preprocessing recipes, and rebuild commands.

The experiments use, among other dependencies, Microsoft SEAL/TenSEAL, Faiss,
PyTorch, Transformers, Datasets, NumPy, scikit-learn, SageMath, and malb's
lattice-estimator. Each remains governed by its upstream license. The pinned
Microsoft SEAL and lattice-estimator source checkouts are not copied into the
journal ZIP; only commit identifiers, small generated inputs, raw transcripts,
and hashes needed to audit the reported results are included.

The Springer Nature `sn-jnl.cls` and `sn-basic.bst` files are included only in
the flat submission-source package to make that package compile independently.
Their use and redistribution remain subject to Springer's template terms.
