Sergey Kurilenko  
Moscow Institute of Physics and Technology (MIPT)  
Dolgoprudny 141701, Russia  
sergkurilenko@gmail.com

12 July 2026

To the Editor-in-Chief  
Journal of Information Security and Applications

Dear Editor,

I submit the manuscript “SHARD: cell-keyed residual splitting for
alignment-resistant private dense retrieval” for consideration as a research
article in the *Journal of Information Security and Applications*.

Dense embeddings are now routine in semantic search and
Retrieval-Augmented Generation, but a compromised vector store can expose
substantial document geometry. The manuscript asks whether cell-local
orthogonal transforms improve on a single globally shared secret rotation.
SHARD separates a short public routing prefix from a cell-keyed residual and
supports ciphertext--plaintext CKKS reranking with a correspondingly
transformed query.

The paper’s main value is a reproducible adversarial audit rather than a broad
privacy claim. It first corrects the scoring protocol: documents use centred
PCA coordinates, while scoring queries remain uncentred. The full score then
differs from the raw dot product only by a query-dependent constant and
reproduces raw ranking across ten BEIR and MIRACL configurations. A
rank-deficient alignment study shows that complete key identifiability is not a
de-anonymisation threshold: minimum-norm OLS obtains high residual-gallery R@1
from roughly 32--36 well-spread in-cell pairs. Increasing the cell count still
spreads diffuse evidence across compartments, but creates no hard threshold.

Independent releases expose the limit of re-keying. Residual norms, within-cell
Gram signatures, and the unchanged public prefix link almost every persistent
record. A partial-overlap extension shows that churn weakens Gram matching but
not the stable prefix or clean residual norm. The manuscript therefore rejects
unlinkability and cancellable-template interpretations and states the exposed
channels directly.

The expanded systems and privacy audit is equally concrete. Across 315 real
TenSEAL/SEAL runs, block-SIMD packing cuts query upload by 74--87% with no
top-1 flips, but increases in-process median latency because the response still
contains one ciphertext per candidate. A replacement-adjacency Gaussian
release is calibrated with an exact analytic accountant: strong privacy noise
destroys retrieval, while every strict SHARD-utility match on the evaluated
grid occurs at epsilon 32768 with native-gallery linkage R@1 at least 0.995.
These are bounded empirical and formal baselines, not a claim that SHARD is
differentially private.

These findings are relevant to JISA because they connect embedding privacy,
homomorphic query evaluation, attack-aware retrieval design, and careful threat
modelling. The artifact includes source code, exact configurations, per-seed
outputs, bootstrap intervals, complete run logs, a measured CKKS phase
breakdown, and a checkpoint-specific learned text-reconstruction outcome. It
separates the local in-process benchmark from unmeasured network, concurrency,
and access-pattern costs.

The earlier global-linear SVD/rotation/CKKS construction is disclosed and cited
as prior work. It appears only as the foil needed to motivate and evaluate
SHARD. The corrected SHARD analysis and Experiments 23--29 are new. The
manuscript is not under consideration elsewhere. I have no competing interests
to declare, and the work received no specific external funding.

Reproducibility materials are available in the `article5` branch of
https://github.com/sergkurilenko/research.

Thank you for considering this work.

Sincerely,

Sergey Kurilenko

