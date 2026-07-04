"""Two-Tower Recommender — retrieval + ranking at scale.

An end-to-end, offline, deterministic recommender system:

* ``data``     — streaming implicit-feedback interaction log generator (Zipf
                 popularity, category affinity, temporal drift), scalable to 1B
                 rows with bounded memory.
* ``model``    — a shallow **two-tower** retrieval model trained from scratch in
                 NumPy with in-batch softmax negatives and a logQ popularity
                 correction. Produces user + item embeddings.
* ``ann``      — approximate nearest-neighbour candidate retrieval over item
                 embeddings (brute-force + a real IVF index) for top-K per user.
* ``ranker``   — a gradient-boosted ranker (scikit-learn) that reorders
                 candidates using embedding, popularity and recency features.
* ``evaluate`` — a leak-free temporal split with Recall@K, nDCG@K, HitRate,
                 catalog coverage and novelty, versus popularity + random.

Everything is seeded and runs with zero network / GPU / paid APIs.
"""
from __future__ import annotations

__version__ = "1.0.0"

SEED = 42
