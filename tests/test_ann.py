"""ANN indexes: exactness of brute force and high recall of IVF."""
from __future__ import annotations

import numpy as np

from twotower.ann import BruteForceIndex, IVFIndex


def _vectors(n=1000, d=32, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_bruteforce_matches_numpy():
    v = _vectors()
    q = v[:20]
    idx, sc = BruteForceIndex(v).search(q, 5)
    # the top-1 for each query embedding is itself (cosine 1.0)
    assert np.array_equal(idx[:, 0], np.arange(20))
    assert np.allclose(sc[:, 0], 1.0, atol=1e-4)


def test_topk_is_sorted_descending():
    v = _vectors(seed=2)
    _, sc = BruteForceIndex(v).search(v[:10], 8)
    assert np.all(np.diff(sc, axis=1) <= 1e-5)


def test_ivf_high_recall():
    # worst case: unclustered random vectors (real item embeddings cluster by
    # category and retrieve even better — see the pipeline's ann_recall).
    v = _vectors(n=2000, seed=1)
    q = _vectors(n=200, seed=99)
    ivf = IVFIndex(nlist=32, nprobe=16).build(v)
    recall = ivf.recall_vs_exact(q, k=10)
    assert recall > 0.85, f"IVF recall too low: {recall:.3f}"


def test_ivf_more_probes_more_recall():
    v = _vectors(n=2000, seed=3)
    q = _vectors(n=150, seed=7)
    lo = IVFIndex(nlist=32, nprobe=2).build(v).recall_vs_exact(q, 10)
    hi = IVFIndex(nlist=32, nprobe=16).build(v).recall_vs_exact(q, 10)
    assert hi >= lo
