"""Approximate nearest-neighbour retrieval over item embeddings.

At serving time the item tower is frozen and its vectors indexed once; each user
vector then fetches its top-K candidates.  Two indexes are provided behind a
common ``search`` API:

* :class:`BruteForceIndex` — exact maximum-inner-product search (a single
  ``Q Vᵀ`` matmul + ``argpartition``).  Ground truth / small catalogues.
* :class:`IVFIndex` — an inverted-file index: KMeans partitions the catalogue
  into ``nlist`` Voronoi cells, and a query probes only its ``nprobe`` nearest
  cells.  This is the coarse-quantiser half of FAISS's ``IVF*`` family and is
  how the design scales to a 1B-item catalogue (see ARCHITECTURE.md).

Vectors are L2-normalized, so inner product ranks identically to cosine
similarity.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from . import SEED


def _topk_from_scores(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k indices (descending score) per row of a (Q, N) score matrix."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row, part], axis=1)
    idx = part[row, order]
    return idx, scores[row, idx]


class BruteForceIndex:
    """Exact maximum-inner-product search."""

    def __init__(self, item_vectors: np.ndarray) -> None:
        self.items = np.asarray(item_vectors, dtype=np.float32)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(queries, dtype=np.float32) @ self.items.T
        return _topk_from_scores(scores, k)


class IVFIndex:
    """Inverted-file (coarse-quantiser) approximate index."""

    def __init__(self, nlist: int = 64, nprobe: int = 8, seed: int = SEED) -> None:
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed
        self._centroids: np.ndarray | None = None
        self._lists: list[np.ndarray] = []
        self._items: np.ndarray | None = None

    def build(self, item_vectors: np.ndarray) -> "IVFIndex":
        self._items = np.asarray(item_vectors, dtype=np.float32)
        n = self._items.shape[0]
        nlist = int(min(self.nlist, max(1, n)))
        km = KMeans(n_clusters=nlist, random_state=self.seed, n_init=4)
        assign = km.fit_predict(self._items)
        self._centroids = km.cluster_centers_.astype(np.float32)
        self._lists = [np.where(assign == c)[0] for c in range(nlist)]
        return self

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self._centroids is not None and self._items is not None, "index not built"
        queries = np.asarray(queries, dtype=np.float32)
        nprobe = int(min(self.nprobe, self._centroids.shape[0]))

        # nearest cells per query (by centroid inner product)
        cell_scores = queries @ self._centroids.T
        probe = np.argpartition(-cell_scores, kth=nprobe - 1, axis=1)[:, :nprobe]

        k_out = min(k, self._items.shape[0])
        out_idx = np.full((queries.shape[0], k_out), -1, dtype=np.int64)
        out_score = np.full((queries.shape[0], k_out), -np.inf, dtype=np.float32)
        for qi in range(queries.shape[0]):
            cand = np.concatenate([self._lists[c] for c in probe[qi]]) if nprobe else np.empty(0, np.int64)
            if cand.size == 0:
                continue
            s = self._items[cand] @ queries[qi]
            kk = min(k_out, cand.size)
            top = np.argpartition(-s, kth=kk - 1)[:kk]
            top = top[np.argsort(-s[top])]
            out_idx[qi, :kk] = cand[top]
            out_score[qi, :kk] = s[top]
        return out_idx, out_score

    def recall_vs_exact(self, queries: np.ndarray, k: int) -> float:
        """Fraction of exact top-k neighbours the IVF search recovers."""
        approx, _ = self.search(queries, k)
        exact, _ = BruteForceIndex(self._items).search(queries, k)
        hits = 0
        for a, e in zip(approx, exact):
            hits += len(set(a.tolist()) & set(e.tolist()))
        return hits / (queries.shape[0] * k)
