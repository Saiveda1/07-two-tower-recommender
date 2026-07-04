"""Leak-free temporal evaluation: split, per-user ground truth, ranking metrics.

The split is **temporal**, not random: interactions are ordered by timestamp and
the last ``test_frac`` of the horizon becomes the test set.  This mirrors how a
recommender is actually judged — trained on the past, scored on the future — and
is strictly harder than a random split because category popularity has drifted.

**No leakage** is guaranteed structurally: ``max(train.ts) < min(test.ts)``, all
popularity/recency statistics and the popularity baseline are computed from the
train slice only, and a user's train-seen items are excluded from their
recommendations (so a "relevant" item is a genuinely *new* future engagement).

Metrics: Recall@K, nDCG@K, HitRate@K, catalog coverage and novelty (mean
self-information of recommended items).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    """A temporal train/test split with per-user histories and ground truth."""

    train_users: np.ndarray
    train_items: np.ndarray
    train_ts: np.ndarray
    test_users: np.ndarray
    test_items: np.ndarray
    test_ts: np.ndarray
    threshold: float
    train_history: dict[int, set[int]]      # user -> items seen in train
    ground_truth: dict[int, set[int]]        # user -> NEW items engaged in test
    eval_users: np.ndarray                   # users with train history AND test positives

    def assert_no_leakage(self) -> None:
        """Structural guarantee that no test signal bled into train."""
        assert self.train_ts.size and self.test_ts.size, "empty split"
        assert self.train_ts.max() < self.test_ts.min(), "temporal overlap between train and test"
        for u in self.eval_users:
            # a relevant item is never something the user already saw in train
            assert not (self.ground_truth[int(u)] & self.train_history[int(u)])


def temporal_split(interactions: dict[str, np.ndarray], *, test_frac: float = 0.2) -> Split:
    """Split interactions in time; build train histories and future ground truth."""
    u = interactions["user_id"]
    i = interactions["item_id"]
    t = interactions["ts"]
    order = np.argsort(t, kind="stable")
    u, i, t = u[order], i[order], t[order]

    threshold = float(np.quantile(t, 1.0 - test_frac))
    is_train = t < threshold
    tr_u, tr_i, tr_t = u[is_train], i[is_train], t[is_train]
    te_u, te_i, te_t = u[~is_train], i[~is_train], t[~is_train]

    train_history: dict[int, set[int]] = {}
    for uu, ii in zip(tr_u.tolist(), tr_i.tolist()):
        train_history.setdefault(uu, set()).add(ii)

    ground_truth: dict[int, set[int]] = {}
    for uu, ii in zip(te_u.tolist(), te_i.tolist()):
        if uu in train_history and ii not in train_history[uu]:
            ground_truth.setdefault(uu, set()).add(ii)

    eval_users = np.array(sorted(uu for uu, g in ground_truth.items() if g), dtype=np.int64)
    return Split(
        train_users=tr_u, train_items=tr_i, train_ts=tr_t,
        test_users=te_u, test_items=te_i, test_ts=te_t,
        threshold=threshold, train_history=train_history,
        ground_truth=ground_truth, eval_users=eval_users,
    )


@dataclass
class TrainStats:
    """Per-item / per-user statistics computed from the TRAIN slice only."""

    item_count: np.ndarray       # interaction count per item
    item_prob: np.ndarray        # normalized popularity (sums to 1)
    item_mean_ts: np.ndarray     # mean timestamp (recency signal)
    item_selfinfo: np.ndarray    # -log2(prob): novelty weight
    user_count: np.ndarray       # activity per user
    user_mean_ts: np.ndarray     # user recency
    user_dom_cat: np.ndarray     # dominant category per user
    pop_ranking: np.ndarray      # items sorted most->least popular

    @classmethod
    def from_train(cls, split: Split, n_users: int, n_items: int, item_category: np.ndarray) -> "TrainStats":
        tr_u, tr_i, tr_t = split.train_users, split.train_items, split.train_ts
        item_count = np.bincount(tr_i, minlength=n_items).astype(np.float64)
        prob = item_count + 1.0
        prob = prob / prob.sum()
        item_selfinfo = -np.log2(prob)

        ts_sum = np.bincount(tr_i, weights=tr_t, minlength=n_items)
        item_mean_ts = np.divide(ts_sum, np.maximum(item_count, 1.0))

        user_count = np.bincount(tr_u, minlength=n_users).astype(np.float64)
        u_ts_sum = np.bincount(tr_u, weights=tr_t, minlength=n_users)
        user_mean_ts = np.divide(u_ts_sum, np.maximum(user_count, 1.0))

        # dominant category per user (argmax over per-user category histogram)
        cats = item_category[tr_i]
        n_cat = int(item_category.max()) + 1
        flat = tr_u.astype(np.int64) * n_cat + cats.astype(np.int64)
        hist = np.bincount(flat, minlength=n_users * n_cat).reshape(n_users, n_cat)
        user_dom_cat = hist.argmax(axis=1).astype(np.int16)

        pop_ranking = np.argsort(-item_count, kind="stable")
        return cls(
            item_count=item_count, item_prob=prob, item_mean_ts=item_mean_ts,
            item_selfinfo=item_selfinfo, user_count=user_count,
            user_mean_ts=user_mean_ts, user_dom_cat=user_dom_cat, pop_ranking=pop_ranking,
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _dcg(rec: list[int], relevant: set[int]) -> float:
    return sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(rec) if item in relevant)


def _idcg(n_relevant: int, k: int) -> float:
    return sum(1.0 / np.log2(rank + 2) for rank in range(min(n_relevant, k)))


@dataclass
class MetricResult:
    recall: float
    ndcg: float
    hit_rate: float
    coverage: float
    novelty: float
    k: int
    n_users: int

    def as_row(self, name: str) -> dict[str, object]:
        return {
            "method": name, "k": self.k, "recall": round(self.recall, 4),
            "ndcg": round(self.ndcg, 4), "hit_rate": round(self.hit_rate, 4),
            "coverage": round(self.coverage, 4), "novelty": round(self.novelty, 4),
            "n_users": self.n_users,
        }


def evaluate_recommendations(
    recommendations: dict[int, list[int]],
    split: Split,
    stats: TrainStats,
    *,
    k: int,
    n_items: int,
) -> MetricResult:
    """Score a ``user -> ranked item list`` mapping against future ground truth."""
    recalls, ndcgs, hits = [], [], []
    recommended_items: set[int] = set()
    novelty_acc: list[float] = []

    for u in split.eval_users:
        u = int(u)
        rec = recommendations.get(u, [])[:k]
        g = split.ground_truth[u]
        if not g:
            continue
        n_hit = len(set(rec) & g)
        recalls.append(n_hit / len(g))
        hits.append(1.0 if n_hit > 0 else 0.0)
        idcg = _idcg(len(g), k)
        ndcgs.append(_dcg(rec, g) / idcg if idcg > 0 else 0.0)
        recommended_items.update(rec)
        if rec:
            novelty_acc.append(float(np.mean(stats.item_selfinfo[rec])))

    return MetricResult(
        recall=float(np.mean(recalls)) if recalls else 0.0,
        ndcg=float(np.mean(ndcgs)) if ndcgs else 0.0,
        hit_rate=float(np.mean(hits)) if hits else 0.0,
        coverage=len(recommended_items) / n_items,
        novelty=float(np.mean(novelty_acc)) if novelty_acc else 0.0,
        k=k, n_users=len(recalls),
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def popularity_recommendations(split: Split, stats: TrainStats, *, k: int) -> dict[int, list[int]]:
    """Recommend the globally most-popular train items, minus each user's history."""
    ranked = stats.pop_ranking.tolist()
    out: dict[int, list[int]] = {}
    for u in split.eval_users:
        u = int(u)
        seen = split.train_history.get(u, set())
        rec: list[int] = []
        for item in ranked:
            if item not in seen:
                rec.append(item)
                if len(rec) >= k:
                    break
        out[u] = rec
    return out


def random_recommendations(split: Split, n_items: int, *, k: int, seed: int = 0) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    out: dict[int, list[int]] = {}
    for u in split.eval_users:
        u = int(u)
        seen = split.train_history.get(u, set())
        rec: list[int] = []
        while len(rec) < k:
            cand = int(rng.integers(0, n_items))
            if cand not in seen and cand not in rec:
                rec.append(cand)
        out[u] = rec
    return out
