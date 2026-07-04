"""Second-stage ranker: a gradient-boosted model over candidate features.

Retrieval (the two-tower + ANN) is optimised for *recall* — cheaply surfacing a
few hundred plausible items.  The ranker then trades a little latency for
*precision*, reordering that shortlist with signals the dot-product alone cannot
see: raw popularity, item/user recency, activity, and category match.

Training pairs come from the **train slice only**: observed interactions are
positives; items the user did not touch are sampled as negatives.  A
:class:`sklearn.ensemble.HistGradientBoostingClassifier` learns
P(engage | features); its score reorders the retrieval shortlist at serving
time.  Because it exploits popularity/recency that the frozen embeddings ignore,
it lifts nDCG over the retrieval-only ordering — which the test-suite asserts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .evaluate import Split, TrainStats
from .model import TwoTowerModel

FEATURE_NAMES = [
    "dot", "cosine", "item_logpop", "item_mean_ts",
    "user_logact", "user_mean_ts", "cat_match", "u_item_norm",
]


def _build_features(
    users: np.ndarray,
    items: np.ndarray,
    model: TwoTowerModel,
    stats: TrainStats,
    item_category: np.ndarray,
    item_norm: np.ndarray,
) -> np.ndarray:
    U = model.user_emb[users]
    V = model.item_emb[items]
    dot = np.einsum("ij,ij->i", U, V) + model.serving_bias()[items]  # full serving score
    un = np.linalg.norm(U, axis=1)
    vn = np.linalg.norm(V, axis=1)
    cosine = dot / np.clip(un * vn, 1e-8, None)
    cat_match = (item_category[items] == stats.user_dom_cat[users]).astype(np.float32)
    feats = np.column_stack(
        [
            dot,
            cosine,
            np.log1p(stats.item_count[items]),
            stats.item_mean_ts[items],
            np.log1p(stats.user_count[users]),
            stats.user_mean_ts[users],
            cat_match,
            vn,  # item embedding norm (confidence / frequency proxy)
        ]
    ).astype(np.float32)
    return feats


@dataclass
class RankerConfig:
    negatives_per_positive: int = 4
    max_positives: int = 200_000
    max_iter: int = 200
    learning_rate: float = 0.1
    max_depth: int = 6
    n_threads: int = 2      # cap GBM OpenMP threads (avoids oversubscription thrash)
    seed: int = 42


class GBMRanker:
    """Gradient-boosted candidate ranker."""

    def __init__(self, cfg: RankerConfig | None = None) -> None:
        self.cfg = cfg or RankerConfig()
        self.clf = HistGradientBoostingClassifier(
            max_iter=self.cfg.max_iter,
            learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth,
            random_state=self.cfg.seed,
            early_stopping=False,
        )
        self._item_norm: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(
        self,
        split: Split,
        model: TwoTowerModel,
        stats: TrainStats,
        item_category: np.ndarray,
        n_items: int,
    ) -> "GBMRanker":
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        self._item_norm = np.linalg.norm(model.item_emb, axis=1)

        pos_u, pos_i = split.train_users, split.train_items
        if pos_u.shape[0] > cfg.max_positives:
            sel = rng.choice(pos_u.shape[0], size=cfg.max_positives, replace=False)
            pos_u, pos_i = pos_u[sel], pos_i[sel]

        # Uniform negative sampling. Popularity-proportional negatives would label
        # popular items "negative" far more often than rare ones, teaching the
        # ranker that popularity is anti-predictive -> it then pushes genuinely
        # relevant (moderately popular, in-category) items DOWN the shortlist and
        # scores below even a popularity baseline. Uniform negatives keep the
        # popularity feature honest; the model still learns popularity's true sign.
        n_neg = pos_u.shape[0] * cfg.negatives_per_positive
        neg_u = np.repeat(pos_u, cfg.negatives_per_positive)
        neg_i = rng.integers(0, n_items, size=n_neg)

        users = np.concatenate([pos_u, neg_u])
        items = np.concatenate([pos_i, neg_i])
        labels = np.concatenate([np.ones(pos_u.shape[0]), np.zeros(n_neg)]).astype(np.int32)

        X = _build_features(users, items, model, stats, item_category, self._item_norm)
        # Cap OpenMP threads: on an oversubscribed box, all-core GBM thrashes.
        with threadpool_limits(limits=self.cfg.n_threads):
            self.clf.fit(X, labels)
        return self

    # ------------------------------------------------------------------
    def rerank(
        self,
        user: int,
        candidates: list[int],
        model: TwoTowerModel,
        stats: TrainStats,
        item_category: np.ndarray,
        *,
        k: int,
    ) -> list[int]:
        if not candidates:
            return []
        assert self._item_norm is not None, "ranker not fitted"
        users = np.full(len(candidates), user, dtype=np.int64)
        items = np.asarray(candidates, dtype=np.int64)
        X = _build_features(users, items, model, stats, item_category, self._item_norm)
        scores = self.clf.predict_proba(X)[:, 1]
        order = np.argsort(-scores)
        return [candidates[j] for j in order[:k]]

    def rerank_batch(
        self,
        pools: dict[int, list[int]],
        model: TwoTowerModel,
        stats: TrainStats,
        item_category: np.ndarray,
        *,
        k: int,
    ) -> dict[int, list[int]]:
        """Rerank every user's shortlist with a single batched scoring pass."""
        assert self._item_norm is not None, "ranker not fitted"
        users_all, items_all, owners = [], [], []
        for u, cands in pools.items():
            if not cands:
                continue
            users_all.extend([u] * len(cands))
            items_all.extend(cands)
            owners.append((u, len(cands)))
        out: dict[int, list[int]] = {u: [] for u in pools}
        if not items_all:
            return out
        X = _build_features(
            np.asarray(users_all, dtype=np.int64), np.asarray(items_all, dtype=np.int64),
            model, stats, item_category, self._item_norm,
        )
        with threadpool_limits(limits=self.cfg.n_threads):
            scores = self.clf.predict_proba(X)[:, 1]
        pos = 0
        for u, n in owners:
            s = scores[pos : pos + n]
            cands = pools[u]
            order = np.argsort(-s)
            out[u] = [cands[j] for j in order[:k]]
            pos += n
        return out

    def feature_importance(self, split: Split, model: TwoTowerModel, stats: TrainStats,
                           item_category: np.ndarray, n_items: int) -> dict[str, float]:
        """Permutation importance of each feature (drop in AUC when shuffled)."""
        from sklearn.inspection import permutation_importance

        rng = np.random.default_rng(self.cfg.seed + 3)
        m = min(20_000, split.train_users.shape[0])
        sel = rng.choice(split.train_users.shape[0], size=m, replace=False)
        pos_u, pos_i = split.train_users[sel], split.train_items[sel]
        neg_i = rng.choice(n_items, size=m, p=stats.item_prob)
        users = np.concatenate([pos_u, pos_u])
        items = np.concatenate([pos_i, neg_i])
        y = np.concatenate([np.ones(m), np.zeros(m)]).astype(np.int32)
        X = _build_features(users, items, model, stats, item_category, self._item_norm)
        r = permutation_importance(self.clf, X, y, n_repeats=3, random_state=self.cfg.seed, scoring="roc_auc")
        return {name: float(v) for name, v in zip(FEATURE_NAMES, r.importances_mean)}
