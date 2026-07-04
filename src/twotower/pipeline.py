"""End-to-end retrieval + ranking pipeline and its evaluation harness.

Wires the components into the standard two-stage recommender:

    interactions
        │  temporal_split (leak-free)
        ▼
    TrainStats ──► TwoTowerModel.fit ──► item/user embeddings
                                             │
                              ANN index (BruteForce / IVF)
                                             │  top-C candidates / user
                                             ▼
                              GBMRanker.rerank ──► top-K
                                             │
                              evaluate_recommendations vs baselines
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ann import BruteForceIndex, IVFIndex
from .evaluate import (
    MetricResult,
    Split,
    TrainStats,
    evaluate_recommendations,
    popularity_recommendations,
    random_recommendations,
    temporal_split,
)
from .model import TwoTowerConfig, TwoTowerModel
from .ranker import GBMRanker, RankerConfig


@dataclass
class PipelineConfig:
    dim: int = 64
    k: int = 20
    candidate_pool: int = 200      # C: shortlist size from ANN before ranking
    test_frac: float = 0.2
    use_ivf: bool = True
    ivf_nlist: int = 128
    ivf_nprobe: int = 16
    tower: TwoTowerConfig = field(default_factory=TwoTowerConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)


@dataclass
class PipelineResult:
    split: Split
    stats: TrainStats
    model: TwoTowerModel
    metrics: dict[str, MetricResult]
    retrieval_recs: dict[int, list[int]]
    ranked_recs: dict[int, list[int]]
    ann_recall: float
    item_category: np.ndarray


def _retrieve_candidates(
    model: TwoTowerModel,
    split: Split,
    n_items: int,
    cfg: PipelineConfig,
) -> tuple[dict[int, list[int]], float]:
    """ANN top-C candidates per eval user (train-seen items removed).

    Retrieval is **maximum inner-product** over the *raw* item vectors: the
    embedding norm encodes how popular / confident an item is, so keeping it
    blends personalisation with a learned popularity prior. L2-normalising here
    would throw that signal away.
    """
    item_vecs = model.retrieval_item_matrix()   # [v_i | b_i]
    users = split.eval_users
    q = model.retrieval_query_matrix(users)      # [u | 1]  -> inner product = u·v + b_i

    # over-fetch so that after removing seen items we still have >= pool
    max_seen = max((len(split.train_history.get(int(u), ())) for u in users), default=0)
    fetch = min(n_items, cfg.candidate_pool + max_seen + cfg.k)

    if cfg.use_ivf and n_items > cfg.ivf_nlist:
        index = IVFIndex(nlist=cfg.ivf_nlist, nprobe=cfg.ivf_nprobe).build(item_vecs)
        ann_recall = index.recall_vs_exact(q[: min(256, len(q))], cfg.k)
    else:
        index = BruteForceIndex(item_vecs)
        ann_recall = 1.0

    idx, _ = index.search(q, fetch)
    pools: dict[int, list[int]] = {}
    for row, u in enumerate(users):
        u = int(u)
        seen = split.train_history.get(u, set())
        pool = [int(it) for it in idx[row] if it >= 0 and it not in seen]
        pools[u] = pool[: cfg.candidate_pool]
    return pools, ann_recall


def run_pipeline(interactions: dict[str, np.ndarray], item_category: np.ndarray,
                 n_users: int, n_items: int, cfg: PipelineConfig | None = None) -> PipelineResult:
    cfg = cfg or PipelineConfig()

    split = temporal_split(interactions, test_frac=cfg.test_frac)
    split.assert_no_leakage()
    stats = TrainStats.from_train(split, n_users, n_items, item_category)

    # --- retrieval model ---------------------------------------------------
    model = TwoTowerModel(n_users, n_items, cfg.tower)
    model.fit(split.train_users, split.train_items, item_counts=stats.item_count)

    # --- candidate generation ---------------------------------------------
    pools, ann_recall = _retrieve_candidates(model, split, n_items, cfg)
    retrieval_recs = {u: pool[: cfg.k] for u, pool in pools.items()}

    # --- ranking stage -----------------------------------------------------
    ranker = GBMRanker(cfg.ranker).fit(split, model, stats, item_category, n_items)
    ranked_recs = ranker.rerank_batch(pools, model, stats, item_category, k=cfg.k)

    # --- evaluation vs baselines ------------------------------------------
    metrics = {
        "random": evaluate_recommendations(
            random_recommendations(split, n_items, k=cfg.k), split, stats, k=cfg.k, n_items=n_items),
        "popularity": evaluate_recommendations(
            popularity_recommendations(split, stats, k=cfg.k), split, stats, k=cfg.k, n_items=n_items),
        "two_tower": evaluate_recommendations(
            retrieval_recs, split, stats, k=cfg.k, n_items=n_items),
        "two_tower_ranked": evaluate_recommendations(
            ranked_recs, split, stats, k=cfg.k, n_items=n_items),
    }

    return PipelineResult(
        split=split, stats=stats, model=model, metrics=metrics,
        retrieval_recs=retrieval_recs, ranked_recs=ranked_recs,
        ann_recall=ann_recall, item_category=item_category,
    )
