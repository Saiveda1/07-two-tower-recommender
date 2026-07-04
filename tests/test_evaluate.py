"""Temporal split integrity and metric correctness."""
from __future__ import annotations

import numpy as np

from twotower.evaluate import (
    MetricResult,
    _dcg,
    _idcg,
    evaluate_recommendations,
    temporal_split,
)


def _interactions(seed=0, n=20_000, n_users=500, n_items=200):
    rng = np.random.default_rng(seed)
    return {
        "user_id": rng.integers(0, n_users, n).astype(np.int32),
        "item_id": rng.integers(0, n_items, n).astype(np.int32),
        "ts": np.sort(rng.random(n)).astype(np.float32),
        "category": rng.integers(0, 8, n).astype(np.int16),
    }


def test_temporal_split_no_leakage():
    sp = temporal_split(_interactions(), test_frac=0.2)
    sp.assert_no_leakage()
    assert sp.train_ts.max() < sp.test_ts.min()


def test_ground_truth_excludes_seen():
    sp = temporal_split(_interactions(seed=1), test_frac=0.25)
    for u in sp.eval_users:
        u = int(u)
        assert sp.ground_truth[u]
        assert not (sp.ground_truth[u] & sp.train_history[u])


def test_split_proportions():
    sp = temporal_split(_interactions(seed=2), test_frac=0.2)
    frac = sp.test_ts.size / (sp.train_ts.size + sp.test_ts.size)
    assert 0.15 < frac < 0.25


def test_dcg_math():
    # perfect ranking of two relevant items at positions 0,1
    rel = {10, 20}
    assert abs(_dcg([10, 20, 5], rel) - (1 / np.log2(2) + 1 / np.log2(3))) < 1e-9
    assert abs(_idcg(2, 5) - (1 / np.log2(2) + 1 / np.log2(3))) < 1e-9


def test_perfect_recommender_scores_one():
    sp = temporal_split(_interactions(seed=4), test_frac=0.2)

    class Stats:  # minimal stub for novelty
        item_selfinfo = np.ones(200)

    recs = {int(u): list(sp.ground_truth[int(u)])[:20] for u in sp.eval_users}
    r = evaluate_recommendations(recs, sp, Stats(), k=20, n_items=200)
    assert isinstance(r, MetricResult)
    assert r.hit_rate == 1.0
    assert r.recall > 0.99  # every truncated ground-truth item retrieved
