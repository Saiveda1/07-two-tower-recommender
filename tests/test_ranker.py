"""Ranker features and reranking behaviour."""
from __future__ import annotations

import numpy as np

from twotower.ranker import FEATURE_NAMES, _build_features


def test_reranker_reorders(pipeline):
    """Reranked lists differ from retrieval order for at least some users."""
    changed = 0
    for u in list(pipeline.ranked_recs)[:200]:
        if pipeline.ranked_recs[u] != pipeline.retrieval_recs[u]:
            changed += 1
    assert changed > 0


def test_feature_matrix_shape(pipeline):
    model, stats = pipeline.model, pipeline.stats
    item_norm = np.linalg.norm(model.item_emb, axis=1)
    users = np.array([0, 1, 2, 3])
    items = np.array([0, 1, 2, 3])
    X = _build_features(users, items, model, stats, pipeline.item_category, item_norm)
    assert X.shape == (4, len(FEATURE_NAMES))
    assert np.isfinite(X).all()


def test_ranked_lists_have_no_duplicates(pipeline):
    for rec in list(pipeline.ranked_recs.values())[:100]:
        assert len(rec) == len(set(rec))
