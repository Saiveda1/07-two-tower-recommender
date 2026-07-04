"""End-to-end headline claims — the assertions that make this portfolio real.

Uses the session-scoped ``pipeline`` fixture (one small, fully-seeded real
training run). The framing is the standard two-stage recommender:

* retrieval (two-tower + ANN) maximises *candidate recall* into a shortlist,
* the ranker maximises the final ordering (nDCG / recall@K),
* the ranked system beats the popularity and random baselines.
"""
from __future__ import annotations


def test_no_leakage_end_to_end(pipeline):
    pipeline.split.assert_no_leakage()


def test_two_tower_loss_decreased(pipeline):
    loss = pipeline.model.history.loss
    assert loss[-1] < loss[0] * 0.9, f"loss did not decrease: {loss[0]:.3f} -> {loss[-1]:.3f}"


def test_system_beats_popularity_recall(pipeline):
    """The full two-tower system (retrieval + ranker) beats the popularity baseline."""
    m = pipeline.metrics
    assert m["two_tower_ranked"].recall > m["popularity"].recall, (
        f"ranked recall {m['two_tower_ranked'].recall:.4f} "
        f"!> popularity {m['popularity'].recall:.4f}"
    )


def test_beats_random_by_a_wide_margin(pipeline):
    m = pipeline.metrics
    assert m["two_tower_ranked"].recall > 3 * max(m["random"].recall, 1e-6)
    assert m["two_tower"].recall > 3 * max(m["random"].recall, 1e-6)


def test_ranking_improves_over_retrieval(pipeline):
    """The ranker lifts BOTH nDCG and recall over the retrieval-only ordering."""
    m = pipeline.metrics
    assert m["two_tower_ranked"].ndcg > m["two_tower"].ndcg, (
        f"ranker nDCG {m['two_tower_ranked'].ndcg:.4f} "
        f"!> retrieval {m['two_tower'].ndcg:.4f}"
    )
    assert m["two_tower_ranked"].recall > m["two_tower"].recall


def test_retrieval_provides_candidate_recall(pipeline):
    """Retrieval + ranker beats a strong-ish bar; retrieval clears random comfortably."""
    m = pipeline.metrics
    assert m["two_tower_ranked"].recall > 0.15
    assert m["two_tower"].ndcg > 0.03


def test_ann_recall_is_high(pipeline):
    assert pipeline.ann_recall > 0.80


def test_metrics_in_valid_ranges(pipeline):
    for name, m in pipeline.metrics.items():
        assert 0.0 <= m.recall <= 1.0
        assert 0.0 <= m.ndcg <= 1.0
        assert 0.0 <= m.hit_rate <= 1.0
        assert 0.0 <= m.coverage <= 1.0
        assert m.novelty > 0.0


def test_retrieval_more_diverse_than_popularity(pipeline):
    """Personalised retrieval covers far more of the catalog than a pop list."""
    m = pipeline.metrics
    assert m["two_tower"].coverage > m["popularity"].coverage
