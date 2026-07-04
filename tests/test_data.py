"""Interaction-log generator: determinism, skew, temporal signal, streaming."""
from __future__ import annotations

import numpy as np

from twotower.data import GeneratorConfig, InteractionGenerator


def _collect(gen, rows, chunk):
    parts = {k: [] for k in ("user_id", "item_id", "ts", "category")}
    for u, i, t, c in gen.stream(rows, chunk_size=chunk):
        parts["user_id"].append(u); parts["item_id"].append(i)
        parts["ts"].append(t); parts["category"].append(c)
    return {k: np.concatenate(v) for k, v in parts.items()}


def test_determinism():
    cfg = GeneratorConfig(n_users=500, n_items=200, seed=7)
    a = _collect(InteractionGenerator(cfg), 20_000, 5_000)
    b = _collect(InteractionGenerator(cfg), 20_000, 5_000)
    for k in a:
        assert np.array_equal(a[k], b[k])


def test_chunk_size_invariant():
    """The stream is identical regardless of chunk size (bounded-memory safe)."""
    cfg = GeneratorConfig(n_users=500, n_items=200, seed=11)
    big = _collect(InteractionGenerator(cfg), 30_000, 30_000)
    small = _collect(InteractionGenerator(cfg), 30_000, 3_000)
    for k in big:
        assert np.array_equal(big[k], small[k])


def test_timestamps_monotonic_and_normalized():
    cfg = GeneratorConfig(n_users=500, n_items=200, seed=3)
    d = _collect(InteractionGenerator(cfg), 40_000, 10_000)
    ts = d["ts"]
    assert ts.min() >= 0.0 and ts.max() < 1.0
    assert np.all(np.diff(ts) >= 0)  # non-decreasing across the whole stream


def test_popularity_skew_is_long_tailed():
    cfg = GeneratorConfig(n_users=1000, n_items=400, pop_zipf=1.1, seed=5)
    d = _collect(InteractionGenerator(cfg), 100_000, 20_000)
    counts = np.bincount(d["item_id"], minlength=400).astype(float)
    counts.sort()
    top10 = counts[-40:].sum()          # top 10% of items
    assert top10 / counts.sum() > 0.35   # head dominates -> real long tail


def test_category_affinity_recoverable():
    """Users mostly consume their primary category (structure to learn)."""
    cfg = GeneratorConfig(n_users=800, n_items=300, n_categories=8, focus=0.72, seed=9)
    gen = InteractionGenerator(cfg)
    d = _collect(gen, 80_000, 20_000)
    item_cat = gen.item_category
    primary = gen.user_primary_cat[d["user_id"]]
    match = (item_cat[d["item_id"]] == primary).mean()
    assert match > 0.35  # far above 1/8 chance


def test_temporal_drift_present():
    """Category mix in the first vs last time-slice differs (real drift)."""
    cfg = GeneratorConfig(n_users=1000, n_items=300, n_categories=10, trend_amp=0.6, seed=13)
    d = _collect(InteractionGenerator(cfg), 120_000, 20_000)
    n = len(d["ts"])
    early = np.bincount(d["category"][: n // 5], minlength=10).astype(float)
    late = np.bincount(d["category"][-n // 5:], minlength=10).astype(float)
    early /= early.sum(); late /= late.sum()
    assert np.abs(early - late).sum() > 0.05  # distributions moved


def test_parquet_roundtrip(tmp_path):
    from twotower.data import load_interactions
    cfg = GeneratorConfig(n_users=400, n_items=150, seed=1)
    path = str(tmp_path / "log.parquet")
    total = InteractionGenerator(cfg).generate_parquet(25_000, path, chunk_size=8_000)
    assert total == 25_000
    d = load_interactions(path)
    assert len(d["user_id"]) == 25_000
    assert d["item_id"].max() < 150
