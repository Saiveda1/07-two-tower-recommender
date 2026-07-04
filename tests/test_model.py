"""Two-tower trainer: the loss must actually decrease and learn structure."""
from __future__ import annotations

import numpy as np

from twotower.model import TwoTowerConfig, TwoTowerModel


def _toy_interactions(seed=0, n_items=80, n=20_000):
    """Users in 4 blocks, each preferring one item cluster — learnable structure."""
    rng = np.random.default_rng(seed)
    n_users = 400
    blocks = 4
    u, i = [], []
    for _ in range(n):
        b = rng.integers(blocks)
        user = b * (n_users // blocks) + rng.integers(n_users // blocks)
        item = b * (n_items // blocks) + rng.integers(n_items // blocks)
        u.append(user); i.append(item)
    return np.array(u), np.array(i), n_users, n_items


def test_loss_decreases():
    u, i, nu, ni = _toy_interactions()
    m = TwoTowerModel(nu, ni, TwoTowerConfig(dim=16, epochs=6, batch_size=512, log_every=10))
    hist = m.fit(u, i)
    assert len(hist.loss) > 5
    first = np.mean(hist.loss[:3])
    last = np.mean(hist.loss[-3:])
    assert last < first * 0.9, f"loss did not decrease: {first:.3f} -> {last:.3f}"


def test_accuracy_improves():
    # a larger item catalogue keeps in-batch positives distinct, so top-1
    # accuracy is a meaningful signal (tiny catalogues repeat items per batch).
    u, i, nu, ni = _toy_interactions(seed=1, n_items=1600, n=30_000)
    B = 256
    m = TwoTowerModel(nu, ni, TwoTowerConfig(dim=16, epochs=6, batch_size=B, log_every=10))
    hist = m.fit(u, i)
    assert hist.accuracy[-1] > hist.accuracy[0]
    assert hist.accuracy[-1] > 3.0 / B  # well above in-batch chance (1/B)


def test_learns_block_structure():
    """Items in the same block end up closer than items across blocks."""
    u, i, nu, ni = _toy_interactions(seed=2)
    m = TwoTowerModel(nu, ni, TwoTowerConfig(dim=16, epochs=8, batch_size=512))
    m.fit(u, i)
    V = m.normalized_item_embeddings()
    within = V[0] @ V[5]          # both in block 0 (items 0..19)
    across = V[0] @ V[60]         # block 0 vs block 3 (items 60..79)
    assert within > across


def test_deterministic_training():
    u, i, nu, ni = _toy_interactions(seed=3)
    cfg = TwoTowerConfig(dim=16, epochs=3, batch_size=512)
    a = TwoTowerModel(nu, ni, cfg).fit(u, i).loss
    b = TwoTowerModel(nu, ni, cfg).fit(u, i).loss
    assert np.allclose(a, b)


def test_score_shapes():
    u, i, nu, ni = _toy_interactions(seed=4)
    m = TwoTowerModel(nu, ni, TwoTowerConfig(dim=16, epochs=1))
    m.fit(u, i)
    s = m.score(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert s.shape == (3,)
