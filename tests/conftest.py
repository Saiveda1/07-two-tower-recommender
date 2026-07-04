"""Shared fixtures: one small end-to-end pipeline run reused across tests."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from twotower.data import GeneratorConfig, InteractionGenerator  # noqa: E402
from twotower.model import TwoTowerConfig  # noqa: E402
from twotower.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from twotower.ranker import RankerConfig  # noqa: E402


@pytest.fixture(scope="session")
def gen_config() -> GeneratorConfig:
    return GeneratorConfig(n_users=2500, n_items=600, n_categories=10, seed=42)


@pytest.fixture(scope="session")
def interactions(gen_config):
    gen = InteractionGenerator(gen_config)
    parts = {k: [] for k in ("user_id", "item_id", "ts", "category")}
    for u, i, t, c in gen.stream(150_000, chunk_size=50_000):
        parts["user_id"].append(u); parts["item_id"].append(i)
        parts["ts"].append(t); parts["category"].append(c)
    return gen, {k: np.concatenate(v) for k, v in parts.items()}


@pytest.fixture(scope="session")
def pipeline(interactions, gen_config):
    gen, inter = interactions
    cfg = PipelineConfig(
        k=20, candidate_pool=100, use_ivf=True, ivf_nlist=24, ivf_nprobe=14,
        tower=TwoTowerConfig(
            dim=32, epochs=8, batch_size=512, temperature=0.05, lr=0.3, log_every=60
        ),
        ranker=RankerConfig(max_positives=20_000, max_iter=60, n_threads=2),
    )
    return run_pipeline(inter, gen.item_category, gen_config.n_users, gen_config.n_items, cfg)
