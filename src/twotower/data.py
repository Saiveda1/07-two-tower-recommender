"""Streaming implicit-feedback interaction-log generator.

The generator produces ``(user_id, item_id, ts, category)`` events for a
catalogue of items with:

* **Popularity skew** — item interaction propensity follows a Zipf law, so a
  small head of items dominates the log (the classic long-tail catalogue).
* **User activity skew** — a Zipf law over users, so a minority of "power
  users" generate most events.
* **Category affinity** — every user has a primary category they mostly consume
  from, giving the embeddings real structure to recover.
* **Temporal drift** — category popularity rises and falls over the time
  horizon (a sinusoidal trend per category), so a *temporal* train/test split
  is genuinely harder than a random one and recency is a real signal.

Generation is fully **streaming and chunked**: :meth:`InteractionGenerator.stream`
yields fixed-size NumPy chunks, so it can emit up to 1B events with memory
bounded by ``chunk_size + n_users + n_items`` — never the full log.  The catalogue
(``n_users``, ``n_items``) stays bounded, so 1B interactions simply means heavy
repetition, exactly as in a real platform.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorConfig:
    """Parameters controlling the synthetic interaction log."""

    n_users: int = 50_000
    n_items: int = 5_000
    n_categories: int = 20
    pop_zipf: float = 1.03          # item popularity skew (higher => steeper long tail)
    user_zipf: float = 0.75         # user activity skew
    focus: float = 0.72             # P(event drawn from user's primary category)
    trend_amp: float = 0.55         # temporal drift amplitude (0..1)
    seed: int = 42


_HASH_C = np.uint64(2654435761)
_SM1 = np.uint64(0x9E3779B97F4A7C15)
_SM2 = np.uint64(0xBF58476D1CE4E5B9)
_SM3 = np.uint64(0x94D049BB133111EB)
_N_TIME_EPOCHS = 128  # granularity of the temporal-drift table


def _zipf_weights(n: int, s: float) -> np.ndarray:
    """Normalized 1/rank^s weights for ranks 1..n."""
    w = 1.0 / np.power(np.arange(1, n + 1, dtype=np.float64), s)
    return w / w.sum()


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 finaliser (uint64 in -> well-mixed uint64 out)."""
    with np.errstate(over="ignore"):
        z = (x + _SM1).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(30))) * _SM2).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(27))) * _SM3).astype(np.uint64)
        z = (z ^ (z >> np.uint64(31))).astype(np.uint64)
    return z


def _hash_uniform(event_idx: np.ndarray, stream: int, seed: int) -> np.ndarray:
    """Deterministic uniform[0,1) keyed by (global event index, stream, seed).

    Counter-based, so a given event yields the same value regardless of how the
    stream is chunked — the split is bit-identical across chunk sizes and can be
    generated in parallel by index range.
    """
    with np.errstate(over="ignore"):
        key = (event_idx.astype(np.uint64) * np.uint64(4) + np.uint64(stream)) ^ (
            np.uint64(seed) * _SM1
        )
    bits = _splitmix64(key.astype(np.uint64))
    # top 53 bits -> double in [0, 1)
    return (bits >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def _inverse_cdf(cdf: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Map uniforms through a cumulative distribution (searchsorted)."""
    idx = np.searchsorted(cdf, u, side="right")
    return np.clip(idx, 0, cdf.shape[0] - 1)


class InteractionGenerator:
    """Deterministic, streamable implicit-feedback event source."""

    def __init__(self, cfg: GeneratorConfig | None = None) -> None:
        self.cfg = cfg or GeneratorConfig()
        c = self.cfg
        rng = np.random.default_rng(c.seed)

        # --- catalogue: item -> category, item latent popularity -------------
        self.item_category = rng.integers(0, c.n_categories, size=c.n_items).astype(np.int16)

        # A latent per-item popularity: Zipf weights assigned to a random
        # permutation of items so the head is not simply items 0..k.
        base = _zipf_weights(c.n_items, c.pop_zipf)
        perm = rng.permutation(c.n_items)
        self.item_popularity = np.empty(c.n_items, dtype=np.float64)
        self.item_popularity[perm] = base

        # Per-category item pools + within-category sampling CDFs (inverse-CDF sampling).
        self._cat_items: list[np.ndarray] = []
        self._cat_item_cdf: list[np.ndarray] = []
        for cat in range(c.n_categories):
            idx = np.where(self.item_category == cat)[0]
            if idx.size == 0:  # ensure every category is non-empty
                idx = np.array([int(rng.integers(0, c.n_items))])
            p = self.item_popularity[idx]
            p = p / p.sum()
            self._cat_items.append(idx)
            self._cat_item_cdf.append(np.cumsum(p))

        # Category base weight and temporal phase (drives temporal drift).
        self.cat_base = _zipf_weights(c.n_categories, 0.5)
        self.cat_phase = rng.uniform(0.0, 1.0, size=c.n_categories)

        # Precompute a per-epoch category CDF table so the trend is a cheap,
        # index-addressable lookup (piecewise-constant in time).
        epochs = (np.arange(_N_TIME_EPOCHS) + 0.5) / _N_TIME_EPOCHS
        m = 1.0 + c.trend_amp * np.sin(2.0 * np.pi * (epochs[:, None] + self.cat_phase[None, :]))
        w = self.cat_base[None, :] * m
        w = w / w.sum(axis=1, keepdims=True)
        self._trend_cdf = np.cumsum(w, axis=1)  # (epochs, n_categories)

        # User activity CDF (Zipf) and per-user primary category.
        self.user_activity_p = _zipf_weights(c.n_users, c.user_zipf)
        self._user_cdf = np.cumsum(self.user_activity_p)
        uids = np.arange(c.n_users, dtype=np.uint64)
        self.user_primary_cat = ((uids * _HASH_C) % np.uint64(c.n_categories)).astype(np.int16)

    # ------------------------------------------------------------------
    def _sample_items(self, categories: np.ndarray, x_item: np.ndarray) -> np.ndarray:
        items = np.empty(categories.shape[0], dtype=np.int64)
        for cat in np.unique(categories):
            mask = categories == cat
            ranks = _inverse_cdf(self._cat_item_cdf[cat], x_item[mask])
            items[mask] = self._cat_items[cat][ranks]
        return items

    # ------------------------------------------------------------------
    def stream(
        self,
        rows: int,
        *,
        chunk_size: int = 1_000_000,
        start: float = 0.0,
        end: float = 1.0,
    ):
        """Yield ``(users, items, ts, categories)`` chunks totalling ``rows`` events.

        Timestamps are normalized to ``[start, end)`` and monotonic across the
        stream. Generation is counter-based (each event's draws are hashed from
        its global index), so the output is **bit-identical for any chunk size**
        and memory is bounded by ``chunk_size`` regardless of ``rows``.
        """
        c = self.cfg
        produced = 0
        while produced < rows:
            k = int(min(chunk_size, rows - produced))
            e = np.arange(produced, produced + k, dtype=np.int64)

            # normalized, monotonic timestamps derived purely from event index
            frac = e.astype(np.float64) / rows
            ts = (start + (end - start) * frac).astype(np.float32)

            # four independent uniform streams per event
            x_user = _hash_uniform(e, 0, c.seed)
            x_focus = _hash_uniform(e, 1, c.seed)
            x_cat = _hash_uniform(e, 2, c.seed)
            x_item = _hash_uniform(e, 3, c.seed)

            users = _inverse_cdf(self._user_cdf, x_user)

            epoch = np.clip((frac * _N_TIME_EPOCHS).astype(np.int64), 0, _N_TIME_EPOCHS - 1)
            use_primary = x_focus < c.focus
            cats = np.empty(k, dtype=np.int16)
            cats[use_primary] = self.user_primary_cat[users[use_primary]]
            expl = ~use_primary
            if expl.any():
                # per-event inverse-CDF against that event's epoch trend row
                cdf_rows = self._trend_cdf[epoch[expl]]                  # (m, n_cat)
                x = x_cat[expl][:, None]
                cats[expl] = (x > cdf_rows).sum(axis=1).clip(0, c.n_categories - 1)

            items = self._sample_items(cats, x_item)
            yield (
                users.astype(np.int32),
                items.astype(np.int32),
                ts,
                cats,
            )
            produced += k

    # ------------------------------------------------------------------
    def generate_parquet(
        self,
        rows: int,
        path: str,
        *,
        chunk_size: int = 1_000_000,
    ) -> int:
        """Stream ``rows`` events to a Parquet file (one row group per chunk)."""
        schema = pa.schema(
            [
                ("user_id", pa.int32()),
                ("item_id", pa.int32()),
                ("ts", pa.float32()),
                ("category", pa.int16()),
            ]
        )
        writer = pq.ParquetWriter(path, schema, compression="zstd")
        total = 0
        try:
            for users, items, ts, cats in self.stream(rows, chunk_size=chunk_size):
                batch = pa.record_batch(
                    [pa.array(users), pa.array(items), pa.array(ts), pa.array(cats)],
                    schema=schema,
                )
                writer.write_batch(batch)
                total += len(users)
        finally:
            writer.close()
        return total


def load_interactions(path: str) -> dict[str, np.ndarray]:
    """Load a generated Parquet log fully into NumPy arrays (for training)."""
    tbl = pq.read_table(path)
    return {
        "user_id": tbl.column("user_id").to_numpy(),
        "item_id": tbl.column("item_id").to_numpy(),
        "ts": tbl.column("ts").to_numpy(),
        "category": tbl.column("category").to_numpy(),
    }
