"""A shallow two-tower retrieval model, trained from scratch in NumPy.

The **user tower** and **item tower** are each an embedding lookup producing a
``dim``-vector; the relevance score of a (user, item) pair is their dot product.
This is the retrieval-stage backbone used by industrial recommenders (YouTube,
Pinterest): cheap to serve because item vectors are precomputed and scored with
an ANN index.

Training uses **in-batch softmax with negative sampling**.  For a mini-batch of
``B`` positive ``(u, i)`` pairs we form the ``B×B`` score matrix ``U Vᵀ`` and
treat every *other* row's item as a negative for the current user.  The loss is
the sampled softmax cross-entropy of the diagonal (the true positives):

    L = -mean_j  log  softmax_k( s_jk )[j]

with a **logQ correction** (``s_jk -= log q_k``) that subtracts each item's
sampling log-probability, de-biasing the popular-item over-penalisation that
plagues naive in-batch negatives (Yi et al., 2019).  Gradients are derived
analytically and applied with plain SGD + L2 regularisation — no autograd, no
framework.  The loss provably decreases, which the test-suite asserts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

from . import SEED


@dataclass
class TwoTowerConfig:
    dim: int = 64
    lr: float = 0.25
    l2: float = 1e-6
    temperature: float = 0.05     # softmax temperature (scales the logits)
    batch_size: int = 512
    epochs: int = 6
    logq_correction: bool = False
    seed: int = SEED
    log_every: int = 200


@dataclass
class TrainHistory:
    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)


class TwoTowerModel:
    """User + item embedding towers with an in-batch-softmax NumPy trainer."""

    def __init__(self, n_users: int, n_items: int, cfg: TwoTowerConfig | None = None) -> None:
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.cfg = cfg or TwoTowerConfig()
        rng = np.random.default_rng(self.cfg.seed)
        scale = 1.0 / np.sqrt(self.cfg.dim)
        # float32 keeps memory bounded and matches serving precision.
        self.user_emb = (rng.standard_normal((self.n_users, self.cfg.dim)) * scale).astype(np.float32)
        self.item_emb = (rng.standard_normal((self.n_items, self.cfg.dim)) * scale).astype(np.float32)
        # Per-item bias (MF-style): absorbs global popularity so the serving
        # score u·v + b_i ranks popular items up without a separate prior.
        self.item_bias = np.zeros(self.n_items, dtype=np.float32)
        self.item_logq: np.ndarray | None = None  # log sampling prob per item
        self.history = TrainHistory()

    # ------------------------------------------------------------------
    @staticmethod
    def _softmax_rows(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _batch_step(self, u_idx: np.ndarray, i_idx: np.ndarray) -> tuple[float, float]:
        """One SGD step on a batch of positive pairs. Returns (loss, top1-acc)."""
        cfg = self.cfg
        B = u_idx.shape[0]
        U = self.user_emb[u_idx]                    # (B, d)
        V = self.item_emb[i_idx]                    # (B, d)

        b = self.item_bias[i_idx]                   # (B,) column-item biases
        # bias lives OUTSIDE the temperature so it learns at log-odds scale
        # (a proper MF/softmax bias) and can actually encode popularity.
        logits = (U @ V.T) / cfg.temperature + b[None, :]        # (B, B)
        if cfg.logq_correction and self.item_logq is not None:
            # subtract sampling log-prob of each *column* item (de-bias popular items)
            logits = logits - self.item_logq[i_idx][None, :]

        # Accidental-hit masking: when the same item id appears in several rows
        # of the batch, it must not act as a negative for the *other* rows whose
        # positive it also is. Without this, a small catalogue (batch ≳ n_items)
        # has colliding columns and the softmax target is ill-defined -> the loss
        # floors at log(B) and nothing is learned.
        same_item = i_idx[:, None] == i_idx[None, :]
        np.fill_diagonal(same_item, False)          # keep each row's true positive
        logits[same_item] = -1e9

        P = self._softmax_rows(logits)              # (B, B)
        # cross-entropy against the diagonal (true positive per row)
        diag = np.clip(P[np.arange(B), np.arange(B)], 1e-12, 1.0)
        loss = float(-np.log(diag).mean())
        acc = float((P.argmax(axis=1) == np.arange(B)).mean())

        # dL/dlogits = (P - I) / B
        G = P.copy()
        G[np.arange(B), np.arange(B)] -= 1.0
        G /= B

        # bias sits outside the temperature -> its gradient is dL/dlogits directly
        db = G.sum(axis=0)                          # (B,) bias gradient per column
        # embeddings sit inside: chain through 1/temperature
        Gs = G / cfg.temperature
        dU = Gs @ V                                 # (B, d)
        dV = Gs.T @ U                               # (B, d)

        # L2 regularisation on the touched rows
        dU += cfg.l2 * U
        dV += cfg.l2 * V

        # scatter-add updates (rows may repeat within a batch)
        np.add.at(self.user_emb, u_idx, (-cfg.lr * dU).astype(np.float32))
        np.add.at(self.item_emb, i_idx, (-cfg.lr * dV).astype(np.float32))
        np.add.at(self.item_bias, i_idx, (-cfg.lr * db).astype(np.float32))
        return loss, acc

    # ------------------------------------------------------------------
    def fit(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        *,
        item_counts: np.ndarray | None = None,
    ) -> TrainHistory:
        """Train on positive interaction pairs (implicit feedback).

        ``item_counts`` (empirical train popularity) powers the logQ correction.
        Data is consumed in shuffled mini-batches per epoch; only index arrays
        and one batch of embeddings are held at once, so memory stays bounded.
        """
        cfg = self.cfg
        user_ids = np.asarray(user_ids)
        item_ids = np.asarray(item_ids)
        n = user_ids.shape[0]

        if cfg.logq_correction:
            if item_counts is None:
                item_counts = np.bincount(item_ids, minlength=self.n_items).astype(np.float64)
            q = item_counts.astype(np.float64) + 1.0
            q = q / q.sum()
            self.item_logq = np.log(q).astype(np.float32)

        rng = np.random.default_rng(cfg.seed + 7)
        step = 0
        # Many small (B×B, tiny-K) matmuls: BLAS thread dispatch dominates, so
        # pin to a single thread — ~20× faster than the default thread pool here.
        with threadpool_limits(limits=1, user_api="blas"):
            for _epoch in range(cfg.epochs):
                order = rng.permutation(n)
                for start in range(0, n - cfg.batch_size + 1, cfg.batch_size):
                    sel = order[start : start + cfg.batch_size]
                    loss, acc = self._batch_step(user_ids[sel], item_ids[sel])
                    if step % cfg.log_every == 0:
                        self.history.steps.append(step)
                        self.history.loss.append(loss)
                        self.history.accuracy.append(acc)
                    step += 1
        return self.history

    # ------------------------------------------------------------------
    def serving_bias(self) -> np.ndarray:
        """Item bias on the serving (dot-product) scale: τ·b_i.

        Training logits are ``u·v/τ + b``; ranking by that equals ranking by
        ``u·v + τ·b``, so the serving-scale bias is ``τ·b_i``.
        """
        return (self.cfg.temperature * self.item_bias).astype(np.float32)

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Serving score u·v + τ·b_i for aligned user/item id arrays."""
        return (
            np.einsum("ij,ij->i", self.user_emb[user_ids], self.item_emb[item_ids])
            + self.cfg.temperature * self.item_bias[item_ids]
        )

    def retrieval_item_matrix(self) -> np.ndarray:
        """Item vectors augmented with the bias as an extra dimension.

        Concatenating ``b_i`` as a coordinate turns the affine serving score
        ``u·v + b_i`` into a plain inner product against ``[u, 1]``, so any
        maximum-inner-product ANN index scores items exactly as served.
        """
        return np.concatenate(
            [self.item_emb, self.serving_bias()[:, None]], axis=1
        ).astype(np.float32)

    def retrieval_query_matrix(self, user_ids: np.ndarray) -> np.ndarray:
        """User vectors augmented with a constant 1 to pair with the item bias."""
        u = self.user_emb[user_ids]
        return np.concatenate([u, np.ones((u.shape[0], 1), np.float32)], axis=1).astype(np.float32)

    def normalized_item_embeddings(self) -> np.ndarray:
        v = self.item_emb
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return (v / np.clip(norm, 1e-8, None)).astype(np.float32)
