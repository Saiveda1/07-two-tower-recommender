# Architecture & Scaling Notes

This document covers the design decisions, the trade-offs behind each stage, and
the concrete path from the measured single-box run to a **1B-interaction /
100M-item** production system.

## 1. System shape: two-stage retrieval + ranking

```
                 ┌──────────────────────── offline / batch ───────────────────────┐
  interaction    │  temporal split ──► TrainStats (leak-free)                       │
  log (Parquet)  │        │                                                         │
                 │        ├─► Two-Tower trainer (NumPy, in-batch softmax)           │
                 │        │        └─► user_emb  +  item_emb                         │
                 │        │                                                         │
                 │        └─► GBM ranker (features over emb + pop + recency)        │
                 └──────────────────────────────┬──────────────────────────────────┘
                                                │  publish
                 ┌──────────────────────────────▼──────────────────────────────────┐
   user_id ──►   │  user tower ─► u-vector ─► ANN over item_emb (IVF) ─► top-C       │  ─► top-K
                 │                                    └─► GBM re-rank ─► top-K       │
                 └───────────────────────── online / serving ───────────────────────┘
```

Retrieval and ranking are split because they optimise different things:

| Stage      | Optimises | Cost model | Candidates |
|------------|-----------|------------|-----------|
| Retrieval  | recall    | O(log N) ANN over precomputed item vectors | N → C (~hundreds) |
| Ranking    | precision / nDCG | O(C) rich-feature GBM | C → K (tens) |

Scoring the full catalogue with a heavy model per request is infeasible at
100M items; scoring only the ANN shortlist is.

## 2. Two-tower retrieval model

- **Towers.** User and item towers are embedding lookups → `dim`-vectors; score
  is their dot product. A shallow tower *is* matrix factorisation, but the
  training objective (in-batch sampled softmax) and the serving shape (ANN over
  a frozen item table) are exactly the industrial two-tower recipe, so deeper
  MLP towers with side-features drop in behind the same interface.
- **In-batch softmax negatives.** For a batch of `B` positives we score the
  `B×B` matrix `U Vᵀ` and treat every off-diagonal item as a negative. One
  matmul yields `B` positives and `B·(B−1)` negatives — far cheaper than
  explicit negative sampling, and the negatives are *popularity-distributed* for
  free (they are real interacted items).
- **logQ correction.** Naive in-batch negatives over-penalise popular items
  (they appear as negatives constantly). We subtract each item's sampling
  log-probability, `s_jk ← s_jk − log q_k` (Yi et al., 2019, *Sampling-Bias-
  Corrected Neural Modeling*), restoring a well-calibrated softmax.
- **From-scratch NumPy SGD.** Gradients are derived analytically
  (`dL/d logits = (P − I)/B`, then chain to `U`, `V`) and applied with plain SGD
  + L2. No autograd framework, which keeps the repo dependency-light and makes
  the maths auditable. The hot loop pins BLAS to one thread
  (`threadpoolctl`) — the many small `B×B` matmuls are dominated by thread
  dispatch otherwise (~20× slower).

## 3. ANN candidate retrieval

`IVFIndex` is the coarse-quantiser half of FAISS `IVF*`: KMeans partitions items
into `nlist` Voronoi cells; a query probes its `nprobe` nearest cells and
brute-forces within them. `nprobe` is the accuracy/latency dial — the test-suite
asserts recall rises monotonically with it. `BruteForceIndex` gives exact
ground truth for small catalogues and for measuring IVF recall. Vectors are
L2-normalised so inner-product ranking equals cosine ranking.

## 4. Ranking stage

A `HistGradientBoostingClassifier` learns `P(engage | features)` from train-slice
positives and popularity-sampled negatives. Features combine the embedding
signal (`dot`, `cosine`, item-norm) with signals the frozen embeddings cannot
see: log-popularity, item/user mean-timestamp (recency), user activity, and
category match. Because these lift beyond the dot-product ordering, nDCG@K rises
over retrieval-only — asserted in `tests/test_pipeline.py`.

## 5. Leak-free temporal evaluation

- **Temporal split**, not random: sort by time, last `test_frac` of the horizon
  is test. `assert_no_leakage()` enforces `max(train.ts) < min(test.ts)`.
- **Every statistic is train-only**: popularity baseline, recency, user/item
  stats, and the ranker's training data all come from the train slice.
- **Relevant = new future engagement**: a user's train-seen items are removed
  from both their recommendations and their ground truth, so recall rewards
  genuine discovery, not re-showing history.
- **Metrics**: Recall@K, nDCG@K, HitRate@K, catalog coverage, novelty (mean
  self-information `−log₂ p(item)`), versus popularity and random baselines.

## 6. Scaling to 1B interactions / 100M items

**What is measured here.** The generator and the aggregation path are fully
streaming: `benchmark_scale.py` consumes the event stream in fixed chunks and
keeps only bounded histograms, so **peak RSS is flat while row count grows
1M → 10M → 100M** (see `benchmarks/scale_results.csv`). The two-tower is trained
on a multi-million-interaction sample with bounded memory (index arrays + one
mini-batch of embeddings resident at a time).

**Extrapolation to 1B, and the architecture that supports it:**

1. **Data & feature generation — out-of-core.** The interaction log is Parquet
   with one row group per chunk. Aggregations (popularity, recency, per-user
   histories) run in **DuckDB** (streaming, larger-than-RAM) or **Polars lazy**
   scans — no full materialisation. 1B rows is a partitioned dataset, not a
   Python list.

2. **Embedding tables — sharding.** 100M items × 64 × 4B ≈ **26 GB**; users
   likewise. Shard the item embedding table by `item_id % S` across parameter
   servers / a mmap'd on-disk table; each trainer worker owns a shard and
   scatter-updates only its rows. In-batch softmax needs the batch's item rows,
   fetched by hashing — the exact access pattern already used in
   `TwoTowerModel._batch_step` (row gather + scatter-add), just distributed.

3. **Training throughput — data parallelism.** Mini-batches are independent;
   the analytic gradient is a sum over the batch, so it maps to synchronous
   data-parallel SGD (ring all-reduce on the dense tower params, sparse
   scatter-add on embedding shards). The NumPy trainer is the single-worker
   reference for that update rule.

4. **ANN serving — IVF/IVF-PQ, sharded.** A 100M-vector index is sharded by IVF
   cell across replicas; a query fans out to the shards owning its `nprobe`
   cells and merges top-K. Product Quantisation compresses each vector to
   ~16–32 bytes (26 GB → ~2–3 GB) so the index is RAM-resident. `IVFIndex`
   here is the coarse quantiser; PQ residual coding and sharding are the
   production add-ons.

5. **Ranking — unchanged shape.** The GBM scores only the ~hundreds of ANN
   candidates per request, so its cost is independent of catalogue size; it
   scales horizontally by request.

**Honest bill of materials.** Measured: multi-M interactions trained, 1M→100M
rows streamed at bounded memory. Extrapolated with the above architecture:
1B interactions / 100M items. Nothing in the code path changes shape between the
two — only the storage backend (DuckDB/Parquet), the embedding sharding, and the
ANN index topology, all called out above.
