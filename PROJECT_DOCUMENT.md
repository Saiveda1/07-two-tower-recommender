# Two-Tower Recommender Project Document

**Prepared For:** Sai Veda  
**GitHub Publishing Account:** Nikeshk834  
**Repository Slug:** `07-two-tower-recommender`  
**Verified Test Count From Portfolio Index:** 33  

## Background

A complete, **offline, deterministic** recommender system: a from-scratch
two-tower retrieval model, ANN candidate generation, a gradient-boosted ranker,
and a leak-free temporal evaluation harness — trained on **millions of implicit
interactions** and architected for **1B**. No GPU, no network, no paid APIs.

```
interaction log ─► temporal split ─► Two-Tower (NumPy, in-batch softmax)
                                        │  user + item embeddings
                                        ▼
                                   ANN (IVF) top-C  ─►  GBM ranker top-K  ─►  eval vs baselines
```

## Why it's interesting

- **Two-tower trained from scratch in NumPy** — in-batch sampled softmax with a
  logQ popularity correction, analytic gradients, plain SGD. The loss provably
  decreases and the item space recovers category structure (both asserted in
  tests). No deep-learning framework.
- **Real two-stage architecture** — ANN retrieval optimised for recall, then a
  `HistGradientBoosting` ranker optimised for nDCG over embedding + popularity +
  recency features. The ranker measurably lifts nDCG over retrieval-only.
- **Honest, leak-free evaluation** — a *temporal* split (train on the past,
  score the future), every statistic computed train-only, seen items excluded.
  Recall@K, nDCG@K, HitRate, coverage and novelty vs popularity + random.
- **Streaming to 1B** — the generator and aggregation path are chunked with
  bounded memory; peak RSS stays flat from 1M to 100M rows.

## Quickstart

```bash
make run          # stream 5M interactions, train two-tower + ranker, evaluate
make screenshots  # render the PNGs in assets/ from that run
make test         # behavioural pytest suite
make bench        # streaming-scale benchmark (bounded memory, up to 100M+ rows)
```

`make run ROWS=20000000` trains on a larger sample. Everything is seeded.

## Results

| method | recall@20 | nDCG@20 | hit-rate | coverage | novelty (bits) |
|--------|----------:|--------:|---------:|---------:|---------------:|
| random | 0.004 | 0.003 | 0.030 | 1.000 | 14.03 |
| popularity | 0.128 | 0.097 | 0.550 | 0.073 | 6.56 |
| **two-tower** (retrieval) | 0.225 | 0.184 | 0.743 | 0.829 | 9.78 |
| **two-tower + ranker** | **0.240** | **0.198** | **0.764** | 0.812 | 9.99 |

*Temporal split (last 20% of the horizon is the future test window), K=20.
Novelty is mean self-information `−log₂ p(item)` in bits; higher = less popular.*

Headline (from the committed run):

- **Two-tower retrieval beats the popularity baseline by +76% recall@20** (0.225 vs
  0.128) and **+90% nDCG** (0.184 vs 0.097) — it captures each user's category
  affinity, which a global popularity ranking structurally cannot.
- **The GBM ranker lifts the shortlist a further +7.8% nDCG / +6.7% recall** →
  final **recall@20 0.240, nDCG@20 0.198, hit-rate 0.764**.
- Trained on **4M interactions** (5M streamed, leak-free temporal split), **49,372**
  eval users, 5,000-item catalogue. ANN (IVF) candidate recall@20 **0.967**;
  two-tower softmax loss **10.87 → 4.55**; coverage **0.81** (personalised, not
  head-only). End-to-end wall time **336 s**, bounded memory.

> Engineering note: two bugs were caught and fixed under review. (1) Without a
> **logQ correction**, in-batch softmax over-penalised popular items and the towers
> learned to *avoid* them — losing to popularity. (2) The ranker sampled negatives
> **popularity-proportionally**, teaching it that popularity is anti-predictive, so
> it inverted a good shortlist below even the popularity baseline; **uniform
> negatives** fixed it and the ranker now genuinely lifts nDCG. Both are classic
> two-tower/two-stage failure modes — see `ARCHITECTURE.md`.

## Project Purpose

This repository is part of the AI engineering portfolio and focuses on the following problem space:

- Retrieval + ranking, trained from scratch
- Headline result from the portfolio index: beats popularity **+76% recall**; ranker **+7.8% nDCG**

## What This Project Solves

This project provides a production-style implementation with benchmark evidence and operational checks committed into the repository.

## Technical Approach

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
  `B×B` matrix `U Vᵀ` and treat every off-diagonal item as a neg

## Benchmark And Validation Evidence

The portfolio root documents **33 passing tests** for this project, and the repo quickstart uses `make test` as the standard validation path. The benchmark outputs committed in `benchmarks/` and the generated visuals in `assets/` are the evidence package for this delivery.



## Visual Artifacts Reviewed

- `assets/metrics_vs_baselines.png`: Retrieval quality vs baselines.
- `assets/embedding_space.png`: Item embedding space (PCA), coloured by category.
- `assets/training_loss.png`: Two-tower training curve.
- `assets/kpi_dashboard.png`: Offline evaluation dashboard.

## Engineering Notes

The primary design and scale decisions are documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md). The benchmark markdown in [`benchmarks/`](./benchmarks) and the generated figures in [`assets/`](./assets) should be read together: the markdown gives the measured numbers, and the screenshots make those results easier to inspect quickly during review.

## Files Included In This Repo

- [`README.md`](./README.md) for project overview, quickstart, and headline results
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) for system design and scaling choices
- [`benchmarks/`](./benchmarks) for measured results from the committed runs
- [`assets/`](./assets) for generated screenshots and dashboards
- [`tests/`](./tests) for the automated validation suite

## Delivery Summary

This project document was prepared for **Sai Veda** so the repository reads like a real project handoff: what the system is for, what problem it solves, what evidence supports it, and where the benchmark and test artifacts live inside the repo.
