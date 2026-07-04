"""Train the two-tower + ranker and evaluate against baselines.

Streams the interaction log (bounded memory), runs the full retrieval+ranking
pipeline on a temporal split, prints headline metrics, and writes:

* ``benchmarks/metrics.csv``  — per-method Recall/nDCG/HitRate/coverage/novelty
* ``data/results.npz``        — arrays for the screenshot generator

    python scripts/run.py --rows 5_000_000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from twotower.data import GeneratorConfig, InteractionGenerator  # noqa: E402
from twotower.model import TwoTowerConfig  # noqa: E402
from twotower.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from twotower.ranker import RankerConfig  # noqa: E402


def _stream_to_memory(gen: InteractionGenerator, rows: int, chunk: int) -> dict[str, np.ndarray]:
    parts = {k: [] for k in ("user_id", "item_id", "ts", "category")}
    for u, i, t, c in gen.stream(rows, chunk_size=chunk):
        parts["user_id"].append(u); parts["item_id"].append(i)
        parts["ts"].append(t); parts["category"].append(c)
    return {k: np.concatenate(v) for k, v in parts.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=5_000_000)
    p.add_argument("--users", type=int, default=50_000)
    p.add_argument("--items", type=int, default=5_000)
    p.add_argument("--categories", type=int, default=20)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--chunk", type=int, default=1_000_000)
    p.add_argument("--out-dir", type=str, default=".")
    args = p.parse_args()

    root = args.out_dir
    os.makedirs(os.path.join(root, "benchmarks"), exist_ok=True)
    os.makedirs(os.path.join(root, "data"), exist_ok=True)

    gcfg = GeneratorConfig(
        n_users=args.users, n_items=args.items, n_categories=args.categories
    )
    gen = InteractionGenerator(gcfg)

    print(f"[data] streaming {args.rows:,} interactions "
          f"({args.users:,} users x {args.items:,} items)...")
    t0 = time.time()
    inter = _stream_to_memory(gen, args.rows, args.chunk)
    print(f"[data] materialised in {time.time() - t0:.1f}s")

    pcfg = PipelineConfig(
        dim=args.dim, k=args.k, candidate_pool=200,
        use_ivf=True, ivf_nlist=128, ivf_nprobe=24,
        tower=TwoTowerConfig(
            dim=args.dim, epochs=args.epochs, batch_size=512, temperature=0.05, lr=0.25,
            # logQ correction de-biases the in-batch-softmax popularity penalty
            # (Yi et al. 2019). Without it the towers learn to AVOID popular items
            # and lose to a plain popularity baseline; with it they keep popularity
            # AND add per-user category affinity. item_counts already flow into fit().
            logq_correction=True,
        ),
        ranker=RankerConfig(max_positives=150_000, max_iter=150, n_threads=4),
    )
    t0 = time.time()
    res = run_pipeline(inter, gen.item_category, args.users, args.items, pcfg)
    train_s = time.time() - t0

    # --- report -----------------------------------------------------------
    rows = [res.metrics[m].as_row(m) for m in ("random", "popularity", "two_tower", "two_tower_ranked")]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, "benchmarks", "metrics.csv"), index=False)

    loss = res.model.history.loss
    pop = res.metrics["popularity"]; tt = res.metrics["two_tower"]; rk = res.metrics["two_tower_ranked"]
    print("\n" + "=" * 66)
    print(f"  pipeline wall time     : {train_s:.1f}s")
    print(f"  train interactions     : {res.split.train_users.size:,}")
    print(f"  eval users             : {len(res.split.eval_users):,}")
    print(f"  two-tower loss         : {loss[0]:.3f} -> {loss[-1]:.3f}")
    print(f"  ANN (IVF) recall@{args.k}    : {res.ann_recall:.3f}")
    print("=" * 66)
    print(df.to_string(index=False))
    print("=" * 66)
    print(f"  Recall@{args.k}  two-tower vs popularity : "
          f"{tt.recall:.4f} vs {pop.recall:.4f}  (+{100*(tt.recall/max(pop.recall,1e-9)-1):.0f}%)")
    print(f"  nDCG@{args.k}    ranked vs retrieval     : "
          f"{rk.ndcg:.4f} vs {tt.ndcg:.4f}  (+{100*(rk.ndcg/max(tt.ndcg,1e-9)-1):.1f}%)")
    print("=" * 66)

    # --- persist arrays for screenshots ----------------------------------
    meta = {
        "loss_steps": res.model.history.steps,
        "loss": loss,
        "accuracy": res.model.history.accuracy,
        "metrics": {m: res.metrics[m].as_row(m) for m in res.metrics},
        "ann_recall": res.ann_recall,
        "k": args.k,
        "train_interactions": int(res.split.train_users.size),
        "eval_users": int(len(res.split.eval_users)),
        "n_items": int(args.items),
        "wall_time_s": train_s,
    }
    np.savez_compressed(
        os.path.join(root, "data", "results.npz"),
        item_emb=res.model.normalized_item_embeddings(),
        item_category=gen.item_category,
        item_count=res.stats.item_count,
        meta=json.dumps(meta),
    )
    with open(os.path.join(root, "benchmarks", "summary.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[out] wrote benchmarks/metrics.csv, benchmarks/summary.json, data/results.npz")


if __name__ == "__main__":
    main()
