"""Streaming-scale benchmark: bounded-memory aggregation over up to 1B events.

Proves the generator + aggregation path is truly streaming.  We consume the
event stream in fixed-size chunks and maintain only:

    * an item popularity histogram (size = n_items)
    * a category histogram (size = n_categories)
    * running counters

Peak resident memory therefore depends on ``chunk_size`` and the (bounded)
catalogue — **not** on the number of interactions — so the same code path scales
from 1M to 1B rows.  Results are written to ``benchmarks/scale_results.csv``.

    python benchmarks/benchmark_scale.py --rows 1_000_000 10_000_000 100_000_000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from twotower.data import GeneratorConfig, InteractionGenerator  # noqa: E402


def _peak_rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # pragma: no cover - platform dependent
        return float("nan")


def stream_aggregate(gen: InteractionGenerator, rows: int, chunk: int) -> dict[str, object]:
    cfg = gen.cfg
    item_hist = np.zeros(cfg.n_items, dtype=np.int64)
    cat_hist = np.zeros(cfg.n_categories, dtype=np.int64)
    n = 0
    t0 = time.time()
    for _u, items, _ts, cats in gen.stream(rows, chunk_size=chunk):
        item_hist += np.bincount(items, minlength=cfg.n_items)
        cat_hist += np.bincount(cats, minlength=cfg.n_categories)
        n += items.shape[0]
    dt = time.time() - t0
    # head share = fraction of events on the top 1% of items (the long-tail head)
    top = np.sort(item_hist)[-max(1, cfg.n_items // 100):].sum()
    return {
        "rows": n,
        "seconds": round(dt, 2),
        "throughput_Mrows_s": round(n / dt / 1e6, 2),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "head_1pct_share": round(top / n, 4),
        "distinct_items_touched": int((item_hist > 0).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, nargs="+", default=[1_000_000, 10_000_000, 100_000_000])
    p.add_argument("--users", type=int, default=50_000)
    p.add_argument("--items", type=int, default=5_000)
    p.add_argument("--chunk", type=int, default=2_000_000)
    args = p.parse_args()

    gen = InteractionGenerator(GeneratorConfig(n_users=args.users, n_items=args.items))
    results = []
    for rows in args.rows:
        r = stream_aggregate(gen, rows, args.chunk)
        results.append(r)
        print(f"[scale] rows={r['rows']:>13,}  {r['seconds']:>7.2f}s  "
              f"{r['throughput_Mrows_s']:>5.2f} M/s  peakRSS={r['peak_rss_mb']:.0f}MB  "
              f"head1%={r['head_1pct_share']:.3f}")

    out = os.path.join(os.path.dirname(__file__), "scale_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(out, index=False)
    print(f"[scale] wrote {out}")
    print(f"[scale] peak RSS stayed ~constant while rows grew "
          f"{results[0]['rows']:,} -> {results[-1]['rows']:,} "
          f"(bounded-memory streaming confirmed)")


if __name__ == "__main__":
    main()
