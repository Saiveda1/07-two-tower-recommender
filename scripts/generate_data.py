"""Generate a synthetic interaction log to Parquet (streaming, bounded memory).

    python scripts/generate_data.py --rows 5_000_000 --out data/interactions.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from twotower.data import GeneratorConfig, InteractionGenerator  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=5_000_000)
    p.add_argument("--users", type=int, default=50_000)
    p.add_argument("--items", type=int, default=5_000)
    p.add_argument("--categories", type=int, default=20)
    p.add_argument("--chunk", type=int, default=1_000_000)
    p.add_argument("--out", type=str, default="data/interactions.parquet")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cfg = GeneratorConfig(
        n_users=args.users, n_items=args.items, n_categories=args.categories, seed=args.seed
    )
    gen = InteractionGenerator(cfg)

    t0 = time.time()
    total = gen.generate_parquet(args.rows, args.out, chunk_size=args.chunk)
    dt = time.time() - t0
    size_mb = os.path.getsize(args.out) / 1e6
    print(
        f"[generate] rows={total:,} users={args.users:,} items={args.items:,} "
        f"-> {args.out} ({size_mb:.1f} MB) in {dt:.1f}s "
        f"({total / dt / 1e6:.2f}M rows/s)"
    )


if __name__ == "__main__":
    main()
