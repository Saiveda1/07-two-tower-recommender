"""Render professional PNG screenshots from a real pipeline run (data/results.npz).

    python scripts/run.py --rows 5_000_000     # produces data/results.npz
    python scripts/make_screenshots.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from twotower.viztheme import (  # noqa: E402
    ACCENT, BAD, GOOD, GRID, MUTED, PALETTE, PANEL, TEXT, WARN,
    apply_theme, kpi, save_panel,
)

METHOD_LABELS = {
    "random": "Random", "popularity": "Popularity",
    "two_tower": "Two-Tower", "two_tower_ranked": "Two-Tower + Ranker",
}
METHOD_COLORS = {"random": MUTED, "popularity": WARN, "two_tower": ACCENT, "two_tower_ranked": GOOD}
ORDER = ["random", "popularity", "two_tower", "two_tower_ranked"]


def _load(root: str):
    d = np.load(os.path.join(root, "data", "results.npz"), allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    return d, meta


# ---------------------------------------------------------------------------
def chart_metrics(root, d, meta):
    m = meta["metrics"]
    k = meta["k"]
    metrics = [("recall", f"Recall@{k}"), ("ndcg", f"nDCG@{k}"), ("hit_rate", f"HitRate@{k}")]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(metrics))
    w = 0.2
    for j, name in enumerate(ORDER):
        vals = [m[name][mk] for mk, _ in metrics]
        bars = ax.bar(x + (j - 1.5) * w, vals, w, label=METHOD_LABELS[name],
                      color=METHOD_COLORS[name], edgecolor=PANEL, linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylabel("score")
    ax.set_title("Retrieval quality vs baselines — temporal split, no leakage")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    ax.set_ylim(0, max(m["two_tower_ranked"]["recall"], m["two_tower_ranked"]["hit_rate"]) * 1.25)
    save_panel(fig, os.path.join(root, "assets", "metrics_vs_baselines.png"))


def chart_embedding(root, d, meta):
    emb = d["item_emb"]; cats = d["item_category"]; counts = d["item_count"]
    n_cat = int(cats.max()) + 1
    # focus on the most populated categories for a legible plot
    top_cats = np.argsort(-np.bincount(cats, minlength=n_cat))[:8]
    mask = np.isin(cats, top_cats)
    xy = PCA(n_components=2, random_state=0).fit_transform(emb[mask])
    c = cats[mask]; sz = 6 + 40 * (counts[mask] / counts[mask].max())

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    for j, cat in enumerate(top_cats):
        sel = c == cat
        ax.scatter(xy[sel, 0], xy[sel, 1], s=sz[sel], alpha=0.75,
                   color=PALETTE[j % len(PALETTE)], label=f"cat {cat}",
                   edgecolors="none")
    ax.set_title("Item embedding space (PCA) — colour = category, size = popularity")
    ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
    ax.legend(ncol=2, fontsize=8, loc="best", title="category")
    save_panel(fig, os.path.join(root, "assets", "embedding_space.png"))


def chart_loss(root, d, meta):
    steps = np.array(meta["loss_steps"]); loss = np.array(meta["loss"]); acc = np.array(meta["accuracy"])
    fig, ax1 = plt.subplots(figsize=(8.6, 5.0))
    ax1.plot(steps, loss, color=ACCENT, lw=2, label="in-batch softmax loss")
    ax1.set_xlabel("SGD step"); ax1.set_ylabel("loss", color=ACCENT)
    ax1.tick_params(axis="y", labelcolor=ACCENT)
    ax2 = ax1.twinx()
    ax2.plot(steps, acc, color=GOOD, lw=1.6, alpha=0.9, label="in-batch top-1 acc")
    ax2.set_ylabel("in-batch top-1 accuracy", color=GOOD)
    ax2.tick_params(axis="y", labelcolor=GOOD)
    ax2.grid(False)
    ax1.set_title(f"Two-tower training — loss {loss[0]:.2f} → {loss[-1]:.2f} "
                  f"({100*(1-loss[-1]/loss[0]):.0f}% ↓)")
    save_panel(fig, os.path.join(root, "assets", "training_loss.png"))


def dashboard(root, d, meta):
    m = meta["metrics"]; k = meta["k"]
    fig = plt.figure(figsize=(11.5, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.9, 1.25], hspace=0.42, wspace=0.32)

    tt, pop, rk = m["two_tower"], m["popularity"], m["two_tower_ranked"]
    lift = 100 * (tt["recall"] / max(pop["recall"], 1e-9) - 1)
    kpi(fig.add_subplot(gs[0, 0]), f"Recall@{k}", f"{rk['recall']:.3f}",
        f"+{lift:.0f}% vs popularity", GOOD)
    kpi(fig.add_subplot(gs[0, 1]), f"nDCG@{k}", f"{rk['ndcg']:.3f}",
        f"ranker +{100*(rk['ndcg']/max(tt['ndcg'],1e-9)-1):.0f}% vs retrieval", ACCENT)
    kpi(fig.add_subplot(gs[0, 2]), "ANN recall", f"{meta['ann_recall']:.2f}",
        "IVF vs exact top-k", WARN)
    kpi(fig.add_subplot(gs[0, 3]), "train rows", f"{meta['train_interactions']/1e6:.1f}M",
        f"{meta['eval_users']:,} eval users", PALETTE[4])

    # coverage vs novelty scatter
    axc = fig.add_subplot(gs[1, :2])
    for name in ORDER:
        axc.scatter(m[name]["coverage"], m[name]["novelty"], s=180,
                    color=METHOD_COLORS[name], edgecolors=PANEL, linewidth=1, zorder=3)
        axc.annotate(METHOD_LABELS[name], (m[name]["coverage"], m[name]["novelty"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=8, color=TEXT)
    axc.set_xlabel("catalog coverage"); axc.set_ylabel("novelty (mean self-info, bits)")
    axc.set_title("Coverage vs novelty — beyond-accuracy quality")

    # recall by method bar
    axb = fig.add_subplot(gs[1, 2:])
    names = ORDER
    vals = [m[n]["recall"] for n in names]
    axb.barh([METHOD_LABELS[n] for n in names], vals,
             color=[METHOD_COLORS[n] for n in names], edgecolor=PANEL)
    for i, v in enumerate(vals):
        axb.text(v + max(vals) * 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=TEXT)
    axb.set_xlabel(f"Recall@{k}"); axb.set_title("Recall@K by method")
    axb.invert_yaxis()

    save_panel(fig, os.path.join(root, "assets", "kpi_dashboard.png"),
               suptitle="Two-Tower Recommender — Offline Evaluation Dashboard")


def main() -> None:
    root = os.path.join(os.path.dirname(__file__), "..")
    os.makedirs(os.path.join(root, "assets"), exist_ok=True)
    apply_theme()
    d, meta = _load(root)
    chart_metrics(root, d, meta)
    chart_embedding(root, d, meta)
    chart_loss(root, d, meta)
    dashboard(root, d, meta)
    print("[screenshots] wrote 4 PNGs to assets/")


if __name__ == "__main__":
    main()
