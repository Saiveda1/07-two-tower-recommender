"""Shared professional matplotlib theme for portfolio 'screenshots'.

Copy this file into each project's src/<pkg>/viztheme.py so every project renders
charts/dashboards in one consistent, product-grade visual language.

Usage:
    from viztheme import apply_theme, PALETTE, save_panel
    apply_theme()
    fig, ax = plt.subplots()
    ...
    save_panel(fig, "assets/overview.png")
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Product-grade dark palette (colorblind-aware, high contrast on the ink background)
INK = "#0d1117"
PANEL = "#161b22"
GRID = "#26303b"
TEXT = "#e6edf3"
MUTED = "#8b949e"
PALETTE = ["#58a6ff", "#3fb950", "#f778ba", "#d29922", "#a371f7", "#ff7b72", "#39c5cf", "#db61a2"]
ACCENT = "#58a6ff"
GOOD = "#3fb950"
WARN = "#d29922"
BAD = "#ff7b72"


def apply_theme() -> None:
    mpl.rcParams.update({
        "figure.facecolor": INK,
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "savefig.facecolor": INK,
        "savefig.bbox": "tight",
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "grid.linewidth": 0.6,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.labelcolor": TEXT,
        "font.family": "DejaVu Sans",
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def kpi(ax, label: str, value: str, sub: str = "", color: str = ACCENT) -> None:
    """Render a single KPI 'stat tile' into an axis."""
    ax.axis("off")
    ax.set_facecolor(PANEL)
    ax.text(0.5, 0.66, value, ha="center", va="center", fontsize=26, fontweight="bold", color=color)
    # NOTE: `letterspacing` (present in the shared theme) is not a valid Text
    # kwarg on matplotlib >= 3.7 in this environment, so it is omitted here.
    ax.text(0.5, 0.30, label.upper(), ha="center", va="center", fontsize=9, color=MUTED)
    if sub:
        ax.text(0.5, 0.10, sub, ha="center", va="center", fontsize=8, color=MUTED)


def save_panel(fig, path: str, suptitle: str | None = None) -> None:
    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold", color=TEXT, y=0.99)
    fig.savefig(path)
    plt.close(fig)
