"""Matplotlib helpers shared by EDA, model comparison and report figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a6fdb", "#e8833a", "#2e9e6b", "#c94f7c", "#7b61c9", "#b8962e", "#4aa3b5", "#777777"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "axes.titlesize": 11,
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
})


def save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
