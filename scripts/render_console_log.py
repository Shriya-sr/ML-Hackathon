"""Render a captured API console log (text) as a terminal-style PNG for the report.

Usage:  python scripts/render_console_log.py experiments/logs/api_demo_console.log reports/figures/api_console_log.png
The input must be the real stdout of `python -m backend.main` (captured with `| Tee-Object` or `> file`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(src: str, dst: str, max_lines: int = 40):
    lines = Path(src).read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [ln for ln in lines if ln.strip()][-max_lines:]
    h = 0.22 * len(lines) + 0.6
    fig = plt.figure(figsize=(13, h), facecolor="#0c0c0c")
    fig.text(0.01, 1 - 0.25 / h, f"Laptop 1 console: python -m backend.main   (captured from {Path(src).name})",
             color="#9cdcfe", family="monospace", fontsize=8.5, va="top")
    for i, ln in enumerate(lines):
        col = "#ce9178" if " 4" in ln.split("->")[-1][:4] or "ERROR" in ln else ("#b5cea8" if "trip=" in ln else "#d4d4d4")
        fig.text(0.01, 1 - (0.5 + 0.22 * i) / h, ln[:175], color=col, family="monospace", fontsize=7.6, va="top")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=150, facecolor=fig.get_facecolor())
    print("saved", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
