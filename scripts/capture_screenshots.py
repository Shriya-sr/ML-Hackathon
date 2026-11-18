"""Capture PNG screenshots of the Folium maps with a headless Chromium browser
(Chrome or Edge), using a throw-away profile directory.

Usage:  python scripts/capture_screenshots.py
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe", "google-chrome", "chromium", "msedge"]


def browser() -> str:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    raise SystemExit("no Chrome/Edge found")


def shot(url: str, out: Path, w=1500, h=950, budget_ms=15000):
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as prof:
        cmd = [browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={prof}",
               f"--window-size={w},{h}", f"--virtual-time-budget={budget_ms}", f"--screenshot={out}", url]
        subprocess.run(cmd, check=True, timeout=180, capture_output=True)
    print("saved", out.relative_to(ROOT))


def main():
    for f in sorted((ROOT / "maps" / "html").glob("*.html")):
        shot(f.resolve().as_uri(), ROOT / "maps" / "screenshots" / f"{f.stem}.png", budget_ms=12000)


if __name__ == "__main__":
    main()
