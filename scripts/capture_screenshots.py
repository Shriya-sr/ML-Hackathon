"""Capture PNG screenshots of the real Folium maps and the real running UI with a
headless Chromium browser (Chrome or Edge), using a throw-away profile directory.

Usage:
  python scripts/capture_screenshots.py maps
  python scripts/capture_screenshots.py ui --ui http://localhost:5173 --trips 00000052 00000009
(the UI mode needs the backend and `npm run dev` running; it opens ?trip=<id>, which makes the UI
 call POST /api/predict/<id> on the real API before the screenshot is taken)
"""
from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["maps", "ui"])
    ap.add_argument("--ui", default="http://localhost:5173")
    ap.add_argument("--trips", nargs="*", default=["00000052", "00000009"])
    a = ap.parse_args()
    if a.mode == "maps":
        for f in sorted((ROOT / "maps" / "html").glob("*.html")):
            shot(f.resolve().as_uri(), ROOT / "maps" / "screenshots" / f"{f.stem}.png", budget_ms=12000)
    else:
        for tid in a.trips:
            shot(f"{a.ui}/?trip={tid}", ROOT / "reports" / "figures" / f"ui_trip_{tid}.png", w=1600, h=1000,
                 budget_ms=60000)
        shot(a.ui, ROOT / "reports" / "figures" / "ui_dashboard_start.png", w=1600, h=1000, budget_ms=8000)


if __name__ == "__main__":
    main()
