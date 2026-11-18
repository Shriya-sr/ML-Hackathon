"""Run the complete experiment chain in order (same commands as README section 2).

Usage:  python scripts/run_all.py [--from STEP]
Steps: eda features train analysis optimise hmm evaluate maps figures report
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("eda", ["scripts/run_eda.py"]),
    ("features", ["scripts/build_features.py", "--jobs", "6"]),
    ("train", ["-W", "ignore", "scripts/train_all.py", "--fresh"]),
    ("analysis", ["-W", "ignore", "scripts/feature_analysis.py"]),
    ("optimise", ["-W", "ignore", "scripts/optimise.py"]),
    ("hmm", ["scripts/tune_hmm.py"]),
    ("evaluate", ["scripts/evaluate_all.py"]),
    ("maps", ["scripts/generate_maps.py"]),
    ("figures", ["scripts/make_report_figures.py"]),
    ("report", ["scripts/build_report.py"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="eda", choices=[s for s, _ in STEPS])
    a = ap.parse_args()
    names = [s for s, _ in STEPS]
    for name, args in STEPS[names.index(a.start):]:
        t0 = time.time()
        print(f"=== {name}: python {' '.join(args)}", flush=True)
        subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
        print(f"=== {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
