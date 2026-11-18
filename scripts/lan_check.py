"""Connectivity + end-to-end check of a running PathSnap API (stdlib only, so it runs on Laptop 2 too).

Usage:  python scripts/lan_check.py http://192.168.1.23:8000 [--all]
Checks /health, /api/system/info, /api/trips, then POST /api/predict for one trip (or all with --all),
an invalid trip id (expects 404), and prints timings. Exit code 0 only if everything passed.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request


def call(url, method="GET", timeout=180):
    t0 = time.perf_counter()
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}"), time.perf_counter() - t0


def main():
    base = sys.argv[1].rstrip("/")
    ok = True
    s, h, dt = call(f"{base}/health", timeout=10)
    print(f"/health -> {s} {h} ({dt * 1000:.0f} ms)")
    ok &= s == 200 and h.get("model_loaded")
    s, info, _ = call(f"{base}/api/system/info")
    print(f"mode={info.get('mode')} partition={info.get('partition')} trips={info.get('n_trips')}")
    s, trips, _ = call(f"{base}/api/trips")
    ids = [t["trip_id"] for t in trips]
    print(f"/api/trips -> {s}: {ids}")
    ok &= s == 200 and len(ids) > 0
    for tid in (ids if "--all" in sys.argv else ids[:1]):
        s, p, dt = call(f"{base}/api/predict/{tid}", "POST")
        m = p.get("metrics", {})
        acc = m.get("hmm", {}).get("top1_accuracy") if m.get("available") else None
        print(f"POST /api/predict/{tid} -> {s} in {dt:.2f}s  fixes={p.get('n_fixes')}  hmm_top1={acc}")
        ok &= s == 200
    s, _, _ = call(f"{base}/api/predict/does-not-exist", "POST")
    print(f"invalid trip id -> {s} (expected 404)")
    ok &= s == 404
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
