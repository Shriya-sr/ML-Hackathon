"""Grid-search the HMM decoder parameters on TRAINING trips only.

Emissions use OUT-OF-FOLD classifier probabilities (each training trip scored by
a model that never saw it), so decoder parameters are tuned on probabilities
with the same quality the decoder will see on unseen trips.

Usage:  python scripts/tune_hmm.py [--trips 30 --jobs 6]
Outputs: experiments/results/hmm_grid.csv, artifacts/metadata/hmm_params.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import SEED, load_config, processed_dir  # noqa: E402
from pathsnap.decoding.hmm import DEFAULT_PARAMS, HMMDecoder, classifier_only  # noqa: E402
from pathsnap.evaluation.networks import load_network  # noqa: E402
from pathsnap.evaluation.route_metrics import route_metrics  # noqa: E402
from pathsnap.training.tracking import run_metadata, write_json  # noqa: E402


def one_trip(tid, cands, obs, configs):
    trip, net = load_network("train", tid)
    obs = obs.sort_values("obs")
    lab = obs["label"].to_numpy()
    m = lab >= 0
    px, py, t = obs["x"].to_numpy(), obs["y"].to_numpy(), obs["t"].to_numpy()
    prob = cands["p"].to_numpy()
    a0, _ = classifier_only(cands, prob, len(obs))
    rows = [{"trip_id": tid, "config": -1, "n_labelled": int(m.sum()), "n_correct": int((a0[m] == lab[m]).sum()),
             "continuity": route_metrics(a0, net)["route_continuity"]}]
    for k, cfg in enumerate(configs):
        r = HMMDecoder(cfg).decode(net, cands, prob, px, py, t)
        rows.append({"trip_id": tid, "config": k, "n_labelled": int(m.sum()),
                     "n_correct": int((r.arc[m] == lab[m]).sum()),
                     "continuity": route_metrics(r.arc, net)["route_continuity"], "seconds": r.seconds})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    hcfg = load_config("hmm")
    meta = json.loads((ROOT / "artifacts" / "metadata" / "classifier.json").read_text())
    pdir = processed_dir()
    c = pd.read_pickle(pdir / "train_candidates.pkl")
    o = pd.read_pickle(pdir / "train_obs.pkl")
    c["p"] = np.load(pdir / meta["oof_file"])
    assert c["p"].notna().all(), "OOF probabilities missing for some rows"
    rng = np.random.default_rng(SEED)
    trips = sorted(c["trip_id"].unique())
    n = args.trips or hcfg["tuning_trips"]
    sel = sorted(rng.choice(trips, size=min(n, len(trips)), replace=False))
    grid = hcfg["grid"]
    keys = list(grid.keys())
    configs = [{**DEFAULT_PARAMS, **hcfg["defaults"], **dict(zip(keys, vals))}
               for vals in itertools.product(*[grid[k] for k in keys])]
    print(f"tuning {len(configs)} configs on {len(sel)} training trips (OOF probs from {meta['variant']})", flush=True)
    res = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(one_trip)(tid, c[c.trip_id == tid], o[o.trip_id == tid], configs) for tid in sel)
    df = pd.DataFrame([r for rows in res for r in rows])
    agg = df.groupby("config").agg(n_labelled=("n_labelled", "sum"), n_correct=("n_correct", "sum"),
                                   continuity=("continuity", "mean"), seconds=("seconds", "sum")).reset_index()
    agg["top1"] = agg["n_correct"] / agg["n_labelled"]
    for k in keys:
        agg[k] = [np.nan if i < 0 else configs[i][k] for i in agg["config"]]
    agg = agg.sort_values("top1", ascending=False)
    out = ROOT / "experiments" / "results"
    agg.to_csv(out / "hmm_grid.csv", index=False)
    df.to_csv(out / "hmm_grid_per_trip.csv", index=False)
    best_k = int(agg[agg.config >= 0].iloc[0]["config"])
    best = configs[best_k]
    base = agg[agg.config == -1].iloc[0]
    print(agg.head(10).to_string())
    print(f"classifier-only top1 on tuning trips: {base.top1:.4f}; best HMM: {agg[agg.config == best_k].top1.iloc[0]:.4f}")
    write_json(ROOT / "artifacts" / "metadata" / "hmm_params.json", {
        **run_metadata(), "params": best, "tuned_on": "train (OOF probabilities)", "tuning_trips": sel,
        "objective": "pooled observation-level Top-1 accuracy", "grid": grid,
        "classifier_only_top1": float(base.top1), "best_hmm_top1": float(agg[agg.config == best_k].top1.iloc[0]),
    })


if __name__ == "__main__":
    main()
