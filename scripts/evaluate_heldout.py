"""Held-out evaluation on the evaluator trips (data/raw/evaluator_test) with the FROZEN deployed artifacts.

This is a final, report-only measurement. It must be run only after every modelling decision is fixed: nothing
in this repository (training, tuning, model selection, HMM grid) reads its outputs.

Per trip:
  1. the deployed pipeline (``MapMatcher``, the same code the API runs) predicts from .track/.nodes/.arcs; the
     .route file is not loaded at this point;
  2. only afterwards is .route read and aligned to the fixes (same labelling as for training), and the prediction
     is scored with the same ``compare_trip`` code that produced the student-test decoder table.
It also checks that no evaluator trip overlaps the training or student-testing data (trip ID or .track content).

Usage:   python scripts/evaluate_heldout.py
Outputs: experiments/results/heldout_evaluator_comparison.csv, heldout_evaluator_summary.json,
         reports/figures/heldout_per_trip.png
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import data_dir, load_config  # noqa: E402
from pathsnap.data.labeling import align_to_route  # noqa: E402
from pathsnap.data.loader import read_route  # noqa: E402
from pathsnap.evaluation.decoder_eval import compare_trip, pooled  # noqa: E402
from pathsnap.evaluation.metrics import classification_metrics  # noqa: E402
from pathsnap.features.candidate_generation import attach_labels  # noqa: E402
from pathsnap.inference.pipeline import MapMatcher  # noqa: E402
from pathsnap.training.tracking import run_metadata, write_json  # noqa: E402
from pathsnap.visualisation.plots import plt, save  # noqa: E402

RES = ROOT / "experiments" / "results"
FIG = ROOT / "reports" / "figures"


def trip_dirs(part: Path) -> list[Path]:
    return sorted(p for p in part.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists()) if part.exists() else []


def overlap_check(root: Path, ev: list[Path]) -> dict:
    def md5(f: Path) -> str:
        return hashlib.md5(f.read_bytes()).hexdigest()
    known = [d for part in ("train", "student_test") for d in trip_dirs(root / part)]
    ids = {d.name for d in known} & {d.name for d in ev}
    hashes = {md5(d / f"{d.name}.track") for d in known} & {md5(d / f"{d.name}.track") for d in ev}
    return {"n_known_trips": len(known), "shared_trip_ids": sorted(ids), "shared_track_files": len(hashes)}


def main():
    root = data_dir()
    ev = trip_dirs(root / "evaluator_test")
    if not ev:
        sys.exit(f"no evaluator trips in {root / 'evaluator_test'}")
    ov = overlap_check(root, ev)
    print(f"overlap with train/student-test: ids={ov['shared_trip_ids']} track files={ov['shared_track_files']}")
    if ov["shared_trip_ids"] or ov["shared_track_files"]:
        sys.exit("evaluator trips overlap the development data - not a held-out set")

    mm = MapMatcher()
    lc = load_config("data")["labelling"]
    hmm = json.loads((ROOT / "artifacts" / "metadata" / "hmm_params.json").read_text())["params"]
    rows, y_all, p_all = [], [], []
    for d in ev:
        pred = mm.predict(d)                                  # .route not loaded
        ctx = pred.ctx
        ref = read_route(d / f"{d.name}.route")               # read only after prediction
        labels, _, _ = align_to_route(ctx.px, ctx.py, ctx.net, ref, lc["distance_cap_m"], lc["max_label_distance_m"])
        cands = attach_labels(ctx.cands, labels)
        hit = np.zeros(len(labels), bool)
        hit[cands.loc[cands["label"] == 1, "obs"].to_numpy()] = True
        obs = pd.DataFrame({"obs": np.arange(len(labels)), "x": ctx.px, "y": ctx.py, "t": ctx.track.t,
                            "label": labels, "true_in_candidates": hit})
        r = compare_trip(d.name, ctx.net, ref, cands, pred.prob, obs, hmm)
        assert np.isclose(r["clf_top1"], float((pred.clf_arc[labels >= 0] == labels[labels >= 0]).mean()))
        r["pipeline_seconds"] = pred.timings["total_s"]
        rows.append(r)
        keep = cands["true_arc"].to_numpy() >= 0
        y_all.append(cands["label"].to_numpy()[keep]); p_all.append(pred.prob[keep])
        print(f"{d.name}: clf={r['clf_top1']:.4f} hmm={r['hmm_top1']:.4f} "
              f"cont {r['clf_route_continuity']:.3f}->{r['recon_route_continuity']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "heldout_evaluator_comparison.csv", index=False)
    cm = classification_metrics(np.concatenate(y_all), np.concatenate(p_all))
    summary = {**pooled(df), "clf_macro_f1_threshold_0_5": cm["macro_f1"], "clf_log_loss": cm["log_loss"],
               "clf_roc_auc": cm.get("roc_auc")}
    write_json(RES / "heldout_evaluator_summary.json", {
        **run_metadata(), "partition": "evaluator_test", "classifier": mm.meta["variant"], "hmm_params": hmm,
        "frozen_artifacts": True, "used_for_model_selection": False, "overlap_check": ov,
        "trips": [d.name for d in ev], "summary": summary})
    print(f"pooled Top-1 {summary['clf_top1']:.4f} -> {summary['hmm_top1']:.4f}; "
          f"HMM better on {(df.hmm_top1 > df.clf_top1).sum()}/{len(df)} trips")
    plot(df)


def plot(df: pd.DataFrame):
    x = np.arange(len(df))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(x - 0.2, df["clf_top1"], 0.4, label="classifier only")
    ax[0].bar(x + 0.2, df["hmm_top1"], 0.4, label="classifier + HMM")
    ax[0].set(xticks=x, xticklabels=df["trip_id"].str[-3:], ylim=(0.7, 1.04), title="Top-1 accuracy per held-out evaluator trip",
              xlabel="trip (last 3 digits)")
    ax[0].legend(loc="upper center", ncol=2, fontsize=8)
    ax[1].bar(x - 0.2, df["clf_route_continuity"], 0.4, label="classifier only")
    ax[1].bar(x + 0.2, df["recon_route_continuity"], 0.4, label="HMM + reconstruction")
    ax[1].set(xticks=x, xticklabels=df["trip_id"].str[-3:], ylim=(0.5, 1.1), title="Route continuity per held-out trip")
    ax[1].legend(loc="upper center", ncol=2, fontsize=8)
    save(fig, FIG / "heldout_per_trip.png")


if __name__ == "__main__":
    main()
