"""Decoder evaluation on the 10 student-testing trips.

Compares, with the tuned HMM parameters:
  A. classifier-only (independent per-fix argmax)
  B. classifier + HMM/Viterbi
  C. B + route reconstruction (shortest-path infill inside each chain)
for the untuned baseline classifier and for the final (tuned) classifier.

Usage:  python scripts/evaluate_all.py
Outputs: experiments/results/decoder_comparison_*.csv, decoder_summary.json,
         reports/figures/hmm_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import processed_dir  # noqa: E402
from pathsnap.decoding.hmm import DEFAULT_PARAMS  # noqa: E402
from pathsnap.evaluation.decoder_eval import compare_trip, pooled  # noqa: E402
from pathsnap.evaluation.networks import load_network  # noqa: E402
from pathsnap.training.tracking import run_metadata, write_json  # noqa: E402
from pathsnap.visualisation.plots import plt, save  # noqa: E402

RES = ROOT / "experiments" / "results"
FIG = ROOT / "reports" / "figures"


def run(variant: str, probs_file: str, params: dict, nets) -> pd.DataFrame:
    pdir = processed_dir()
    c = pd.read_pickle(pdir / "student_test_candidates.pkl")
    o = pd.read_pickle(pdir / "student_test_obs.pkl")
    c["p"] = np.load(pdir / probs_file)
    rows = []
    for tid, g in c.groupby("trip_id"):
        trip, net = nets[tid]
        r = compare_trip(tid, net, trip.route, g, g["p"].to_numpy(), o[o.trip_id == tid], params)
        rows.append(r)
        print(f"[{variant}] {tid}: clf={r['clf_top1']:.4f} hmm={r['hmm_top1']:.4f} "
              f"cont {r['clf_route_continuity']:.3f}->{r['hmm_route_continuity']:.3f}->{r['recon_route_continuity']:.3f}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"decoder_comparison_{variant}.csv", index=False)
    return df


def preliminary():
    """Review-1 checkpoint: untuned HGB + default (untuned) HMM parameters."""
    pdir = processed_dir()
    st_trips = sorted(pd.read_pickle(pdir / "student_test_obs.pkl")["trip_id"].unique())
    nets = {tid: load_network("student_test", tid) for tid in st_trips}
    df = run("preliminary", "student_probs_hist_gradient_boosting.npy", DEFAULT_PARAMS, nets)
    write_json(RES / "decoder_summary_preliminary.json", {**run_metadata(), "classifier": "hist_gradient_boosting (untuned)",
                                                          "hmm_params": DEFAULT_PARAMS, "summary": pooled(df)})


def main():
    if "--preliminary" in sys.argv:
        return preliminary()
    meta = json.loads((ROOT / "artifacts" / "metadata" / "classifier.json").read_text())
    hmm = json.loads((ROOT / "artifacts" / "metadata" / "hmm_params.json").read_text())["params"]
    pdir = processed_dir()
    st_trips = sorted(pd.read_pickle(pdir / "student_test_obs.pkl")["trip_id"].unique())
    nets = {tid: load_network("student_test", tid) for tid in st_trips}
    name = meta["model_name"]
    variants = {"final": (meta["student_probs_file"], hmm)}
    if meta["variant"] != name:
        variants["untuned"] = (f"student_probs_{name}.npy", hmm)
    variants["final_default_hmm"] = (meta["student_probs_file"], DEFAULT_PARAMS)
    summary = {}
    for v, (pf, params) in variants.items():
        df = run(v, pf, params, nets)
        summary[v] = pooled(df)
    write_json(RES / "decoder_summary.json", {**run_metadata(), "classifier": meta["variant"], "hmm_params": hmm,
                                              "summary": summary})
    plots(pd.read_csv(RES / "decoder_comparison_final.csv", dtype={"trip_id": str}), summary)


def plots(df, summary):
    x = np.arange(len(df))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(x - 0.2, df["clf_top1"], 0.4, label="classifier only")
    ax[0].bar(x + 0.2, df["hmm_top1"], 0.4, label="classifier + HMM")
    ax[0].set(xticks=x, xticklabels=df["trip_id"].str[-3:], ylim=(0.7, 1.0), title="Top-1 accuracy per student-test trip",
              xlabel="trip (last 3 digits)")
    ax[0].legend()
    ax[1].bar(x - 0.27, df["clf_route_continuity"], 0.27, label="classifier only")
    ax[1].bar(x, df["hmm_route_continuity"], 0.27, label="HMM (per-fix arcs)")
    ax[1].bar(x + 0.27, df["recon_route_continuity"], 0.27, label="HMM + reconstruction")
    ax[1].set(xticks=x, xticklabels=df["trip_id"].str[-3:], ylim=(0.5, 1.02), title="Route continuity per trip")
    ax[1].legend(fontsize=8)
    save(fig, FIG / "hmm_vs_classifier_per_trip.png")
    s = summary["final"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    keys = [("Top-1", "clf_top1", "hmm_top1"), ("Sel. Macro-F1", "clf_sel_macro_f1", "hmm_sel_macro_f1"),
            ("Continuity", "clf_route_continuity", "recon_route_continuity"),
            ("Route F1", "clf_route_f1", "recon_route_f1"), ("Edit sim.", "clf_edit_similarity", "recon_edit_similarity")]
    xx = np.arange(len(keys))
    ax.bar(xx - 0.2, [s[a] for _, a, _ in keys], 0.4, label="classifier only")
    ax.bar(xx + 0.2, [s[b] for _, _, b in keys], 0.4, label="classifier + HMM (+reconstruction)")
    for i, (_, a, b) in enumerate(keys):
        ax.text(i - 0.2, s[a] + 0.005, f"{s[a]:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, s[b] + 0.005, f"{s[b]:.3f}", ha="center", fontsize=8)
    ax.set(xticks=xx, xticklabels=[k for k, _, _ in keys], ylim=(0.5, 1.05), title="Student-test: classifier-only vs HMM")
    ax.legend(loc="lower right")
    save(fig, FIG / "hmm_vs_classifier_summary.png")


if __name__ == "__main__":
    main()
