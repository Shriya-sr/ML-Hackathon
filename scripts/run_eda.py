"""Dataset inventory, data-quality checks and EDA figures.

Usage:  python scripts/run_eda.py
Outputs: reports/eda/*.csv, reports/eda/data_quality_report.md, reports/eda/figures/*.png

Partitions used: train (80) and student_test (10). The evaluator partition is
only inventoried (file presence) if it exists; its labels are never read here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import load_config  # noqa: E402
from pathsnap.data.dataset import partition_dir, process_trip  # noqa: E402
from pathsnap.data.loader import EXTENSIONS, DataFormatError, list_trips, load_trip, trip_files  # noqa: E402
from pathsnap.data.validator import validate_trip  # noqa: E402
from pathsnap.features.candidate_generation import generate_candidates  # noqa: E402
from pathsnap.visualisation.plots import plt, save  # noqa: E402

OUT = ROOT / "reports" / "eda"
FIG = OUT / "figures"
RADII = [10, 15, 25, 35, 50, 75, 100]
KS = [2, 4, 6, 8, 10, 12, 16, 24]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_cfg = load_config("data")
    feat_cfg = load_config("features")
    inv_rows, val_rows, traj_rows, sweep_rows = [], [], [], []
    dt_all = {"train": [], "student_test": []}
    gps_route_d = {"train": [], "student_test": []}
    reps = {}
    for part in ["train", "student_test", "evaluator_test"]:
        pdir = partition_dir(part)
        for tid in list_trips(pdir):
            files = trip_files(pdir / tid)
            row = {"partition": part, "trip_id": tid}
            for ext in EXTENSIONS:
                row[f"has_{ext}"] = files[ext].exists()
                row[f"{ext}_bytes"] = files[ext].stat().st_size if files[ext].exists() else 0
            row["missing_files"] = ",".join(e for e in EXTENSIONS if not files[e].exists())
            if part == "evaluator_test":
                inv_rows.append(row)          # inventory only - never open evaluator labels
                continue
            try:
                trip = load_trip(pdir / tid, with_route=True, partition=part)
            except DataFormatError as exc:
                row["load_error"] = str(exc)
                inv_rows.append(row)
                continue
            v = validate_trip(trip, data_cfg["preprocessing"]["gap_threshold_s"])
            row.update({"n_obs": v["n_obs"], "n_nodes": v["n_nodes"], "n_arcs": v["n_arcs"],
                        "n_route": v.get("route_len", 0)})
            inv_rows.append(row)
            val_rows.append(v)
            t = trip.track[:, 2]
            dt_all[part].append(np.diff(t))

            ctx = process_trip(trip, with_labels=True, feat_cfg=feat_cfg, data_cfg=data_cfg)
            px, py, tt = ctx.px, ctx.py, ctx.track.t
            step = np.hypot(np.diff(px), np.diff(py))
            dtc = np.diff(tt)
            spd = np.where(dtc > 0, step / np.maximum(dtc, 1e-9), np.nan)
            route = trip.route
            route_len = float(ctx.net.length[route].sum())
            gps_route_d[part].append(ctx.label_dist)
            traj_rows.append({
                "partition": part, "trip_id": tid, "n_obs_clean": len(ctx.track),
                "gps_path_len_m": float(step.sum()), "route_len_m": route_len,
                "duration_s": float(tt[-1] - tt[0]), "median_speed_mps": float(np.nanmedian(spd)) if len(spd) else 0,
                "p95_speed_mps": float(np.nanpercentile(spd, 95)) if len(spd) else 0,
                "max_step_m": float(step.max()) if len(step) else 0,
                "bbox_w_km": float((px.max() - px.min()) / 1000), "bbox_h_km": float((py.max() - py.min()) / 1000),
                "utm_epsg": ctx.projector.epsg,
                "gps_to_route_median_m": float(np.median(ctx.label_dist)),
                "gps_to_route_p95_m": float(np.percentile(ctx.label_dist, 95)),
                "frac_obs_unlabelled": float((ctx.obs_labels < 0).mean()),
                "mean_arc_len_m": float(ctx.net.length.mean()),
                "route_mean_arc_len_m": float(ctx.net.length[route].mean()),
                "candidate_miss_rate": ctx.coverage["miss_rate"],
                "mean_candidates": float(ctx.cands.groupby("obs").size().mean()),
            })
            if part == "train":
                big = generate_candidates(px, py, ctx.net, radius_m=max(RADII), max_candidates=max(KS),
                                          fallback_radius_m=None)
                lab = ctx.obs_labels
                labelled = lab >= 0
                is_true = big["arc"].to_numpy() == lab[big["obs"].to_numpy()]
                for R in RADII:
                    for K in KS:
                        sel = (big["dist_m"].to_numpy() <= R) & (big["dist_rank"].to_numpy() < K)
                        hit = np.zeros(len(lab), bool)
                        hit[big["obs"].to_numpy()[sel & is_true]] = True
                        nc = np.bincount(big["obs"].to_numpy()[sel], minlength=len(lab))
                        sweep_rows.append({"trip_id": tid, "radius_m": R, "k": K,
                                           "n_labelled": int(labelled.sum()),
                                           "n_missed": int((~hit[labelled]).sum()),
                                           "candidates_sum": int(nc.sum())})
            if tid in ("00000052", "00000009", "00000056", "00000065"):
                reps[tid] = ctx
            print(f"[eda] {part}/{tid}: obs={v['n_obs']} arcs={v['n_arcs']} miss={ctx.coverage['miss_rate']:.4f}", flush=True)

    inv = pd.DataFrame(inv_rows)
    val = pd.DataFrame(val_rows)
    traj = pd.DataFrame(traj_rows)
    sweep = pd.DataFrame(sweep_rows)
    inv.to_csv(OUT / "dataset_inventory.csv", index=False)
    ts_cols = ["partition", "trip_id", "n_obs", "n_invalid_time", "n_duplicate_timestamps", "timestamps_monotonic",
               "n_nonpositive_dt", "dt_min", "dt_median", "dt_max", "frac_dt_1s", "n_gaps_over_threshold", "duration_s"]
    val[ts_cols].to_csv(OUT / "timestamp_analysis.csv", index=False)
    net_cols = ["partition", "trip_id", "n_nodes", "n_arcs", "n_isolated_nodes", "n_weak_components",
                "largest_weak_component_frac", "n_strong_components", "largest_strong_component_frac",
                "n_self_loops", "n_duplicate_arcs", "two_way_arc_frac", "route_len", "route_unique_arcs",
                "route_invalid_ids", "route_breaks", "route_continuity"]
    val[net_cols].to_csv(OUT / "network_connectivity.csv", index=False)
    val.to_csv(OUT / "data_validation_full.csv", index=False)
    traj.to_csv(OUT / "trajectory_statistics.csv", index=False)
    agg = sweep.groupby(["radius_m", "k"]).sum(numeric_only=True).reset_index()
    agg["miss_rate"] = agg["n_missed"] / agg["n_labelled"]
    agg["mean_candidates"] = agg["candidates_sum"] / agg["n_labelled"]
    agg.to_csv(OUT / "candidate_radius_sweep_train.csv", index=False)

    make_figures(inv, val, traj, agg, dt_all, gps_route_d, reps)
    write_report(inv, val, traj, agg)


def make_figures(inv, val, traj, agg, dt_all, gps_route_d, reps):
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"train": "#2a6fdb", "student_test": "#e8833a"}
    # 1. observation counts & trip length
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for p in ["train", "student_test"]:
        s = inv[inv.partition == p]
        ax[0].hist(s.n_obs, bins=30, alpha=.7, label=p, color=colors[p])
        t = traj[traj.partition == p]
        ax[1].hist(t.route_len_m / 1000, bins=30, alpha=.7, label=p, color=colors[p])
        ax[2].hist(t.duration_s / 60, bins=30, alpha=.7, label=p, color=colors[p])
    ax[0].set(title="GPS observations per trip", xlabel="observations", ylabel="trips")
    ax[1].set(title="Reference route length", xlabel="km")
    ax[2].set(title="Trip duration", xlabel="minutes")
    ax[0].legend()
    save(fig, FIG / "01_trip_size_distributions.png")
    # 2. timestamp intervals
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    dt = np.concatenate(dt_all["train"] + dt_all["student_test"])
    vals, cnts = np.unique(np.clip(dt, 0, 11), return_counts=True)
    ax[0].bar([str(int(v)) if v < 11 else ">10" for v in vals], cnts / cnts.sum(), color="#2a6fdb")
    ax[0].set(title="Sampling interval (all 90 trips)", xlabel="dt (s)", ylabel="fraction of steps")
    big = dt[dt > 10]
    ax[1].hist(np.log10(big), bins=40, color="#c94f7c")
    ax[1].set(title=f"GPS gaps > 10 s (n={len(big)})", xlabel="log10(gap seconds)", ylabel="count")
    save(fig, FIG / "02_timestamp_intervals.png")
    # 3. coordinate distribution (world)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for p in ["train", "student_test"]:
        s = val[val.partition == p]
        ax.scatter((s.lon_min + s.lon_max) / 2, (s.lat_min + s.lat_max) / 2, s=25, label=p, color=colors[p], alpha=.8)
    ax.set(xlim=(-180, 180), ylim=(-60, 80), xlabel="longitude", ylabel="latitude",
           title="Trip centroids: data is world-wide -> per-trip UTM projection")
    ax.legend()
    save(fig, FIG / "03_trip_locations.png")
    # 4. network size & connectivity
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].scatter(val.n_nodes, val.n_arcs, c=[colors[p] for p in val.partition], s=18)
    ax[0].set(xscale="log", yscale="log", title="Network size per trip", xlabel="nodes", ylabel="arcs")
    ax[1].hist(val.largest_strong_component_frac, bins=25, color="#2e9e6b")
    ax[1].set(title="Largest strongly-connected component", xlabel="fraction of non-isolated nodes")
    ax[2].hist(val.two_way_arc_frac, bins=25, color="#7b61c9")
    ax[2].set(title="Fraction of two-way arcs", xlabel="fraction")
    save(fig, FIG / "04_network_size_connectivity.png")
    # 5. data quality bars
    fig, ax = plt.subplots(figsize=(9, 3.6))
    q = val[["n_invalid_coord", "n_invalid_time", "n_duplicate_rows", "n_duplicate_timestamps",
             "n_nonpositive_dt", "n_gaps_over_threshold", "route_breaks", "n_isolated_nodes"]].sum()
    ax.barh(q.index, q.values, color="#e8833a")
    for i, v in enumerate(q.values):
        ax.text(v, i, f" {int(v)}", va="center")
    ax.set(title="Data-quality issue counts (all 90 trips)", xlabel="count", xscale="symlog")
    save(fig, FIG / "05_data_quality_counts.png")
    # 6. GPS to reference distance
    fig, ax = plt.subplots(figsize=(7, 3.6))
    d = np.concatenate(gps_route_d["train"])
    ax.hist(np.clip(d, 0, 80), bins=80, color="#2a6fdb")
    ax.axvline(60, color="k", ls="--", lw=1)
    ax.set(title=f"Distance GPS fix -> aligned reference arc (train)\nmedian {np.median(d):.1f} m, "
                 f"p95 {np.percentile(d, 95):.1f} m", xlabel="metres (clipped at 80)", ylabel="fixes")
    save(fig, FIG / "06_gps_to_reference_distance.png")
    # 7. candidate sweep
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for R in sorted(agg.radius_m.unique()):
        s = agg[agg.radius_m == R]
        ax[0].plot(s.k, 100 * s.miss_rate, marker="o", label=f"R={R} m")
        ax[1].plot(s.k, s.mean_candidates, marker="o", label=f"R={R} m")
    ax[0].set(title="Candidate miss rate (train, no fallback)", xlabel="max candidates K", ylabel="miss rate (%)",
              yscale="log")
    ax[1].set(title="Mean candidates per fix", xlabel="K", ylabel="candidates")
    ax[0].legend(fontsize=8, ncol=2)
    save(fig, FIG / "07_candidate_radius_sweep.png")
    # 8. representative trajectories with road network
    fig, axes = plt.subplots(1, len(reps), figsize=(4.2 * len(reps), 4.2))
    for ax, (tid, ctx) in zip(np.atleast_1d(axes), reps.items()):
        net = ctx.net
        pad = 150
        bx = (ctx.px.min() - pad, ctx.px.max() + pad, ctx.py.min() - pad, ctx.py.max() + pad)
        m = (np.maximum(net.ax, net.bx) > bx[0]) & (np.minimum(net.ax, net.bx) < bx[1]) & \
            (np.maximum(net.ay, net.by) > bx[2]) & (np.minimum(net.ay, net.by) < bx[3])
        from matplotlib.collections import LineCollection
        segs = np.stack([np.stack([net.ax[m], net.ay[m]], 1), np.stack([net.bx[m], net.by[m]], 1)], 1)
        ax.add_collection(LineCollection(segs, colors="#bbbbbb", linewidths=.5))
        r = ctx.trip.route
        rsegs = np.stack([np.stack([net.ax[r], net.ay[r]], 1), np.stack([net.bx[r], net.by[r]], 1)], 1)
        ax.add_collection(LineCollection(rsegs, colors="#2e9e6b", linewidths=2))
        ax.scatter(ctx.px, ctx.py, s=2, color="#2a6fdb", zorder=3)
        ax.set(xlim=bx[:2], ylim=bx[2:], aspect="equal", title=f"{ctx.trip.partition}/{tid}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Raw GPS (blue), road network (grey), reference route (green) - projected metres")
    save(fig, FIG / "08_representative_trajectories.png")


def write_report(inv, val, traj, agg):
    n_tr = (inv.partition == "train").sum()
    n_st = (inv.partition == "student_test").sum()
    n_ev = (inv.partition == "evaluator_test").sum()
    missing = inv[inv.missing_files.fillna("") != ""]
    best = agg[(agg.radius_m == 50) & (agg.k == 10)].iloc[0]
    lines = [
        "# Data-Quality Report (generated by scripts/run_eda.py)",
        "",
        "## Inventory",
        f"* Training trips: **{n_tr}**, student-testing trips: **{n_st}**, evaluator-testing trips present: **{n_ev}**.",
        f"* Raw files found: {int(inv[[f'has_{e}' for e in EXTENSIONS]].sum().sum())}; trips with missing files: {len(missing)}.",
        f"* GPS observations: train {int(inv[inv.partition=='train'].n_obs.sum())}, student_test {int(inv[inv.partition=='student_test'].n_obs.sum())}.",
        f"* Observations per trip: min {int(val.n_obs.min())}, median {int(val.n_obs.median())}, max {int(val.n_obs.max())}.",
        f"* Network size per trip: nodes {int(val.n_nodes.min())}-{int(val.n_nodes.max())}, arcs {int(val.n_arcs.min())}-{int(val.n_arcs.max())}.",
        "",
        "## GPS observations",
        f"* Invalid coordinates (NaN/out of range): {int(val.n_invalid_coord.sum())}. Non-numeric timestamps: {int(val.n_invalid_time.sum())}.",
        f"* Duplicate rows: {int(val.n_duplicate_rows.sum())}; duplicate timestamps: {int(val.n_duplicate_timestamps.sum())}.",
        f"* Trips with non-monotonic timestamps: {int((~val.timestamps_monotonic).sum())}.",
        f"* Median fraction of 1-second steps per trip: {val.frac_dt_1s.median():.3f}; minimum {val.frac_dt_1s.min():.3f}.",
        f"* Steps with dt > 30 s (GPS gaps): {int(val.n_gaps_over_threshold.sum())} across {int((val.n_gaps_over_threshold>0).sum())} trips; largest gap {val.dt_max.max():.0f} s.",
        f"* Trips are spread world-wide (lon {val.lon_min.min():.1f}..{val.lon_max.max():.1f}, lat {val.lat_min.min():.1f}..{val.lat_max.max():.1f});"
        " each trip is projected into the UTM zone of its centroid.",
        "",
        "## Road networks",
        f"* Isolated nodes (no arc): {int(val.n_isolated_nodes.sum())} in total.",
        f"* Weakly-connected components per trip (excluding isolated nodes): median {val.n_weak_components.median():.0f}, max {int(val.n_weak_components.max())}.",
        f"* Largest strongly-connected component covers median {100*val.largest_strong_component_frac.median():.1f}% of active nodes.",
        f"* Two-way arcs (reverse arc exists): median {100*val.two_way_arc_frac.median():.1f}% of arcs. Self-loops: {int(val.n_self_loops.sum())}, duplicate arcs: {int(val.n_duplicate_arcs.sum())}.",
        "",
        "## Reference routes",
        f"* Route arc ids out of range: {int(val.route_invalid_ids.sum())}.",
        f"* Consecutive route arcs that do not share a node (route breaks): {int(val.route_breaks.sum())} across {int((val.route_breaks>0).sum())} trips.",
        f"* Routes contain fewer entries than GPS fixes (median {np.median(val.n_obs/val.route_len):.1f} fixes per route arc) -> labels per fix must be derived (monotone alignment, see docs/methodology.md).",
        f"* After alignment, median GPS-to-reference distance is {traj.gps_to_route_median_m.median():.1f} m (per-trip median); "
        f"{100*traj.frac_obs_unlabelled.mean():.2f}% of fixes are farther than 60 m from their aligned arc and are marked unlabelled.",
        "",
        "## Candidate generation",
        f"* With radius 50 m and K = 10 (no fallback) the training candidate miss rate is **{100*best.miss_rate:.2f}%** with {best.mean_candidates:.1f} candidates per fix on average.",
        "* Full sweep: `candidate_radius_sweep_train.csv`, figure `figures/07_candidate_radius_sweep.png`.",
        "",
        "## Decisions taken",
        "1. Invalid coordinates are dropped, rows are re-sorted by timestamp, duplicate timestamps keep the first fix.",
        "2. Distances use per-trip UTM coordinates (never raw degrees).",
        "3. GPS gaps > 30 s are flagged; the HMM breaks its chain on gaps it cannot bridge.",
        "4. Fixes whose aligned reference arc is > 60 m away get label -1 and are excluded from supervised rows and accuracy denominators.",
        "5. Reference routes are treated as reference, not error-free truth (route breaks exist).",
    ]
    (OUT / "data_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
