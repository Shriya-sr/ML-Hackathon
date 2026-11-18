"""Explanatory figures (README / docs), drawn from REAL pipeline output on student-testing trips.

Usage:   python scripts/make_explainer_figures.py [--trip 00000077] [--banner-trip 00000052]
Outputs: reports/figures/explainer_{hmm_zoom,one_fix,trellis,groupkfold}.png, docs/assets/banner.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matplotlib.patches import FancyArrowPatch  # noqa: E402

from pathsnap.config import data_dir, load_config  # noqa: E402
from pathsnap.data.labeling import align_to_route  # noqa: E402
from pathsnap.data.loader import read_route  # noqa: E402
from pathsnap.inference.pipeline import MapMatcher  # noqa: E402
from pathsnap.visualisation.plots import plt, save  # noqa: E402

OUT = ROOT / "reports" / "figures"
ASSETS = ROOT / "docs" / "assets"
BLUE, ORANGE, GREEN, GREY, RED = "#2f74e0", "#f07b1d", "#1fa45a", "#b9bec6", "#d64545"


def pick_window(clf_ok, hmm_ok, lab_ok, w=26, px=None, py=None, min_extent=0.0):
    """Window where the HMM corrects the most classifier mistakes and is itself (almost) always right."""
    best, best_i = -1, 0
    for i in range(0, len(clf_ok) - w):
        s = slice(i, i + w)
        if lab_ok[s].sum() < w:
            continue
        if min_extent and max(np.ptp(px[s]), np.ptp(py[s])) < min_extent:
            continue
        gain = hmm_ok[s].sum() - clf_ok[s].sum()
        score = gain * 10 + hmm_ok[s].sum()
        if hmm_ok[s].sum() >= w - 1 and score > best:
            best, best_i = score, i
    return best_i


def draw_arcs(ax, net, arcs, **kw):
    for a in arcs:
        ax.plot([net.ax[a], net.bx[a]], [net.ay[a], net.by[a]], solid_capstyle="round", **kw)


def nearby_arcs(net, x0, x1, y0, y1):
    mx, my = (net.ax + net.bx) / 2, (net.ay + net.by) / 2
    return np.where((mx > x0) & (mx < x1) & (my > y0) & (my < y1))[0]


def arrow(ax, net, a, color, lw):
    x0, y0, x1, y1 = net.ax[a], net.ay[a], net.bx[a], net.by[a]
    xm, ym = x0 + 0.55 * (x1 - x0), y0 + 0.55 * (y1 - y0)
    ax.add_patch(FancyArrowPatch((x0 + 0.35 * (x1 - x0), y0 + 0.35 * (y1 - y0)), (xm, ym), arrowstyle="-|>",
                                 mutation_scale=11, color=color, lw=0))
    _ = lw


def banner(tid: str):
    """README banner: the real GPS fixes and the real HMM route of one student-test trip, on its road network."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    pred = MapMatcher().predict(data_dir() / "student_test" / tid)
    ctx, net = pred.ctx, pred.ctx.net
    px, py = ctx.px, ctx.py
    aspect = 16 / 4.45
    H = np.ptp(py) * 1.22
    W = max(H * aspect, np.ptp(px) / 0.42)          # the trip fills the right ~42% of the banner
    H = W / aspect
    cy = (py.min() + py.max()) / 2
    x1 = px.max() + 0.05 * W
    x0, y0, y1 = x1 - W, cy - H / 2, cy + H / 2
    fig = plt.figure(figsize=(16, 4.45), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("#0b1f3a"); fig.patch.set_facecolor("#0b1f3a")
    mx, my = (net.ax + net.bx) / 2, (net.ay + net.by) / 2
    sel = np.where((mx > x0 - 300) & (mx < x1 + 300) & (my > y0 - 300) & (my < y1 + 300))[0]
    from matplotlib.collections import LineCollection
    segs = np.stack([np.stack([net.ax[sel], net.ay[sel]], 1), np.stack([net.bx[sel], net.by[sel]], 1)], 1)
    ax.add_collection(LineCollection(segs, colors="#1e3d66", linewidths=0.9, zorder=1))
    route = pred.route[pred.route >= 0]
    rsegs = np.stack([np.stack([net.ax[route], net.ay[route]], 1), np.stack([net.bx[route], net.by[route]], 1)], 1)
    ax.add_collection(LineCollection(rsegs, colors="#0fa3a3", linewidths=11, alpha=0.18, zorder=2, capstyle="round"))
    ax.add_collection(LineCollection(rsegs, colors="#19c2c2", linewidths=3.6, zorder=3, capstyle="round"))
    ax.scatter(px, py, s=7, color="#8cc0ff", alpha=0.9, zorder=4, linewidths=0)
    ax.scatter([px[0]], [py[0]], s=110, color="#f07b1d", zorder=5); ax.scatter([px[-1]], [py[-1]], s=110, color="#1fa45a", zorder=5)
    ax.set(xlim=(x0, x1), ylim=(y0, y1)); ax.set_aspect("auto"); ax.axis("off")
    fig.text(0.035, 0.62, "PathSnap", color="white", fontsize=50, fontweight="bold", family="DejaVu Sans")
    fig.text(0.037, 0.46, "Map-matching noisy GPS to the road a car really drove", color="#c9d6e8", fontsize=17)
    fig.text(0.037, 0.33, "classifier  →  HMM + Viterbi  →  FastAPI  →  live React map", color="#0fd0d0", fontsize=13.5)
    fig.text(0.037, 0.12, f"real data: student-test trip {tid}  ·  blue = raw GPS fixes  ·  teal = our decoded route",
             color="#7f93ad", fontsize=10)
    fig.savefig(ASSETS / "banner.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", default="00000077")
    ap.add_argument("--banner-trip", default="00000065")
    a = ap.parse_args()
    banner(a.banner_trip)
    OUT.mkdir(parents=True, exist_ok=True)
    d = data_dir() / "student_test" / a.trip
    mm = MapMatcher()
    pred = mm.predict(d)
    ctx, net = pred.ctx, pred.ctx.net
    lc = load_config("data")["labelling"]
    lab, _, _ = align_to_route(ctx.px, ctx.py, net, read_route(d / f"{a.trip}.route"), lc["distance_cap_m"],
                               lc["max_label_distance_m"])
    lab_ok = lab >= 0
    clf_ok = (pred.clf_arc == lab) & lab_ok
    hmm_ok = (pred.hmm_arc == lab) & lab_ok
    i_slow = pick_window(clf_ok, hmm_ok, lab_ok)
    i0 = pick_window(clf_ok, hmm_ok, lab_ok, w=40, px=ctx.px, py=ctx.py, min_extent=180)
    idx = np.arange(i0, i0 + 40)
    px, py = ctx.px[idx], ctx.py[idx]
    pad = 45
    x0, x1, y0, y1 = px.min() - pad, px.max() + pad, py.min() - pad, py.max() + pad
    near = nearby_arcs(net, x0, x1, y0, y1)
    c = ctx.cands.assign(p=pred.prob)

    # ---------- 1. classifier alone vs classifier + HMM (zoom)
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, arcs, ok, col, title in [(axs[0], pred.clf_arc[idx], clf_ok[idx], ORANGE, "Classifier alone (each fix decided on its own)"),
                                     (axs[1], pred.hmm_arc[idx], hmm_ok[idx], GREEN, "Classifier + HMM/Viterbi (whole sequence)")]:
        draw_arcs(ax, net, near, color=GREY, lw=1.2, zorder=1)
        draw_arcs(ax, net, np.unique(lab[idx][lab[idx] >= 0]), color="#9fd8b5", lw=7, alpha=0.55, zorder=2)
        good = np.unique(arcs[ok & (arcs >= 0)]); bad = np.unique(arcs[~ok & (arcs >= 0)])
        draw_arcs(ax, net, good, color=col, lw=3.2, zorder=3)
        draw_arcs(ax, net, bad, color=RED, lw=3.2, zorder=3)
        for b in bad:
            arrow(ax, net, b, RED, 3)
        ax.plot(px, py, "-", color=BLUE, lw=0.8, alpha=0.6, zorder=4)
        ax.scatter(px, py, s=26, color=BLUE, edgecolor="white", lw=0.8, zorder=5)
        ax.scatter(px[~ok], py[~ok], s=70, facecolor="none", edgecolor=RED, lw=1.6, zorder=6)
        ax.set(xlim=(x0, x1), ylim=(y0, y1), title=f"{title}\n{int(ok.sum())}/{len(ok)} fixes on the correct road")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    axs[0].plot([], [], color="#9fd8b5", lw=7, alpha=0.7, label="reference road (truth)")
    axs[0].plot([], [], color=RED, lw=3, label="wrong road / wrong direction picked")
    axs[0].scatter([], [], s=26, color=BLUE, label="GPS fix (1 per second)")
    axs[0].legend(loc="lower left", fontsize=8, framealpha=0.95)
    axs[0].plot([x0 + 10, x0 + 60], [y1 - 12, y1 - 12], color="black", lw=2)
    axs[0].text(x0 + 35, y1 - 9, "50 m", ha="center", fontsize=8)
    fig.suptitle(f"Real example (car moving): student-test trip {a.trip}, fixes {i0}-{i0 + 39}", fontsize=11)
    save(fig, OUT / "explainer_hmm_zoom.png")

    # ---------- 2. one fix: its candidates and probabilities
    idx_s = np.arange(i_slow, i_slow + 26)
    cand_counts = c[c.obs.isin(idx_s)].groupby("obs").size()
    wrong = [o for o in idx_s if not clf_ok[o] and hmm_ok[o] and cand_counts.get(o, 0) >= 3]
    f = wrong[len(wrong) // 2] if wrong else int(cand_counts.idxmax())
    cf = c[c.obs == f].sort_values("p", ascending=False)
    fx, fy = ctx.px[f], ctx.py[f]
    r = 55
    near2 = nearby_arcs(net, fx - r - 20, fx + r + 20, fy - r - 20, fy + r + 20)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    draw_arcs(ax, net, near2, color=GREY, lw=1.2, zorder=1)
    ax.add_patch(plt.Circle((fx, fy), 50, fill=False, ls="--", color="#555", lw=1))
    ax.text(fx, fy + 52, "50 m search radius", ha="center", fontsize=8, color="#555")
    cmap = plt.get_cmap("viridis")
    letters = "ABCDEFGHIJ"
    for k, (_, row) in enumerate(cf.iterrows()):
        aa = int(row.arc)
        col = cmap(0.15 + 0.8 * row.p)
        ax.plot([net.ax[aa], net.bx[aa]], [net.ay[aa], net.by[aa]], color=col, lw=4, zorder=3)
        arrow(ax, net, aa, col, 4)
        mxx, myy = (net.ax[aa] + net.bx[aa]) / 2, (net.ay[aa] + net.by[aa]) / 2
        tag = letters[k]
        extra = "  <- HMM choice = truth" if aa == pred.hmm_arc[f] else ("  <- classifier's pick" if aa == pred.clf_arc[f] else "")
        if pred.hmm_arc[f] >= 0 and net.reverse_arc[pred.hmm_arc[f]] == aa:
            extra += " (same street, OPPOSITE direction)"
        ax.annotate(f"{tag}: P={row.p:.2f}, {row.dist_m:.0f} m{extra}", (mxx, myy), xytext=(8, 6 - 13 * k),
                    textcoords="offset points", fontsize=8, color="#222",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=1, alpha=0.9), zorder=6)
    ax.plot(ctx.px[f - 6:f + 7], ctx.py[f - 6:f + 7], "-", color=BLUE, lw=0.8, alpha=0.5)
    ax.scatter(ctx.px[f - 6:f + 7], ctx.py[f - 6:f + 7], s=14, color=BLUE, alpha=0.5)
    ax.scatter([fx], [fy], s=120, color=BLUE, edgecolor="white", lw=1.5, zorder=7, marker="o")
    ax.text(fx, fy - 9, f"fix #{f}", ha="center", va="top", fontsize=9, color=BLUE, fontweight="bold")
    ax.set(xlim=(fx - r - 15, fx + r + 15), ylim=(fy - r - 15, fy + r + 15),
           title=f"One GPS fix, {len(cf)} candidate road segments,\nand the classifier's probability for each")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    save(fig, OUT / "explainer_one_fix.png")

    # ---------- 3. Viterbi trellis with real probabilities
    seq = [o for o in range(f - 3, f + 3)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    names, cols = {}, []
    for j, o in enumerate(seq):
        co = c[c.obs == o].sort_values("p", ascending=False).head(4)
        cols.append(co)
        for aa in co.arc:
            if int(aa) not in names:
                names[int(aa)] = chr(65 + len(names)) if len(names) < 26 else "?"
    pos = {}
    for j, co in enumerate(cols):
        for k, (_, row) in enumerate(co.iterrows()):
            pos[(j, int(row.arc))] = (j, -k)
    for j in range(len(cols) - 1):
        for (jj, a1), (x1_, y1_) in pos.items():
            if jj != j:
                continue
            for (j2, a2), (x2_, y2_) in pos.items():
                if j2 == j + 1:
                    ax.plot([x1_, x2_], [y1_, y2_], color="#dde1e6", lw=0.8, zorder=1)
    path = [(j, int(pred.hmm_arc[o])) for j, o in enumerate(seq)]
    for (j, aa), (j2, a2) in zip(path[:-1], path[1:]):
        if (j, aa) in pos and (j2, a2) in pos:
            (xa, ya), (xb, yb) = pos[(j, aa)], pos[(j2, a2)]
            ax.plot([xa, xb], [ya, yb], color=GREEN, lw=3.5, zorder=2)
    for j, co in enumerate(cols):
        o = seq[j]
        for k, (_, row) in enumerate(co.iterrows()):
            aa = int(row.arc)
            x, y = pos[(j, aa)]
            is_h, is_c = aa == pred.hmm_arc[o], aa == pred.clf_arc[o]
            ax.scatter([x], [y], s=900, color=cmap(0.15 + 0.8 * row.p), edgecolor=GREEN if is_h else "white",
                       lw=3 if is_h else 1, zorder=3)
            ax.text(x, y + 0.03, f"{names[aa]}", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold", zorder=4)
            ax.text(x, y - 0.03, f"{row.p:.2f}", ha="center", va="top", fontsize=7.5, color="white", zorder=4)
            if is_c and not is_h:
                ax.scatter([x], [y], s=1300, facecolor="none", edgecolor=ORANGE, lw=2, ls="--", zorder=3)
        ax.text(j, 0.62, f"fix {o}", ha="center", fontsize=9, color="#333")
        ax.text(j, -len(co) + 0.35, "classifier wrong" if not clf_ok[o] else "classifier right", ha="center", fontsize=7.5,
                color=RED if not clf_ok[o] else "#666")
    ax.plot([], [], color=GREEN, lw=3.5, label="Viterbi best path (HMM answer)")
    ax.scatter([], [], s=80, facecolor="none", edgecolor=ORANGE, ls="--", lw=2, label="classifier's own top pick (when different)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5, frameon=False)
    ax.set(xlim=(-0.6, len(seq) - 0.4), ylim=(-4.2, 0.9))
    ax.axis("off")
    ax.set_title("Viterbi trellis (real numbers): columns = GPS fixes, circles = candidate roads (letter = road, number = classifier P)",
                 fontsize=10)
    save(fig, OUT / "explainer_trellis.png")

    # ---------- 4. GroupKFold picture
    rng = np.random.default_rng(42)
    fold_of = rng.permutation(np.repeat(np.arange(5), 16))
    fig, axs = plt.subplots(5, 1, figsize=(11, 3.6), sharex=True)
    for k in range(5):
        for t in range(80):
            axs[k].add_patch(plt.Rectangle((t, 0), 0.85, 0.85, color=ORANGE if fold_of[t] == k else "#4d8fe8"))
        axs[k].set(xlim=(-0.5, 80.5), ylim=(-0.1, 1)); axs[k].axis("off")
        axs[k].text(-1.5, 0.4, f"Fold {k + 1}", ha="right", va="center", fontsize=9)
    fig.suptitle("5-fold GroupKFold by trip: every square is one whole training trip. Orange = validation trips in that fold, "
                 "blue = training trips.\nA trip is never split, so the model is always scored on journeys it has never seen.",
                 fontsize=9.5)
    save(fig, OUT / "explainer_groupkfold.png")
    print("fix", f, "slow window", i_slow, "moving window", i0, "candidates", len(cf),
          "clf arc is reverse of hmm arc:", bool(net.reverse_arc[pred.hmm_arc[f]] == pred.clf_arc[f]))


if __name__ == "__main__":
    main()
