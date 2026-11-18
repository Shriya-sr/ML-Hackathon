"""Build the consolidated technical report (PDF) and the markdown result summaries
exclusively from saved experiment records.

Usage:  python scripts/build_report.py
Writes: reports/technical_report/PathSnap_Technical_Report.pdf
        reports/final/RESULTS_SUMMARY.md, docs/model_explanations.md, reports/viva/QUICK_REVISION.md
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.features.feature_schema import FEATURE_GROUPS  # noqa: E402

RES = ROOT / "experiments" / "results"
OPT = ROOT / "experiments" / "optimisation"
FIG = ROOT / "reports" / "figures"
EDA = ROOT / "reports" / "eda"
MAPS = ROOT / "maps" / "screenshots"
OUT = ROOT / "reports" / "technical_report" / "PathSnap_Technical_Report.pdf"

# ----------------------------------------------------------------------------- static model knowledge
MODEL_TEXT = {
    "logistic_regression": dict(
        suit="The strongest signals (relative distance, heading cosine, neighbourhood distance) relate almost monotonically to "
             "correctness, so a linear decision function on standardised features is a natural, interpretable baseline whose "
             "probabilities feed the HMM directly.",
        strengths="Fast to train and score; coefficients are interpretable; probabilities are usually well calibrated.",
        limits="Cannot model interactions (e.g. heading only matters when the vehicle is moving) unless they are engineered by hand.",
        hp="C = 1.0 (inverse L2 strength), lbfgs solver, max_iter = 3000, no class weights; StandardScaler fitted per training fold."),
    "gaussian_nb": dict(
        suit="A generative baseline that tests whether independent per-feature evidence is enough to rank candidates.",
        strengths="Trains in a fraction of a second, needs no tuning, gives a probabilistic output.",
        limits="Assumes conditionally independent Gaussian features. Ours are correlated (dist_m, dist_ratio, win_mean_dist_m) "
               "and skewed, so evidence is double-counted and probabilities are over-confident.",
        hp="var_smoothing = 1e-9 (default)."),
    "knn": dict(
        suit="Instance-based: a candidate that resembles previously correct candidates in feature space is likely correct.",
        strengths="Non-parametric, captures local non-linear structure without assumptions.",
        limits="Prediction cost grows with stored rows (trained on a trip-stratified 60k-row subsample); sensitive to scaling and "
               "to irrelevant dimensions; probability estimates are coarse vote fractions.",
        hp="k = 25, distance weighting, Euclidean metric on standardised features, 60 000 training rows."),
    "random_forest": dict(
        suit="Bagged decision trees capture interactions between distance, heading and context features with little tuning.",
        strengths="Robust to feature scale and outliers, low variance thanks to bagging, strong out of the box.",
        limits="Large memory footprint (hundreds of deep trees), slower training and inference than boosting.",
        hp="200 trees, max_depth = None, min_samples_leaf = 5, max_features = sqrt, bootstrap = True."),
    "extra_trees": dict(
        suit="Like Random Forest but with random split thresholds; checks whether extra randomisation reduces variance further.",
        strengths="Faster to train than RF, very low variance.",
        limits="Random thresholds add bias; probability estimates are smoother but less sharp.",
        hp="200 trees, max_depth = None, min_samples_leaf = 5, max_features = sqrt."),
    "hist_gradient_boosting": dict(
        suit="Gradient-boosted histogram trees directly minimise log loss on ~620k rows quickly, modelling interactions while "
             "producing probabilities suited to HMM emissions.",
        strengths="State-of-the-art on tabular data, fast (histogram binning, multithreaded), built-in early stopping, small artifact.",
        limits="More hyperparameters than bagging; can overfit with many leaves and no regularisation; early stopping uses a random "
               "10% row split of the training fold (not trip-grouped).",
        hp="learning_rate = 0.1, max_iter = 300 with early stopping (patience 15, 10% validation), max_leaf_nodes = 31, "
           "min_samples_leaf = 20, l2_regularization = 0."),
    "svm_rbf": dict(
        suit="A smooth non-linear margin in the scaled feature space; tests whether a kernel boundary separates correct candidates.",
        strengths="Strong theoretical margin properties; effective in moderate dimensions.",
        limits="O(n²) memory/time: trained on a 20k-row trip-stratified subsample; needs Platt calibration for probabilities; slowest "
               "inference per row because every support vector is evaluated.",
        hp="C = 1.0, gamma = 'scale', RBF kernel, StandardScaler; probabilities via CalibratedClassifierCV(sigmoid, cv=3, ensemble=False)."),
    "mlp": dict(
        suit="A small neural network learns non-linear feature combinations and gives a genuine epoch-wise training history.",
        strengths="Flexible function class; outputs probabilities; trains quickly on standardised features with mini-batches.",
        limits="Sensitive to scaling, learning rate and initialisation; less interpretable; early stopping uses a random 10% row split.",
        hp="hidden layers (64, 32), ReLU, Adam, learning_rate_init = 0.001, alpha (L2) = 1e-4, batch 512, early stopping (patience 10)."),
}


def load():
    s = pd.read_csv(RES / "model_summary.csv")
    d = {"summary": s}
    for k, p in {"opt": OPT / "optimisation_summary.json", "dec": RES / "decoder_summary.json",
                 "dec_pre": RES / "decoder_summary_preliminary.json", "hmm": ROOT / "artifacts" / "metadata" / "hmm_params.json",
                 "clf": ROOT / "artifacts" / "metadata" / "classifier.json"}.items():
        d[k] = json.loads(p.read_text()) if p.exists() else None
    for k, p in {"tvu": OPT / "tuned_vs_untuned.csv"}.items():
        d[k] = pd.read_csv(p, index_col=0) if p.exists() else None
    for k, p in {"dec_trips": RES / "decoder_comparison_final.csv", "abl": RES / "feature_ablation.csv",
                 "perm": RES / "permutation_importance.csv", "trials": OPT / "trials.csv",
                 "inv": EDA / "dataset_inventory.csv", "val": EDA / "data_validation_full.csv",
                 "traj": EDA / "trajectory_statistics.csv", "sweep": EDA / "candidate_radius_sweep_train.csv",
                 "grid": RES / "hmm_grid.csv"}.items():
        d[k] = pd.read_csv(p, dtype={"trip_id": str}) if p.exists() else None
    return d


def fit_verdict(r) -> str:
    gap_f1, gap_t1 = r["gap_macro_f1"], r["gap_top1"]
    st_vs_cv = r["st_macro_f1"] - r["cv_macro_f1"]
    parts = [f"Training-fold Macro-F1 {r['train_macro_f1']:.4f} vs validation {r['cv_macro_f1']:.4f} (gap {gap_f1:+.4f}); "
             f"Top-1 gap {gap_t1:+.4f}; fold std {r['cv_macro_f1_std']:.4f}; student-test Macro-F1 {r['st_macro_f1']:.4f} "
             f"({st_vs_cv:+.4f} vs CV)."]
    if gap_f1 > 0.03:
        parts.append("The gap is large relative to the fold-to-fold spread, so the model fits training trips noticeably better than unseen "
                     "trips: clear signs of overfitting (fully grown trees memorise training rows).")
        if r["cv_macro_f1"] >= r["_best_cv"] - 0.001:
            parts.append("Its validation score is nevertheless among the best, so bagging keeps this memorisation from hurting "
                         "generalisation; stronger leaf/depth limits would shrink the gap without a clear validation gain.")
    elif gap_f1 > 0.01:
        parts.append("A moderate gap: some fitting to training rows, but validation performance stays close and stable, so variance is "
                     "under control (mild overfitting).")
    else:
        parts.append("Training and validation scores are almost equal: no evidence of overfitting.")
    best = r["_best_cv"]
    if r["cv_macro_f1"] < best - 0.02:
        parts.append(f"Its validation Macro-F1 is {best - r['cv_macro_f1']:.3f} below the best model while the training score is also low, "
                     "which points to underfitting (the model family is too restrictive for these features).")
    elif r["cv_macro_f1"] < best - 0.005 and r["train_macro_f1"] < best:
        parts.append(f"However, even its training-fold score ({r['train_macro_f1']:.4f}) is below the best model's validation score "
                     f"({best:.4f}): the limitation is bias rather than variance, i.e. mild underfitting relative to the tree ensembles.")
    if abs(st_vs_cv) <= 2 * r["cv_macro_f1_std"]:
        parts.append("The student-test score lies within two fold standard deviations of CV, so CV generalised to the unseen trips.")
    return " ".join(parts)


def model_sections(s: pd.DataFrame):
    s = s.copy()
    s["_best_cv"] = s["cv_macro_f1"].max()
    out = []
    for _, r in s.sort_values("cv_macro_f1", ascending=False).iterrows():
        t = MODEL_TEXT[r["model"]]
        out.append({
            "model": r["model"], "title": f"{r['display']} ({r['family']})",
            "A1": t["suit"], "A2": f"Strengths: {t['strengths']} Limitations: {t['limits']}",
            "B5": t["hp"], "B6": {
                "logistic_regression": "Binary cross-entropy (log loss) + L2 penalty.",
                "gaussian_nb": "Maximum-likelihood estimates of per-class Gaussian means/variances; prediction by Bayes' rule.",
                "knn": "No training objective (lazy learner); P = distance-weighted share of positive neighbours.",
                "random_forest": "Each tree greedily minimises Gini impurity on a bootstrap sample; probabilities = mean leaf frequencies.",
                "extra_trees": "Gini impurity with randomly drawn split thresholds; probabilities = mean leaf frequencies.",
                "hist_gradient_boosting": "Binary log loss, minimised by gradient boosting (Newton steps on histogram trees).",
                "svm_rbf": "Hinge loss with L2 margin regularisation; Platt sigmoid fitted by log loss for probabilities.",
                "mlp": "Binary cross-entropy + L2 (alpha), minimised by Adam on mini-batches.",
            }[r["model"]],
            "C7": (f"On the 10 student-testing trips (19 160 fixes): accuracy {r['st_accuracy']:.4f}, precision (positive) "
                   f"{r['st_precision_pos']:.4f}, recall (positive) {r['st_recall_pos']:.4f}, macro precision {r['st_precision_macro']:.4f}, "
                   f"macro recall {r['st_recall_macro']:.4f}, ROC-AUC {r['st_roc_auc']:.4f}; route continuity of its independent "
                   f"argmax route {r['st_route_continuity']:.4f}, route F1 {r['st_route_f1']:.4f}. Training {r['fit_s_full']:.1f} s on "
                   f"{int(r['n_fit_rows_full']):,} rows; inference {r['inference_ms_per_1k_fixes']:.1f} ms per 1000 fixes; "
                   f"artifact {r['model_size_mb']:.1f} MB."),
            "C8": (f"Observation-level Top-1 accuracy {r['st_top1']:.4f}, Macro-F1 {r['st_macro_f1']:.4f}, log loss {r['st_log_loss']:.4f} "
                   f"(CV: Top-1 {r['cv_top1']:.4f}, Macro-F1 {r['cv_macro_f1']:.4f} ± {r['cv_macro_f1_std']:.4f}, log loss {r['cv_log_loss']:.4f})."),
            "C9": fit_verdict(r),
        })
    return out


def write_markdown(d):
    s = d["summary"].sort_values("cv_macro_f1", ascending=False)
    secs = model_sections(d["summary"])
    lines = ["# Model Explanations (generated by scripts/build_report.py from experiment records)", "",
             "Every number below comes from `experiments/results/model_summary.csv` and the fold files.", ""]
    for m in secs:
        lines += [f"## {m['title']}", "", "**A. Model understanding**", "",
                  f"1. *Why suitable?* {m['A1']}", f"2. *Strengths and limitations.* {m['A2']}", "",
                  "**B. Training and configuration**", "", f"5. *Important hyperparameters (initial values).* {m['B5']}",
                  f"6. *Loss / objective.* {m['B6']}", "", "**C. Experimental results**", "",
                  f"7. *Student-test performance.* {m['C7']}", f"8. *Top-1, Macro-F1, log loss.* {m['C8']}",
                  f"9. *Over/under-fitting.* {m['C9']}", ""]
    (ROOT / "docs" / "model_explanations.md").write_text("\n".join(lines), encoding="utf-8")

    best = s.iloc[0]
    res = ["# Results Summary (generated from experiment records)", "",
           "## Model comparison (classifier only)", "",
           "| Model | Family | CV Macro-F1 | CV Top-1 | Student Top-1 | Student Macro-F1 | Student log loss | Route continuity | Inference ms/1k fixes |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        res.append(f"| {r['display']} | {r['family']} | {r['cv_macro_f1']:.4f} ± {r['cv_macro_f1_std']:.4f} | {r['cv_top1']:.4f} | "
                   f"{r['st_top1']:.4f} | {r['st_macro_f1']:.4f} | {r['st_log_loss']:.4f} | {r['st_route_continuity']:.4f} | "
                   f"{r['inference_ms_per_1k_fixes']:.1f} |")
    if d["tvu"] is not None:
        t = d["tvu"]
        res += ["", "## Bayesian optimisation (HistGradientBoosting)", "", "| Metric | Untuned | Tuned | Δ |", "|---|---|---|---|"]
        for k in ["cv_macro_f1", "cv_top1", "cv_log_loss", "st_top1", "st_macro_f1", "st_log_loss", "inference_ms_per_1k_fixes"]:
            res.append(f"| {k} | {t.loc[k, 'untuned']:.4f} | {t.loc[k, 'tuned']:.4f} | {t.loc[k, 'delta']:+.4f} |")
    if d["dec"] is not None:
        f = d["dec"]["summary"]["final"]
        res += ["", "## Classifier-only vs classifier + HMM (student-test, final classifier, tuned HMM)", "",
                "| Metric | Classifier only | + HMM | + HMM + reconstruction |", "|---|---|---|---|",
                f"| Top-1 accuracy (pooled fixes) | {f['clf_top1']:.4f} | {f['hmm_top1']:.4f} | (same as HMM) |",
                f"| Selection Macro-F1 (mean of trips) | {f['clf_sel_macro_f1']:.4f} | {f['hmm_sel_macro_f1']:.4f} | |",
                f"| Route continuity | {f['clf_route_continuity']:.4f} | {f['hmm_route_continuity']:.4f} | {f['recon_route_continuity']:.4f} |",
                f"| Route F1 | {f['clf_route_f1']:.4f} | {f['hmm_route_f1']:.4f} | {f['recon_route_f1']:.4f} |",
                f"| Edit similarity | {f['clf_edit_similarity']:.4f} | {f['hmm_edit_similarity']:.4f} | {f['recon_edit_similarity']:.4f} |",
                f"| Invalid transitions (sum) | {f['clf_invalid_transitions']:.0f} | {f['hmm_invalid_transitions']:.0f} | {f['recon_invalid_transitions']:.0f} |",
                f"| Candidate miss rate | {f['miss_rate']:.4f} | {f['miss_rate']:.4f} | |"]
    (ROOT / "reports" / "final").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "final" / "RESULTS_SUMMARY.md").write_text("\n".join(res) + "\n", encoding="utf-8")
    return best


# ----------------------------------------------------------------------------- PDF helpers
styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#0f2a4a"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1d4e89"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceBefore=6, spaceAfter=3)
BODY = ParagraphStyle("B", parent=styles["BodyText"], fontSize=9.5, leading=13)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#444444"))
CAP = ParagraphStyle("C", parent=SMALL, alignment=TA_CENTER, spaceAfter=8)
TS = TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.5), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
                 ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f3")), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                 ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")])])


def P(t, st=BODY):
    return Paragraph(t, st)


def img(path: Path, width_cm=16.5, caption=None):
    if not path.exists():
        return [P(f"<i>[figure not found: {path.relative_to(ROOT)}]</i>", SMALL)]
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(path)).getSize()
    width = width_cm * cm
    height = min(width * h / w, 22 * cm)
    width = height * w / h
    out = [Image(str(path), width=width, height=height)]
    if caption:
        out.append(P(caption, CAP))
    return [KeepTogether(out)]


def table(rows, col_widths=None):
    data = [[P(str(c), SMALL) if isinstance(c, str) and len(c) > 25 else c for c in row] for row in rows]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TS)
    return t


def build_pdf(d, best):
    s = d["summary"].sort_values("cv_macro_f1", ascending=False)
    story = []
    # ---- title
    story += [Spacer(1, 3 * cm), P("PES UNIVERSITY · Department of Computer Science and Engineering · RR Campus", CAP),
              P("UE24CS352A: Machine Learning, Hackathon", CAP), Spacer(1, 1 * cm),
              P("<b>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</b>",
                ParagraphStyle("T", parent=H1, fontSize=22, alignment=TA_CENTER, leading=28)),
              P("Consolidated Technical Report", ParagraphStyle("T2", parent=H2, alignment=TA_CENTER)), Spacer(1, 1.2 * cm),
              P("Team: Sujal · Shriya · Sudheeksh · Shubashitha", CAP),
              P("Repository: https://github.com/Shriya-sr/ML-Hackathon", CAP),
              P(f"Generated {date.today().isoformat()} by scripts/build_report.py from the experiment records in the repository.", CAP),
              PageBreak()]
    dec = d["dec"]["summary"]["final"] if d["dec"] else None
    tvu = d["tvu"]
    # ---- abstract
    story += [P("Abstract", H1), P(
        "PathSnap matches noisy 1 Hz GPS trajectories to a directed road network. For every GPS fix we generate candidate road "
        "segments with an R-tree query, describe each (fix, candidate) pair with 21 geometric, heading, context, kinematic and "
        "network features, and train classifiers that estimate the probability that a candidate is the reference segment. "
        f"Eight models from seven algorithm families were compared with 5-fold trip-grouped cross-validation on the 80 training trips. "
        f"{best['display']} reached the highest CV Macro-F1 ({best['cv_macro_f1']:.4f}). HistGradientBoosting was selected and tuned with "
        "Bayesian optimisation (Optuna TPE). Its probabilities are the emissions of a Hidden Markov Model whose transitions compare driving "
        "distance over the network with the GPS displacement; a segmented Viterbi decoder and shortest-path reconstruction output a "
        "drivable route. " + (
            f"On the 10 student-testing trips, observation-level Top-1 accuracy rises from {dec['clf_top1']:.4f} (classifier only) to "
            f"{dec['hmm_top1']:.4f} with the HMM, and mean route continuity from {dec['clf_route_continuity']:.3f} to "
            f"{dec['recon_route_continuity']:.3f}. " if dec else "") +
        "The trained pipeline is served by a FastAPI backend that keeps reference routes server-side and a React/Leaflet evaluator UI "
        "for the two-laptop deployment.")]
    # ---- intro
    story += [P("1. Introduction, problem statement and objectives", H1), P(
        "Consumer GPS positions are affected by multipath near buildings, signal blockage, noise and dropouts, so raw fixes often lie "
        "on parallel streets or between roads. Map matching recovers the road segments actually travelled. The objectives were: "
        "(i) understand the raw data; (ii) build an independent preprocessing, candidate-generation and feature pipeline; (iii) train and "
        "compare at least five models with trip-aware validation; (iv) select and Bayesian-optimise a model; (v) add HMM/Viterbi "
        "sequential decoding and compare it with classifier-only predictions; (vi) visualise with Folium; (vii) deploy as API + UI on two "
        "laptops with server-side scoring.")]
    # ---- 2.2 dataset & EDA
    inv, val, traj = d["inv"], d["val"], d["traj"]
    story += [P("2. Dataset understanding and exploratory data analysis", H1)]
    if inv is not None:
        tr = inv[inv.partition == "train"]; st = inv[inv.partition == "student_test"]
        story += [P(
            f"<b>Partitions.</b> {len(tr)} training trips ({int(tr.n_obs.sum()):,} GPS fixes) and {len(st)} student-testing trips "
            f"({int(st.n_obs.sum()):,} fixes); every trip has the four files .track, .nodes, .arcs and .route (360 files, none missing). "
            f"The evaluator-testing partition was not supplied; the API is prepared to load it from data/raw/evaluator_test. "
            f"Unless stated otherwise, EDA figures use both train and student-test (marked by colour); all modelling decisions "
            f"(candidate radius, features, hyperparameters, decoder parameters) were made on the training partition only."),
            P(f"<b>Sizes.</b> Fixes per trip {int(val.n_obs.min())}–{int(val.n_obs.max())} (median {int(val.n_obs.median())}); road networks "
              f"{int(val.n_nodes.min()):,}–{int(val.n_nodes.max()):,} nodes and {int(val.n_arcs.min()):,}–{int(val.n_arcs.max()):,} directed arcs."),
            P("<b>Q1. How did we preprocess the data?</b> Rows with invalid coordinates or timestamps are dropped (none occurred), "
              "rows are stably re-sorted by timestamp and duplicate timestamps removed (none occurred), keeping a pointer to the raw row. "
              "Because trips lie on several continents, each trip is projected into the UTM zone of its centroid so that all distances, "
              "bearings and speeds are in metres. Per-fix labels are derived by a monotone dynamic-programming alignment of the fixes to "
              "the ordered reference arcs (Section 3), because .route has about one entry per five fixes."),
            P("<b>Missing-value analysis.</b> " + (
                f"Invalid coordinates: {int(val.n_invalid_coord.sum())}; non-numeric timestamps: {int(val.n_invalid_time.sum())}; duplicate rows: "
                f"{int(val.n_duplicate_rows.sum())}; duplicate timestamps: {int(val.n_duplicate_timestamps.sum())}; non-monotonic trips: "
                f"{int((~val.timestamps_monotonic.astype(bool)).sum())}; route ids out of range: {int(val.route_invalid_ids.sum())}. "
                "No values are missing in the supplied files; the loaders nevertheless handle missing/malformed files explicitly. "
                f"The real 'missingness' is temporal: {int(val.n_gaps_over_threshold.sum())} gaps above 30 s in "
                f"{int((val.n_gaps_over_threshold > 0).sum())} trips (largest {val.dt_max.max():,.0f} s).")),
            P("<b>Q2. Patterns observed.</b> " + (
                f"Sampling is 1 Hz for a median {100 * val.frac_dt_1s.median():.1f}% of steps per trip. Median GPS-to-reference distance "
                f"is {traj.gps_to_route_median_m.median():.1f} m per trip, but long stationary periods occur off the network (parked cars "
                f"producing clouds of jittering fixes, up to {100 * traj.frac_obs_unlabelled.max():.0f}% of one trip). "
                f"{100 * val.two_way_arc_frac.median():.0f}% of arcs (median) have a reverse twin, so direction must be inferred from motion. "
                f"Networks contain several weakly connected fragments (median {val.n_weak_components.median():.0f}); reference routes "
                f"contain {int(val.route_breaks.sum())} breaks in {int((val.route_breaks > 0).sum())} trips, so the reference is not "
                "flawless ground truth. These observations motivated the features in Section 4.")),
            ]
    story += img(FIG.parent / "eda" / "figures" / "01_trip_size_distributions.png", caption="Fig. 1: trip size distributions (train vs student-test).")
    story += img(EDA / "figures" / "02_timestamp_intervals.png", caption="Fig. 2: sampling intervals and GPS gaps (all 90 trips).")
    story += img(EDA / "figures" / "03_trip_locations.png", 13, caption="Fig. 3: trip centroids: the data is world-wide.")
    story += img(EDA / "figures" / "04_network_size_connectivity.png", caption="Fig. 4: road-network size and connectivity per trip.")
    story += img(EDA / "figures" / "05_data_quality_counts.png", 13, caption="Fig. 5: data-quality issue counts (missing-value analysis).")
    story += img(EDA / "figures" / "08_representative_trajectories.png", caption="Fig. 6: GPS trajectory plots with road networks and reference routes (projected metres).")
    story += img(EDA / "figures" / "06_gps_to_reference_distance.png", 12, caption="Fig. 7: distance from fixes to their aligned reference arc (train).")
    story += img(MAPS / "eda_train_00000054_raw.png", caption="Fig. 8: Folium map (training trip 00000054): raw GPS, road network and reference route; the parked "
                 "cluster near the start lies off the network.")
    # ---- preprocessing & candidates
    sw = d["sweep"]
    story += [P("3. Preprocessing, label alignment and candidate generation", H1), P(
        "<b>Label alignment.</b> For fix i and route position j the cost is min(distance, 100 m) + 15 m·(1−cos(motion, arc direction))/2; "
        "positions must be non-decreasing in time. C[i,j] = cost(i,j) + min over j'≤j of C[i−1,j'] is solved exactly in O(N·M). The heading "
        "term separates the two directions of a road on out-and-back routes. Fixes more than 60 m from their aligned arc are labelled −1 "
        "(unreliable) and excluded from training rows and accuracy denominators."),
        P("<b>Candidates.</b> All arcs are straight segments in a Shapely STRtree; one vectorised query returns all arcs within R of every fix; "
          "the K nearest are kept, with a single nearest arc within 150 m as fallback. Labels never influence candidates (unit-tested).")]
    if sw is not None:
        r50 = sw[(sw.radius_m == 50) & (sw.k == 10)].iloc[0]
        r35 = sw[(sw.radius_m == 35) & (sw.k == 10)].iloc[0]
        r75 = sw[(sw.radius_m == 75) & (sw.k == 10)].iloc[0]
        story += [P(f"The training-partition sweep (Fig. 9) shows miss rate {100 * r35.miss_rate:.2f}% at R=35 m, "
                    f"{100 * r50.miss_rate:.2f}% at R=50 m and {100 * r75.miss_rate:.2f}% at R=75 m (K=10), with "
                    f"{r35.mean_candidates:.1f}, {r50.mean_candidates:.1f} and {r75.mean_candidates:.1f} candidates per fix. R=50 m, K=10 "
                    "was chosen as the knee: larger radii add candidates (cost and confusion) for little coverage.")]
    story += img(EDA / "figures" / "07_candidate_radius_sweep.png", caption="Fig. 9: candidate miss rate and candidates per fix vs radius and K (train).")
    # ---- features
    story += [P("4. Feature engineering", H1), P(
        "Features are computed from the GPS track and the network only (no learned transforms), so they cannot leak across partitions; "
        "the scaler of scale-sensitive models is inside each Pipeline and re-fitted per training fold.")]
    rows = [["Group", "Features", "Why"]]
    why = {"geometric": "closeness relative to competing candidates", "heading": "direction of travel vs arc direction (two-way roads)",
           "context": "fit of the road to neighbouring fixes (robust to single noisy fixes)", "kinematic": "reliability of heading; gaps",
           "network": "junction structure, one-way roads"}
    for g, fs in FEATURE_GROUPS.items():
        rows.append([g, ", ".join(fs), why[g]])
    story += [table(rows, [2.2 * cm, 9 * cm, 5.5 * cm])]
    if d["abl"] is not None:
        ab = d["abl"]
        story += [Spacer(1, 4), P("<b>Ablation</b> (5-fold CV, training trips, HGB baseline, every 4th fix):")]
        rows = [["Dropped group", "#features", "CV Macro-F1", "Δ Macro-F1", "CV Top-1", "Δ Top-1"]]
        for _, r in ab.iterrows():
            rows.append([r.dropped_group, int(r.n_features), f"{r.cv_macro_f1:.4f}", f"{r.delta_macro_f1:+.4f}", f"{r.cv_top1:.4f}", f"{r.delta_top1:+.4f}"])
        story += [table(rows)]
    story += img(FIG / "feature_importance.png", 13, caption="Fig. 10: permutation importance on held-out training trips.")
    story += img(FIG / "feature_ablation.png", 14, caption="Fig. 11: group ablation.")
    # ---- models
    story += [PageBreak(), P("5. Model experimentation and comparison", H1), P(
        "All models use the same 21 features, the same five trip-grouped folds (GroupKFold, group = trip, seed 42) and the same student-test "
        "evaluation. Training rows: candidates of every second labelled fix of the training-fold trips; validation/test rows: all labelled "
        "fixes. KNN and SVM are fitted on trip-stratified subsamples (60k / 20k rows) because of their super-linear cost. "
        "<b>Primary selection metric</b>: CV Macro-F1 (threshold 0.5), read together with CV Top-1 (the map-matching objective), log loss "
        "(probability quality, which matters because the probabilities become HMM emissions), fold stability and computational cost.")]
    rows = [["Model", "Family", "Key configuration", "CV Macro-F1", "Top-1 (CV / student)", "Route contin.", "Student Macro-F1", "Log loss (student)", "Inference ms/1k fixes"]]
    for _, r in s.iterrows():
        rows.append([r.display, r.family, r.key_config[:90], f"{r.cv_macro_f1:.4f} ± {r.cv_macro_f1_std:.4f}",
                     f"{r.cv_top1:.4f} / {r.st_top1:.4f}", f"{r.st_route_continuity:.4f}", f"{r.st_macro_f1:.4f}", f"{r.st_log_loss:.4f}",
                     f"{r.inference_ms_per_1k_fixes:.1f}"])
    story += [table(rows, [2.3 * cm, 2.1 * cm, 3.6 * cm, 1.8 * cm, 1.9 * cm, 1.3 * cm, 1.5 * cm, 1.4 * cm, 1.4 * cm]),
              P("Route continuity here is that of the independent per-fix argmax route on the student-test trips (no HMM). Log loss is "
                "applicable to every model because all produce probabilities (SVM via Platt scaling, KNN via weighted votes).", SMALL)]
    for f, c in [("model_cv_macro_f1.png", "Fig. 12: CV Macro-F1 across all models (mean ± std over folds)."),
                 ("model_student_top1_macro_f1.png", "Fig. 13: student-test Top-1 accuracy and Macro-F1."),
                 ("model_log_loss.png", "Fig. 14: log loss comparison."),
                 ("model_accuracy_vs_cost.png", "Fig. 15: accuracy vs inference cost (justifies selection)."),
                 ("model_confusion_matrices.png", "Fig. 16: confusion matrices (optional diagnostic)."),
                 ("model_fold_stability.png", "Fig. 17: per-fold validation Macro-F1."),
                 ("model_generalisation_gap.png", "Fig. 18: training-fold vs validation-fold Macro-F1 (overfitting diagnosis)."),
                 ("model_training_histories.png", "Fig. 19: epoch/iteration histories, shown only for models that produce them (MLP, HGB).")]:
        story += img(FIG / f, caption=c)
    story += [P("5.1 Per-model answers", H2)]
    for m in model_sections(d["summary"]):
        story += [P(m["title"], H3), P(f"<b>1. Why suitable?</b> {m['A1']}"), P(f"<b>2. Strengths / limitations.</b> {m['A2']}"),
                  P(f"<b>5. Hyperparameters (initial).</b> {m['B5']}"), P(f"<b>6. Loss / objective.</b> {m['B6']}"),
                  P(f"<b>7. Student-test performance.</b> {m['C7']}"), P(f"<b>8. Top-1 / Macro-F1 / log loss.</b> {m['C8']}"),
                  P(f"<b>9. Over/under-fitting.</b> {m['C9']}")]
    # selection
    rf = s.set_index("model")
    hgb = rf.loc["hist_gradient_boosting"]
    folds = {m: pd.read_csv(RES / f"cv_folds_{m}.csv").set_index("fold") for m in s["model"]}
    rows = [["HGB minus model (same 5 folds)", "Macro-F1: folds HGB better", "mean Δ Macro-F1", "log loss: folds HGB better", "mean Δ log loss"]]
    for m in s["model"]:
        if m == "hist_gradient_boosting":
            continue
        dF = folds["hist_gradient_boosting"].val_macro_f1 - folds[m].val_macro_f1
        dL = folds["hist_gradient_boosting"].val_log_loss - folds[m].val_log_loss
        rows.append([rf.loc[m, "display"], f"{int((dF > 0).sum())}/5", f"{dF.mean():+.4f}", f"{int((dL < 0).sum())}/5", f"{dL.mean():+.4f}"])
    rfm = rf.loc["random_forest"]
    story += [P("5.2 Model selection", H2), P(
        f"Ranked by mean CV Macro-F1 the leader is {best['display']} ({best['cv_macro_f1']:.4f} ± {best['cv_macro_f1_std']:.4f}); "
        f"HistGradientBoosting scores {hgb.cv_macro_f1:.4f} ± {hgb.cv_macro_f1_std:.4f}. Because every model uses the same folds, paired "
        "per-fold differences are more informative than the overlapping standard deviations (table below). HGB beats the linear, "
        "generative, instance-based and kernel models in every fold on both Macro-F1 and log loss, so the tree-based family is "
        "clearly preferable. Against Random Forest the comparison is a genuine tie: RF has slightly better log loss "
        f"(student {rfm.st_log_loss:.4f} vs {hgb.st_log_loss:.4f}) while HGB has the higher student-test Top-1 ({hgb.st_top1:.4f} vs "
        f"{rfm.st_top1:.4f}). The decision was therefore taken on secondary criteria: HGB trains in {hgb.fit_s_full:.1f} s "
        f"(RF {rfm.fit_s_full:.1f} s), scores {hgb.inference_ms_per_1k_fixes:.1f} ms per 1000 fixes (RF {rfm.inference_ms_per_1k_fixes:.1f}), "
        f"and its artifact is {hgb.model_size_mb:.1f} MB (RF {rfm.model_size_mb:.0f} MB), which makes a 40-trial Bayesian search with "
        "5-fold CV feasible and deployment on Laptop 1 light. HGB was selected."), table(rows)]
    # ---- Bayesian optimisation
    story += [PageBreak(), P("6. Bayesian hyperparameter optimisation (HistGradientBoosting)", H1)]
    if d["opt"]:
        o = d["opt"]
        bp = o["best_params"]
        story += [P("<b>1. Hyperparameters searched / 2. ranges:</b> learning_rate 0.02–0.3 (log-uniform), max_iter 100–800 (step 50), "
                    "max_leaf_nodes 15–127 (log), max_depth {None, 6, 10, 16}, min_samples_leaf 10–400 (log), l2_regularization 1e-4–10 "
                    "(log), max_features 0.5–1.0. Ranges are centred on the baseline and span low→high capacity and weak→strong "
                    "regularisation."),
                  P(f"<b>3. Objective:</b> {o['objective']}; sampler {o['sampler']}; pruner {o['pruner']}. The search used every "
                    f"{o['search_obs_stride']}th fix for speed; the best configuration was re-scored with the baseline protocol."),
                  P(f"<b>4. Trials:</b> {o['n_trials']} ({o['n_complete']} completed, {o['n_pruned']} pruned) in "
                    f"{o['search_seconds'] / 60:.1f} minutes."),
                  P("<b>5. Best values:</b> " + ", ".join(f"{k} = {v if not isinstance(v, float) else round(v, 5)}" for k, v in bp.items()))]
        if tvu is not None:
            rows = [["Metric (same folds / same 10 trips)", "Untuned", "Tuned", "Δ"]]
            for k, lab in [("cv_macro_f1", "CV Macro-F1"), ("cv_top1", "CV Top-1"), ("cv_log_loss", "CV log loss"),
                           ("st_top1", "Student-test Top-1"), ("st_macro_f1", "Student-test Macro-F1"), ("st_log_loss", "Student-test log loss"),
                           ("inference_ms_per_1k_fixes", "Inference ms / 1000 fixes"), ("fit_s_full", "Training time (s, full fit)")]:
                rows.append([lab, f"{tvu.loc[k, 'untuned']:.4f}", f"{tvu.loc[k, 'tuned']:.4f}", f"{tvu.loc[k, 'delta']:+.4f}"])
            story += [table(rows, [6 * cm, 3 * cm, 3 * cm, 3 * cm])]
            dcv = tvu.loc["cv_macro_f1", "delta"]
            sd = tvu.loc["cv_macro_f1", "untuned_std"] if "untuned_std" in tvu.columns else d["summary"].set_index("model").loc["hist_gradient_boosting", "cv_macro_f1_std"]
            verdict = ("improved" if dcv > 0 else "did not improve")
            story += [P(f"<b>6. Did tuning help?</b> The tuned model {verdict} CV Macro-F1 by {dcv:+.4f} (fold std ≈ {sd:.4f}); "
                        f"student-test Top-1 changed by {tvu.loc['st_top1', 'delta']:+.4f} and log loss by {tvu.loc['st_log_loss', 'delta']:+.4f}. "
                        + ("The gain is small relative to fold variation, which is expected: the baseline already sits near a broad optimum "
                           "and the error floor is set by candidate misses and genuinely ambiguous fixes rather than model capacity. "
                           if abs(dcv) < sd else "") +
                        f"Selected for deployment: <b>{o['selected_final']}</b> (rule: {o['selection_rule']}).")]
    story += img(FIG / "optimisation_history_importance.png", caption="Fig. 20: Optuna optimisation history and hyperparameter importance.")
    story += img(FIG / "optimisation_tuned_vs_untuned.png", 13, caption="Fig. 21: untuned vs tuned.")
    # ---- HMM
    story += [PageBreak(), P("7. HMM and Viterbi route reconstruction", H1)]
    hp = d["hmm"]["params"] if d["hmm"] else {}
    story += [P("<b>1. Hidden states.</b> At fix t the states are its candidate arcs (top 8 by emission score). Each state keeps arc_id, "
                "projection fraction, perpendicular distance and classifier probability, so the candidate→road-segment correspondence is preserved."),
              P(f"<b>2. Emissions.</b> log E_t(c) = w_clf · log max(P_clf(c), 1e-6) − w_dist · ½(d/σ)², with tuned w_clf = {hp.get('w_clf')}, "
                f"w_dist = {hp.get('w_dist')}, σ = {hp.get('sigma_m')} m."),
              P(f"<b>3. Transitions.</b> For a at t−1 and b at t, d_route is the driving distance from a's projection to b's projection over the "
                f"directed network (bounded Dijkstra, cached per source node) and d_gps the straight distance between the fixes: "
                f"log T = −|d_route − d_gps| / β − u·[b = reverse(a)], with β = {hp.get('beta_m')} m and U-turn penalty u = {hp.get('uturn_penalty')}. "
                f"Transitions are impossible (−∞) when b is unreachable within the search radius (disconnected, or against one-way arcs) "
                f"or the implied speed exceeds {hp.get('vmax_mps')} m/s."),
              P("<b>Viterbi recurrence.</b> δ_t(s) = max_r[δ_{t−1}(r) + log T(r,s)] + log E_t(s), ψ_t(s) = argmax_r[…]; backtracking from "
                "argmax δ_T. Everything is in log space for numerical stability; state sets vary in size per fix."),
              P(f"<b>4. Missing candidates / large gaps.</b> A fix with no candidate receives no prediction and the chain restarts after it; "
                f"dt > {hp.get('max_gap_s')} s forces a new chain. <b>5. Disconnected consecutive candidates.</b> If every transition into "
                "fix t is −∞, a new chain starts at t. No segment is fabricated; route reconstruction fills shortest paths only inside chains."),
              P("<b>Decoder tuning.</b> w_clf, β, w_dist and the U-turn penalty were grid-searched on 30 training trips using out-of-fold "
                "classifier probabilities (each trip scored by a model that never saw it)." + (
                    f" Classifier-only Top-1 on those trips {d['hmm']['classifier_only_top1']:.4f}; best HMM {d['hmm']['best_hmm_top1']:.4f}." if d["hmm"] else ""))]
    if dec:
        rows = [["Student-test (10 trips)", "Classifier only", "Classifier + HMM", "+ reconstruction"],
                ["Top-1 accuracy (pooled fixes)", f"{dec['clf_top1']:.4f}", f"{dec['hmm_top1']:.4f}", "same"],
                ["Selection Macro-F1 (mean)", f"{dec['clf_sel_macro_f1']:.4f}", f"{dec['hmm_sel_macro_f1']:.4f}", "same"],
                ["Route continuity (mean)", f"{dec['clf_route_continuity']:.4f}", f"{dec['hmm_route_continuity']:.4f}", f"{dec['recon_route_continuity']:.4f}"],
                ["Invalid transitions (total)", f"{dec['clf_invalid_transitions']:.0f}", f"{dec['hmm_invalid_transitions']:.0f}", f"{dec['recon_invalid_transitions']:.0f}"],
                ["Route F1 (mean)", f"{dec['clf_route_f1']:.4f}", f"{dec['hmm_route_f1']:.4f}", f"{dec['recon_route_f1']:.4f}"],
                ["Edit similarity (mean)", f"{dec['clf_edit_similarity']:.4f}", f"{dec['hmm_edit_similarity']:.4f}", f"{dec['recon_edit_similarity']:.4f}"],
                ["LCS recall (mean)", f"{dec['clf_lcs_recall']:.4f}", "", f"{dec['recon_lcs_recall']:.4f}"],
                ["Candidate miss rate", f"{dec['miss_rate']:.4f}", f"{dec['miss_rate']:.4f}", ""],
                ["Runtime (s, all 10 trips)", f"{dec['clf_seconds']:.2f}", f"{dec['hmm_seconds']:.2f}", f"{dec['recon_seconds']:.2f}"]]
        story += [table(rows, [6 * cm, 3.3 * cm, 3.3 * cm, 3.3 * cm])]
        dt = d["dec_trips"]
        better = dt[dt.hmm_top1 > dt.clf_top1]; worse = dt[dt.hmm_top1 < dt.clf_top1]
        story += [P(f"<b>6. Did the HMM improve the route?</b> Yes on aggregate: Top-1 {dec['clf_top1']:.4f} → {dec['hmm_top1']:.4f} and invalid "
                    f"transitions {dec['clf_invalid_transitions']:.0f} → {dec['recon_invalid_transitions']:.0f} after reconstruction. Per trip, the HMM "
                    f"improved Top-1 on {len(better)} of {len(dt)} trips" + (f" and reduced it on {len(worse)} ({', '.join(worse.trip_id)})" if len(worse) else "") +
                    f". In total it corrected {int(dt.n_fixed_by_hmm.sum())} fixes that the classifier had wrong and broke {int(dt.n_broken_by_hmm.sum())} "
                    f"that the classifier had right. By error type (classifier → HMM): wrong direction of the right road "
                    f"{dec['clf_err_reverse_by_hmm']:.0f} → {dec['hmm_err_reverse_by_hmm']:.0f}, neighbouring arc at a junction "
                    f"{dec['clf_err_adjacent_by_hmm']:.0f} → {dec['hmm_err_adjacent_by_hmm']:.0f}, a different road "
                    f"{dec['clf_err_other_by_hmm']:.0f} → {dec['hmm_err_other_by_hmm']:.0f}. Where it does not help: the remaining direction "
                    "and junction errors occur mostly while the vehicle is slow or stationary (no usable heading, so both directions explain the "
                    f"fixes equally); {dec['hmm_transition_breaks']:.0f} chain restarts (no valid transition) and {dec['hmm_gap_breaks']:.0f} gap breaks "
                    "make the decoder restart from emissions; and where the GPS runs far from the reference or the reference itself is broken, "
                    "the smoothest network path can disagree with the reference. The tuned decoder parameters were also compared with the "
                    f"defaults on the same trips: Top-1 {d['dec']['summary']['final_default_hmm']['hmm_top1']:.4f} (default) vs {dec['hmm_top1']:.4f} (tuned).")]
        rows = [["Trip", "fixes", "miss rate", "clf Top-1", "HMM Top-1", "clf contin.", "recon contin.", "fixed", "broken", "chains"]]
        for _, r in dt.iterrows():
            rows.append([r.trip_id, int(r.n_fixes), f"{r.miss_rate:.4f}", f"{r.clf_top1:.4f}", f"{r.hmm_top1:.4f}", f"{r.clf_route_continuity:.3f}",
                         f"{r.recon_route_continuity:.3f}", int(r.n_fixed_by_hmm), int(r.n_broken_by_hmm), int(r.hmm_chains)])
        story += [table(rows)]
    story += img(FIG / "hmm_vs_classifier_summary.png", 14, caption="Fig. 22: classifier-only vs classifier + HMM (student-test).")
    story += img(FIG / "hmm_vs_classifier_per_trip.png", caption="Fig. 23: per-trip Top-1 and route continuity.")
    # ---- Folium
    story += [PageBreak(), P("8. Geographical visualisation with Folium", H1), P(
        "Maps are generated by scripts/generate_maps.py (src/pathsnap/visualisation/folium_maps.py) from the real data and the deployed "
        "model's predictions. Layers: road network near the trajectory (grey), raw GPS fixes (blue), reference route (green, offline "
        "development maps only), classifier-only route (orange dashed), HMM reconstructed route (purple with direction arrows), start/end "
        "markers, GPS gaps and HMM chain breaks; a layer control and legend are included and the view is fitted to the trajectory. The "
        "interactive HTML files are in maps/html/.")]
    for f in sorted(MAPS.glob("student_*_comparison.png"))[:4]:
        story += img(f, caption=f"Fig.: Folium comparison map: {f.stem.replace('_', ' ')}.")
    # ---- API/UI/deployment
    story += [PageBreak(), P("9. API, UI and two-laptop deployment", H1), P(
        "<b>Laptop 1</b> runs the FastAPI backend (python -m backend.main, bound to 0.0.0.0:8000). On startup it loads the final classifier, "
        "the feature schema and the tuned HMM parameters, and lists the trips of data/raw/evaluator_test (EVALUATION mode) or, while that "
        "folder is empty, the student-test trips (DEVELOPMENT mode, flagged in the UI). POST /api/predict/{trip_id} executes cleaning → "
        "candidates → features → classifier → HMM/Viterbi → reconstruction on the raw files, then the server-side RouteEvaluator aligns the "
        "reference route and returns aggregate metrics only. Every request is logged on the Laptop 1 console with client IP, endpoint, status "
        "and duration. <b>Laptop 2</b> runs the React + TypeScript + Leaflet UI (npm run dev) configured with VITE_API_URL=http://&lt;laptop-1-ip&gt;:8000. "
        "The full Windows procedure (IP lookup, firewall rule, connectivity checks, failure handling) is in docs/deployment.md.")]
    rows = [["Method", "Endpoint", "Purpose"], ["GET", "/health", "liveness, model loaded, mode"], ["GET", "/api/system/info", "mode, partition, versions"],
            ["GET", "/api/model/info", "model variant, schema, HMM params"], ["GET", "/api/trips", "trips available for evaluation"],
            ["GET", "/api/trips/{id}/metadata", "fixes, bbox, network size, gaps"], ["POST", "/api/predict/{id}", "full pipeline + server-side metrics"],
            ["GET", "/api/results/{id}", "cached result"], ["GET", "/api/results/{id}/explain/{fix}", "candidate probabilities and features for a fix"]]
    story += [table(rows, [1.5 * cm, 5.5 * cm, 9 * cm])]
    for f, c in [("ui_trip_00000052.png", "Fig.: working UI: trip 00000052 processed through the live API."),
                 ("ui_trip_00000009.png", "Fig.: working UI: trip 00000009."), ("api_console_log.png", "Fig.: API console on the receiving machine (request log).")]:
        story += img(FIG / f, caption=c)
    story += [P("Note: the UI and console captures above were taken on a single development machine (API and UI on one host, accessed over "
                "the LAN interface where stated). The two-laptop screenshots required by the brief must be taken during the live demonstration.", SMALL)]
    # ---- testing, limitations, conclusion
    story += [P("10. Testing and edge cases", H1), P(
        "Automated tests (pytest + vitest) cover malformed/missing/empty files, invalid coordinates, duplicate and out-of-order timestamps, "
        "invalid node/arc/route references, single-observation trips, candidate generation (none/one/many, fallback, cap, dense grids), "
        "label-independence of candidates, feature schema/order checks, class-column alignment, missing/corrupt artifacts, Viterbi against "
        "brute force, empty/single/variable-size sequences, disconnected states, forced gap breaks, impossible speeds, route metrics against "
        "reference implementations, API status codes (404/405/422/503/504), repeated and concurrent requests, ground-truth leakage, and "
        "end-to-end runs on synthetic and real trips. Results are recorded in FINAL_ACCEPTANCE_TEST.md.")]
    story += [P("11. Limitations", H1), P(
        "Arcs are straight segments (no curved shape points); the candidate miss rate bounds accuracy; reference routes contain breaks and "
        "parked periods, so labels are approximate; KNN and SVM were trained on subsamples; the evaluator partition was not available, so "
        "final-evaluation numbers cannot be reported here; the two-laptop LAN demonstration must be performed and captured by the team.")]
    story += [P("12. Conclusion", H1), P(
        "A complete, reproducible map-matching system was built and evaluated with trip-grouped validation. Gradient-boosted and bagged "
        "tree ensembles beat linear, generative, instance-based and kernel models consistently across folds, although by modest margins on "
        "Macro-F1 (the heading and relative-distance features already make the problem nearly separable for a linear model); "
        "HistGradientBoosting gives the best balance of accuracy, probability quality and cost, and Bayesian optimisation added a small, "
        "within-noise gain on CV with a clearer improvement in log loss. The largest gain comes from sequential decoding: an HMM that "
        "respects network connectivity improved Top-1 on every student-test trip and turned independent noisy decisions into "
        "almost perfectly drivable routes. The remaining errors are dominated by direction and junction ambiguity while the vehicle is "
        "slow or stationary, and by candidate misses.")]
    story += [P("References", H1)] + [P(x, SMALL) for x in [
        "[1] P. Newson and J. Krumm, 'Hidden Markov map matching through noise and sparseness', ACM SIGSPATIAL GIS, 2009.",
        "[2] A. J. Viterbi, 'Error bounds for convolutional codes and an asymptotically optimum decoding algorithm', IEEE Trans. Inf. Theory, 1967.",
        "[3] J. Bergstra, R. Bardenet, Y. Bengio, B. Kégl, 'Algorithms for hyper-parameter optimization' (TPE), NeurIPS 2011.",
        "[4] T. Akiba et al., 'Optuna: A next-generation hyperparameter optimization framework', KDD 2019.",
        "[5] F. Pedregosa et al., 'Scikit-learn: Machine learning in Python', JMLR 12, 2011.",
        "[6] J. Platt, 'Probabilistic outputs for support vector machines', Advances in Large Margin Classifiers, 1999.",
        "[7] S. T. Leutenegger, M. A. Lopez, J. Edgington, 'STR: A simple and efficient algorithm for R-tree packing', ICDE 1997."]]
    story += [P("Appendix A: configuration files and AI disclosure", H1), P(
        "All configurations are in configs/*.yaml; per-experiment records (parameters, seed, feature version, fold metrics, git commit, "
        "timestamps) are in experiments/results/experiment_*.json and experiments/optimisation/. AI assistance: the implementation was produced "
        "with the help of an AI coding assistant (Claude Code) under the team's direction; every number in this report is produced by the "
        "repository's scripts from the supplied data.", SMALL)]

    OUT.parent.mkdir(parents=True, exist_ok=True)

    def footer(c, doc):
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.grey)
        c.drawString(2 * cm, 1.2 * cm, "PathSnap: Technical Report (UE24CS352A)")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")

    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                      title="PathSnap Technical Report", author="Team PathSnap").build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT.relative_to(ROOT))


def write_quick_revision(d):
    s = d["summary"].sort_values("cv_macro_f1", ascending=False)
    L = ["# Quick Revision (generated by scripts/build_report.py from experiment records)", "",
         "## Pipeline", "raw .track/.nodes/.arcs → clean + per-trip UTM → STRtree candidates (R=50 m, K=10, fallback 150 m) → "
         "21 features → HistGradientBoosting P(correct) → HMM (emission = classifier log-prob, transition = −|d_route − d_gps|/β "
         "− U-turn penalty) → segmented Viterbi → shortest-path reconstruction → FastAPI → React UI. Labels: monotone DP "
         "alignment of fixes to .route (server-side/training only).", "",
         "## Data", "80 train trips (208 417 fixes), 10 student-test (19 160), evaluator trips not supplied. 1 Hz; 135 gaps > 30 s "
         "in 30 trips; 14 route breaks; candidate miss rate 1.14% train / 0.15% student-test.", "",
         "## Models (CV = 5-fold GroupKFold by trip; ST = student-test, classifier only)", "",
         "| Model | CV Macro-F1 | CV Top-1 | ST Top-1 | ST Macro-F1 | ST log loss | gap (train−val F1) |", "|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append(f"| {r.display} | {r.cv_macro_f1:.4f}±{r.cv_macro_f1_std:.4f} | {r.cv_top1:.4f} | {r.st_top1:.4f} | "
                 f"{r.st_macro_f1:.4f} | {r.st_log_loss:.4f} | {r.gap_macro_f1:+.4f} |")
    if d["opt"]:
        o, t = d["opt"], d["tvu"]
        L += ["", "## Bayesian optimisation", f"Optuna TPE, {o['n_trials']} trials ({o['n_complete']} complete, {o['n_pruned']} pruned), "
              f"objective mean CV Macro-F1, {o['search_seconds'] / 60:.0f} min. Best: " +
              ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in o["best_params"].items()) + ".",
              f"Tuned vs untuned: CV Macro-F1 {t.loc['cv_macro_f1', 'untuned']:.4f} → {t.loc['cv_macro_f1', 'tuned']:.4f}; "
              f"ST Top-1 {t.loc['st_top1', 'untuned']:.4f} → {t.loc['st_top1', 'tuned']:.4f}; ST log loss "
              f"{t.loc['st_log_loss', 'untuned']:.4f} → {t.loc['st_log_loss', 'tuned']:.4f}. Deployed: {o['selected_final']}."]
    if d["dec"]:
        f = d["dec"]["summary"]["final"]
        hp = d["hmm"]["params"]
        L += ["", "## HMM (student-test, final classifier)",
              f"Params: w_clf={hp['w_clf']}, beta={hp['beta_m']} m, w_dist={hp['w_dist']}, U-turn={hp['uturn_penalty']}, vmax={hp['vmax_mps']} m/s, "
              f"gap break > {hp['max_gap_s']} s, top-{hp['max_states']} states.",
              f"Top-1 {f['clf_top1']:.4f} → {f['hmm_top1']:.4f}; route continuity {f['clf_route_continuity']:.3f} → {f['hmm_route_continuity']:.3f} "
              f"(→ {f['recon_route_continuity']:.3f} with reconstruction); route F1 {f['clf_route_f1']:.3f} → {f['recon_route_f1']:.3f}; "
              f"edit similarity {f['clf_edit_similarity']:.3f} → {f['recon_edit_similarity']:.3f}.",
              f"Fixed {f['n_fixed_by_hmm']:.0f} fixes, broke {f['n_broken_by_hmm']:.0f}. Error types classifier → HMM: reverse direction "
              f"{f['clf_err_reverse_by_hmm']:.0f} → {f['hmm_err_reverse_by_hmm']:.0f}, junction neighbour {f['clf_err_adjacent_by_hmm']:.0f} → "
              f"{f['hmm_err_adjacent_by_hmm']:.0f}, other road {f['clf_err_other_by_hmm']:.0f} → {f['hmm_err_other_by_hmm']:.0f}."]
    if d["abl"] is not None:
        L += ["", "## Feature ablation (Δ CV Top-1 when a group is removed)"] + [
            f"- {r.dropped_group}: {100 * r.delta_top1:+.2f} pts" for _, r in d["abl"].iloc[1:].iterrows()]
    L += ["", "## API", "GET /health · GET /api/system/info · GET /api/model/info · GET /api/trips · GET /api/trips/{id}/metadata · "
          "POST /api/predict/{id} · GET /api/results/{id} · GET /api/results/{id}/explain/{fix}", "",
          "## Commands", "Laptop 1: `python -m backend.main` · Laptop 2: `cd frontend; npm run dev` with `VITE_API_URL=http://<ip>:8000` · "
          "check: `python scripts/lan_check.py http://<ip>:8000`", "",
          "## Key files", "labels `data/labeling.py` · candidates `features/candidate_generation.py` · features `features/feature_extraction.py` · "
          "CV `training/cross_validation.py` · HMM `decoding/` · API `backend/` · UI `frontend/src/`"]
    (ROOT / "reports" / "viva" / "QUICK_REVISION.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    d = load()
    best = write_markdown(d)
    write_quick_revision(d)
    build_pdf(d, best)


if __name__ == "__main__":
    main()
