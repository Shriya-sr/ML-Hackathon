from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "output" / "pdf" / "PathSnap_Technical_Report.pdf"


def ptext(value):
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def chart_models(comparison: pd.DataFrame, metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=160)
    cv = comparison.drop_duplicates("model").sort_values("cv_macro_f1")
    ax.barh(cv["model"], cv["cv_macro_f1"], color="#2d718e")
    ax.set_xlabel("Trip-grouped CV Macro F1")
    ax.set_title("Model comparison on the training partition")
    ax.grid(axis="x", alpha=.2); fig.tight_layout()
    cv_path = RESULTS / "cv_macro_f1.png"; fig.savefig(cv_path); plt.close(fig)

    summary = metrics.groupby(["model", "method"], as_index=False)[["top1_accuracy", "macro_f1"]].mean()
    pivot = summary.pivot(index="model", columns="method", values="top1_accuracy").fillna(0)
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=160)
    pivot.plot(kind="bar", ax=ax, color=["#1769aa", "#e53935"])
    ax.set_ylabel("Student-test Top-1 Accuracy"); ax.set_xlabel("")
    ax.set_title("Classifier and classifier + HMM")
    ax.legend(["Classifier", "Classifier + Viterbi"]); ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    compare_path = RESULTS / "student_test_accuracy.png"; fig.savefig(compare_path); plt.close(fig)
    return cv_path, compare_path


def page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(colors.HexColor("#d7e0e4")); canvas.line(17*mm, h-15*mm, w-17*mm, h-15*mm)
    canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#5b6b73"))
    canvas.drawString(17*mm, h-11*mm, "PATHSNAP | TECHNICAL REPORT")
    canvas.drawRightString(w-17*mm, 10*mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    inv = pd.read_csv(RESULTS / "dataset_inventory.csv")
    comparison = pd.read_csv(RESULTS / "model_comparison.csv")
    metrics = pd.read_csv(RESULTS / "student_test_metrics.csv")
    trials = json.loads((RESULTS / "bayesian_trials.json").read_text())
    summary = json.loads((RESULTS / "dataset_summary.json").read_text())
    cv_png, accuracy_png = chart_models(comparison, metrics)
    map_files = sorted((RESULTS / "maps").glob("trip_*.png"))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBlue", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25,
        leading=30, textColor=colors.HexColor("#124b5f"), alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle(name="Sub", parent=styles["Normal"], fontSize=11, leading=16,
        textColor=colors.HexColor("#54636a"), spaceAfter=18))
    styles.add(ParagraphStyle(name="H1b", parent=styles["Heading1"], fontSize=17, leading=22,
        textColor=colors.HexColor("#124b5f"), spaceBefore=10, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2b", parent=styles["Heading2"], fontSize=12, leading=16,
        textColor=colors.HexColor("#21738b"), spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="Bodyb", parent=styles["BodyText"], fontSize=9.3, leading=13.3, spaceAfter=6))
    styles.add(ParagraphStyle(name="Smallb", parent=styles["BodyText"], fontSize=7.2, leading=9))
    story = [Spacer(1, 22*mm), Paragraph("PathSnap", styles["TitleBlue"]),
             Paragraph("Distributed map matching and route reconstruction", styles["Sub"]),
             Paragraph("Technical report | PES University Machine Learning Hackathon", styles["Bodyb"]),
             Spacer(1, 8*mm), Paragraph("Executive summary", styles["H1b"]),
             Paragraph("PathSnap matches noisy GPS observations to road segments and applies a Viterbi decoder to encourage a continuous route. This report documents the implementation and measured development results generated from the organizer-provided training and student-test partitions. Evaluator-test data was not present in the local package and was not used.", styles["Bodyb"]),
             Paragraph("Dataset and exploration", styles["H1b"])]

    part_rows = [["Partition", "Trips", "GPS observations", "Road nodes", "Road arcs"]]
    for part, values in summary.items():
        part_rows.append([part, str(values["trips"]), f"{values['observations']:,}", f"{values['nodes']:,}", f"{values['arcs']:,}"])
    tbl = Table(part_rows, colWidths=[38*mm, 24*mm, 37*mm, 34*mm, 34*mm], repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#124b5f")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#b8c7ce")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eff5f6")]),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story += [tbl, Spacer(1, 5*mm), Paragraph(f"The inventory contains {len(inv)} trips. Tracks were read as longitude, latitude, and timestamp. GPS rows with non-finite values were removed by the loader. Candidate geometry uses a local equirectangular projection centered at the trip's median latitude. Road candidates are selected from the nearest eight road-segment midpoints. The reference route is aligned monotonically to observations with a dynamic program; one GPS row is not assumed to equal one route row.", styles["Bodyb"]),
              Paragraph("Model experimentation", styles["H1b"]),
              Paragraph("Each candidate road segment is represented by perpendicular distance, heading difference, segment length, GPS-derived speed, and time gap. The binary classifier scores whether a candidate matches the aligned reference arc. Up to 300 evenly spaced observations per training trip were used to bound runtime. Cross-validation splits by trip, preventing observations from one trip from appearing in both training and validation folds.", styles["Bodyb"]),
              Image(str(cv_png), width=168*mm, height=76*mm)]

    story += [PageBreak(), Paragraph("Results", styles["H1b"]),
              Paragraph("The following summary is computed from the actual student-test trip predictions. Macro F1 treats road-arc IDs as classes; candidate miss rate measures whether the aligned reference arc appeared among the eight nearest midpoint candidates. Candidate log loss is binary candidate-level loss for the classifier output. Decoder log loss is not reported because Viterbi returns a best sequence rather than calibrated per-observation probabilities.", styles["Bodyb"])]
    headers = ["Model", "Method", "CV Macro F1", "Top-1", "Macro F1", "Continuity", "Miss rate", "Log loss"]
    rows = [headers]
    for _, r in comparison.iterrows():
        rows.append([str(r.model).replace("_", " "), str(r.method).replace("_", " "), ptext(r.cv_macro_f1),
                     ptext(r.top1_accuracy), ptext(r.macro_f1), ptext(r.route_continuity),
                     ptext(r.candidate_miss_rate), ptext(r.candidate_log_loss)])
    tbl = Table(rows, colWidths=[29*mm,25*mm,22*mm,15*mm,18*mm,21*mm,17*mm,17*mm], repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#124b5f")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),6.5),("GRID",(0,0),(-1,-1),.25,colors.HexColor("#b8c7ce")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#eff5f6")]),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story += [tbl, Spacer(1,3*mm), Image(str(accuracy_png), width=168*mm, height=84*mm),
              Paragraph("Five models were trained: Logistic Regression, Gaussian Naive Bayes, Random Forest, Extra Trees, and HistGradientBoosting. The tuned Random Forest row uses the best observed configuration from the Bayesian search. Model choice should be based on the table and the team's validation of route continuity and inference cost, rather than training score alone.", styles["Bodyb"]),
              Paragraph("Per-model discussion", styles["H1b"])]
    descriptions = {
        "logistic_regression": ("A linear probabilistic baseline for candidate features.", "max_iter=300; class_weight=balanced.", "Fast and interpretable; its linear decision boundary may miss interactions between GPS geometry and road shape."),
        "gaussian_naive_bayes": ("A compact probabilistic baseline for candidate scoring.", "StandardScaler followed by GaussianNB defaults.", "Trains quickly and returns probabilities; the conditional-independence and Gaussian assumptions are unlikely to hold exactly for correlated geometric features."),
        "random_forest": ("A nonlinear tree ensemble selected for Bayesian tuning.", "40 trees; max_depth=18; min_samples_leaf=2; balanced_subsample; seed=42.", "Captures feature interactions and is robust to nonlinear boundaries; it costs more memory and inference time than linear baselines and its probabilities may need calibration."),
        "extra_trees": ("A randomized tree ensemble used to compare an alternative bagging strategy.", "40 trees; max_depth=18; min_samples_leaf=2; balanced class weights; seed=42.", "Randomized splits provide a diverse nonlinear baseline; model size and probability calibration remain considerations."),
        "gradient_boosting": ("A boosting model that sequentially corrects classification errors.", "40 estimators; learning_rate=0.1; max_depth=2; seed=42.", "Can capture nonlinear feature interactions; boosting can be sensitive to settings and may overfit if iteration depth is not controlled."),
    }
    for name, (why, configuration, tradeoff) in descriptions.items():
        rows_for_model = comparison[comparison["model"] == name]
        direct = rows_for_model[rows_for_model["method"] == "classifier"]
        decoded = rows_for_model[rows_for_model["method"] == "classifier_hmm"]
        cv = ptext(rows_for_model.iloc[0]["cv_macro_f1"]) if not rows_for_model.empty else "N/A"
        direct_f1 = ptext(direct.iloc[0]["macro_f1"]) if not direct.empty else "N/A"
        decoded_f1 = ptext(decoded.iloc[0]["macro_f1"]) if not decoded.empty else "N/A"
        story += [Paragraph(name.replace("_", " ").title(), styles["H2b"]),
                  Paragraph(f"Suitability: {why} Configuration: {configuration} Trade-offs: {tradeoff} Measured trip-grouped CV Macro F1={cv}; student-test classifier Macro F1={direct_f1}; with Viterbi={decoded_f1}.", styles["Bodyb"])]
    story += [
              Paragraph("Bayesian optimization", styles["H1b"]),
              Paragraph("The Random Forest search maximized trip-grouped CV Macro F1. A Gaussian-process model with a Matérn kernel and expected improvement selected six trials over max_depth 8-30 and min_samples_leaf 1-5. Trial results and the selected configuration are recorded in results/bayesian_trials.json. The tuned model was refit on the sampled training observations and compared on the same student-test trips as the untuned models.", styles["Bodyb"])]
    best = max(trials, key=lambda t: t["cv_macro_f1"])
    story.append(Paragraph(f"Best observed trial: max_depth={best['max_depth']}, min_samples_leaf={best['min_samples_leaf']}, CV Macro F1={best['cv_macro_f1']:.3f}.", styles["Bodyb"]))

    story += [PageBreak(), Paragraph("Route reconstruction and system", styles["H1b"]),
              Paragraph("The HMM hidden state is a candidate directed road arc at each GPS observation. Classifier candidate probabilities provide emission scores. A transition receives no penalty when the same arc is retained or the previous arc's destination node equals the next arc's source node; disconnected transitions receive a fixed log penalty. Viterbi finds the highest-scoring candidate sequence. The student-test table reports classifier-only and classifier-plus-decoder accuracy, macro F1, continuity, and candidate miss rate. This decoder uses direct adjacency and does not insert intermediate road arcs.", styles["Bodyb"]),
              Paragraph("Geographical visualization", styles["H1b"])]
    if map_files:
        story += [Paragraph("The map below overlays original GPS observations, the supplied reference route, and the decoded route for a representative student-test trip. The matching interactive Folium HTML is generated by backend.export_maps and stored in results/maps when run locally.", styles["Bodyb"]),
                  Image(str(map_files[0]), width=160*mm, height=124*mm)]
    else:
        story.append(Paragraph("Run backend.export_maps to generate the representative route visualization before finalizing this report.", styles["Bodyb"]))
    story += [Paragraph("API, UI, and deployment", styles["H1b"]),
              Paragraph("A FastAPI service loads the trained artifact, predicts candidate probabilities, decodes the route, and returns route geometry and development metrics. The browser UI requests the available student-test trip IDs and displays the predicted route. Reference labels remain on the server and are never included in the prediction response. For the event's two-laptop setup, host the API on laptop 1 using uvicorn and open the laptop 1 LAN address on laptop 2.", styles["Bodyb"]),
              Paragraph("Limitations and next steps", styles["H1b"]),
              Paragraph("The local data package contains 80 training and 10 student-test trips but no evaluator-test partition. Final evaluation against the reserved trips, deployment screenshots, and validation of all ten evaluator requests remain pending organizer access. The report's numeric results are development results only. The team should review the route alignment and candidate miss cases, reproduce the run, and be prepared to explain all feature, model, optimization, and decoder choices.", styles["Bodyb"])]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=17*mm, leftMargin=17*mm,
        topMargin=22*mm, bottomMargin=18*mm, title="PathSnap Technical Report", author="PathSnap Team")
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(OUT)


if __name__ == "__main__":
    build()
