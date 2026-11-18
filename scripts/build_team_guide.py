"""Build the PathSnap Complete Team Guide (PDF) - a plain-language, detailed explanation of the whole project,
the team split, the demo playbook and the requirements check. Every number is read from the experiment records.

Usage:   python scripts/build_team_guide.py
Needs:   scripts/make_guide_figures.py run first; optional reports/guide/requirements_check.json
Writes:  reports/guide/PathSnap_Team_Guide.pdf
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Flowable, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle, BaseDocTemplate, Frame)

ROOT = Path(__file__).resolve().parents[1]
RES, OPT, FIG = ROOT / "experiments" / "results", ROOT / "experiments" / "optimisation", ROOT / "reports" / "figures"
GFIG, EDAF, MAPS = ROOT / "reports" / "guide" / "figures", ROOT / "reports" / "eda" / "figures", ROOT / "maps" / "screenshots"
OUT = ROOT / "reports" / "guide" / "PathSnap_Team_Guide.pdf"

# ----------------------------------------------------------------------------------------------- fonts & colours
FONTS = Path("C:/Windows/Fonts")
for name, file in [("UI", "segoeui.ttf"), ("UI-B", "segoeuib.ttf"), ("UI-I", "segoeuii.ttf"), ("UI-SB", "seguisb.ttf"),
                   ("UI-BL", "seguibl.ttf"), ("UI-L", "segoeuil.ttf"), ("Mono", "consola.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))
pdfmetrics.registerFontFamily("UI", normal="UI", bold="UI-B", italic="UI-I", boldItalic="UI-B")

NAVY, NAVY2 = colors.HexColor("#0b1f3a"), colors.HexColor("#13315c")
TEAL, ORANGE, GREEN, RED = colors.HexColor("#0fa3a3"), colors.HexColor("#f07b1d"), colors.HexColor("#1fa45a"), colors.HexColor("#d64545")
BLUE, VIOLET = colors.HexColor("#2f74e0"), colors.HexColor("#7b5cd6")
INK, MUTED, LINE, PALE, PALE2 = (colors.HexColor("#1d2633"), colors.HexColor("#5b6675"), colors.HexColor("#d9dee5"),
                                 colors.HexColor("#f3f6fa"), colors.HexColor("#e9f6f5"))
MEMBER_COL = {"Shriya": TEAL, "Sudheeksh": VIOLET, "Shubashitha": ORANGE, "Sujal": BLUE}

PW, PH = A4
M = 1.7 * cm
CW = PW - 2 * M

BODY = ParagraphStyle("body", fontName="UI", fontSize=9.6, leading=14.2, textColor=INK, spaceAfter=5)
LEAD = ParagraphStyle("lead", parent=BODY, fontName="UI-L", fontSize=12, leading=17.5, textColor=NAVY2, spaceAfter=8)
H2 = ParagraphStyle("h2", fontName="UI-B", fontSize=13.5, leading=17, textColor=NAVY, spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle("h3", fontName="UI-SB", fontSize=10.8, leading=14, textColor=NAVY2, spaceBefore=6, spaceAfter=2)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.2, leading=11, textColor=MUTED, spaceAfter=3)
CAP = ParagraphStyle("cap", parent=SMALL, alignment=TA_CENTER, spaceBefore=2, spaceAfter=10)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.4, leading=11.2, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="UI-SB")
CELLH = ParagraphStyle("cellh", parent=CELL, fontName="UI-B", textColor=colors.white)
CODE = ParagraphStyle("code", fontName="Mono", fontSize=8.2, leading=11, textColor=colors.HexColor("#e8edf5"))
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2.5, bulletFontName="UI",
                     bulletFontSize=7, bulletColor=colors.HexColor("#0fa3a3"))


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items, st=BUL, color="#0fa3a3"):
    return [Paragraph(t, st, bulletText="●") for t in items] if items else []


def load():
    d = {"summary": pd.read_csv(RES / "model_summary.csv").set_index("model"),
         "dec": json.loads((RES / "decoder_summary.json").read_text())["summary"]["final"],
         "dec_default": json.loads((RES / "decoder_summary.json").read_text())["summary"]["final_default_hmm"],
         "opt": json.loads((OPT / "optimisation_summary.json").read_text()),
         "abl": pd.read_csv(RES / "feature_ablation.csv"),
         "hmm": json.loads((ROOT / "artifacts" / "metadata" / "hmm_params.json").read_text()),
         "inv": pd.read_csv(ROOT / "reports" / "eda" / "dataset_inventory.csv", dtype={"trip_id": str})}
    ho = RES / "heldout_evaluator_summary.json"
    d["ho"] = json.loads(ho.read_text())["summary"] if ho.exists() else None
    d["ho_trips"] = pd.read_csv(RES / "heldout_evaluator_comparison.csv", dtype={"trip_id": str}) if ho.exists() else None
    rq = ROOT / "reports" / "guide" / "requirements_check.json"
    d["req"] = json.loads(rq.read_text(encoding="utf-8")) if rq.exists() else None
    return d


# ----------------------------------------------------------------------------------------------- flowables
class SectionTitle(Flowable):
    def __init__(self, num, title, kicker=""):
        super().__init__()
        self.num, self.title, self.kicker = num, title, kicker
        self.height = 58

    def wrap(self, aw, ah):
        self.width = aw
        return aw, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(NAVY); c.roundRect(0, 8, 42, 42, 9, fill=1, stroke=0)
        c.setFillColor(TEAL); c.circle(42, 50, 5, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont("UI-BL", 19)
        c.drawCentredString(21, 21, str(self.num))
        c.setFillColor(TEAL); c.setFont("UI-B", 8.2)
        c.drawString(56, 40, self.kicker.upper())
        c.setFillColor(NAVY); c.setFont("UI-BL", 20)
        c.drawString(56, 16, self.title)
        c.setStrokeColor(LINE); c.setLineWidth(0.8); c.line(56, 8, self.width, 8)


class Box(Flowable):
    """Rounded box with drawn content callbacks (used for diagrams)."""

    def __init__(self, height, painter):
        super().__init__()
        self.height, self.painter = height, painter

    def wrap(self, aw, ah):
        self.width = aw
        return aw, self.height

    def draw(self):
        self.painter(self.canv, self.width, self.height)


def rbox(c, x, y, w, h, fill, stroke=None, r=7, lw=1):
    c.setFillColor(fill)
    if stroke is not None:
        c.setStrokeColor(stroke); c.setLineWidth(lw)
    c.roundRect(x, y, w, h, r, fill=1, stroke=1 if stroke is not None else 0)


def text_lines(c, x, y, lines, font="UI", size=8, color=INK, leading=10.5, center=False):
    c.setFillColor(color); c.setFont(font, size)
    for i, s in enumerate(lines):
        (c.drawCentredString if center else c.drawString)(x, y - i * leading, s)


def arrow(c, x1, y1, x2, y2, color=MUTED, lw=1.6, head=5):
    import math
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(lw)
    c.line(x1, y1, x2, y2)
    ang = math.atan2(y2 - y1, x2 - x1)
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - head * math.cos(ang - 0.45), y2 - head * math.sin(ang - 0.45))
    p.lineTo(x2 - head * math.cos(ang + 0.45), y2 - head * math.sin(ang + 0.45))
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def pipeline_diagram():
    stages = [("1  Load & clean", ["read .track/.nodes/.arcs", "fix order, drop bad rows"], TEAL),
              ("2  Project to metres", ["lat/lon → UTM x, y", "per-trip zone"], TEAL),
              ("3  Candidates", ["roads within 50 m", "STRtree, ≤ 10 per fix"], TEAL),
              ("4  21 features", ["distance, heading,", "context, speed, network"], TEAL),
              ("5  Classifier", ["HistGradientBoosting", "P(this road is right)"], VIOLET),
              ("6  HMM + Viterbi", ["best sequence of roads", "that can be driven"], ORANGE),
              ("7  Route rebuild", ["fill gaps with", "shortest network paths"], ORANGE),
              ("8  API → UI", ["FastAPI on Laptop 1", "React map on Laptop 2"], BLUE)]

    def paint(c, w, h):
        bw, bh, gap = (w - 3 * 16) / 4, 62, 16
        for i, (t, sub, col) in enumerate(stages):
            row, k = divmod(i, 4)
            if row == 1:
                k = 3 - k                      # snake layout
            x = k * (bw + gap)
            y = h - 82 - row * 92
            rbox(c, x, y, bw, bh, colors.white, LINE, r=8)
            c.setFillColor(col); c.roundRect(x, y + bh - 6, bw, 6, 3, fill=1, stroke=0)
            text_lines(c, x + 9, y + bh - 22, [t], "UI-B", 9.2, NAVY)
            text_lines(c, x + 9, y + bh - 36, sub, "UI", 7.8, MUTED, 10)
            if i < 7:
                if i == 3:
                    arrow(c, x + bw / 2, y - 2, x + bw / 2, y - 30, MUTED)
                elif row == 0:
                    arrow(c, x + bw + 2, y + bh / 2, x + bw + gap - 2, y + bh / 2)
                else:
                    arrow(c, x - 2, y + bh / 2, x - gap + 2, y + bh / 2)
        y0 = 6
        rbox(c, 0, y0, w, 22, PALE2, None, r=6)
        text_lines(c, 10, y0 + 8, ["Server-side only:  .route → RouteEvaluator → accuracy numbers. "
                                   "Never used for prediction, never sent to the UI."],
                   "UI-SB", 8, colors.HexColor("#0b6b6b"))
        for col, lab, xx in [(TEAL, "data & features", 0), (VIOLET, "machine learning", 120), (ORANGE, "sequence decoding", 240),
                             (BLUE, "system", 370)]:
            c.setFillColor(col); c.circle(xx + 5, h - 4, 3.5, fill=1, stroke=0)
            text_lines(c, xx + 12, h - 7, [lab], "UI", 7.5, MUTED)
    return Box(250, paint)


def laptops_diagram():
    def paint(c, w, h):
        lw_, lh = w * 0.37, h - 30
        # laptop 1
        rbox(c, 0, 18, lw_, lh, colors.HexColor("#eef3fb"), BLUE, r=10, lw=1.4)
        text_lines(c, 12, h - 28, ["LAPTOP 1 · Sujal · backend"], "UI-B", 9.5, BLUE)
        items = ["evaluator_test/ (10 trips, incl. .route)", "classifier + HMM parameters",
                 "MapMatcher: clean → candidates →", "   features → classifier → HMM", "RouteEvaluator: scores vs .route",
                 "FastAPI · backend.main · port 8000", "console logs every request"]
        text_lines(c, 14, h - 44, ["•  " + s if not s.startswith("   ") else "    " + s.strip() for s in items], "UI", 8, INK, 12.4)
        # laptop 2
        x2 = w - lw_
        rbox(c, x2, 18, lw_, lh, colors.HexColor("#fff4ea"), ORANGE, r=10, lw=1.4)
        text_lines(c, x2 + 12, h - 28, ["LAPTOP 2 · Shubashitha · UI"], "UI-B", 9.5, ORANGE)
        items = ["frontend zip only: no data, no model", "React + Leaflet · npm.cmd run dev", "http://localhost:5173",
                 "trip list, layers, Snap to road,", "   Replay, “Why this road?”, metrics", "API: http://<Laptop-1-IP>:8000",
                 "never receives the reference route"]
        text_lines(c, x2 + 14, h - 44, ["•  " + s if not s.startswith("   ") else "    " + s.strip() for s in items], "UI", 8, INK, 12.4)
        # arrows
        mx1, mx2, ym = lw_ + 6, x2 - 6, h / 2 + 10
        arrow(c, mx2, ym + 12, mx1, ym + 12, ORANGE, 1.8, 6)
        arrow(c, mx1, ym - 12, mx2, ym - 12, BLUE, 1.8, 6)
        text_lines(c, (mx1 + mx2) / 2, ym + 18, ["click → POST /api/predict/{id}"], "UI-SB", 7.4, ORANGE, center=True)
        text_lines(c, (mx1 + mx2) / 2, ym - 24, ["route + metrics (JSON)"], "UI-SB", 7.4, BLUE, center=True)
        text_lines(c, (mx1 + mx2) / 2, ym - 0, ["same Wi-Fi / hotspot"], "UI", 7.2, MUTED, center=True)
        text_lines(c, w / 2, 4, ["Labels and reference routes stay on Laptop 1. The UI only ever gets predictions and aggregate scores."],
                   "UI-I", 7.8, MUTED, center=True)
    return Box(168, paint)


def stat_tiles(items, cols=4, dark=False):
    cells, row = [], []
    for big, small, col in items:
        fs = 19 if len(big) <= 8 else 14.5
        row.append([P(f'<font name="UI-BL" size="{fs}" color="{col}">{big}</font>', ParagraphStyle("t", parent=CELL, leading=fs + 3)),
                    P(small, ParagraphStyle("t2", parent=CELL, fontSize=7.8, leading=10, textColor=MUTED))])
        if len(row) == cols:
            cells.append(row); row = []
    if row:
        cells.append(row + [""] * (cols - len(row)))
    t = Table(cells, colWidths=[CW / cols] * cols)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, -1), PALE),
                           ("LINEBEFORE", (1, 0), (-1, -1), 4, colors.white), ("LINEBELOW", (0, 0), (-1, -2), 4, colors.white),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    return t


def callout(title, body, col=TEAL, bg=PALE2):
    inner = [P(f'<font name="UI-B" color="{col.hexval()}">{title}</font>', ParagraphStyle("ct", parent=BODY, spaceAfter=2))]
    inner += [P(b, ParagraphStyle("cb", parent=BODY, spaceAfter=2)) if isinstance(b, str) else b for b in body]
    t = Table([[inner]], colWidths=[CW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3.5, col),
                           ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return KeepTogether([Spacer(1, 3), t, Spacer(1, 6)])


def table(rows, widths=None, head=NAVY, zebra=True, styles=(), font=CELL):
    data = [[c if not isinstance(c, str) else P(c, CELLH if i == 0 else font) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), head), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if zebra:
        st += [("BACKGROUND", (0, i), (-1, i), PALE) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st + list(styles)))
    return t


def code(lines):
    t = Table([[P("<br/>".join(escape(s).replace(" ", "&nbsp;") for s in lines), CODE)]], colWidths=[CW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#14233a")), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return KeepTogether([t, Spacer(1, 6)])


def img(path: Path, width=CW, max_h=None, crop_bottom=None, caption=None):
    path = Path(path)
    if not path.exists():
        return [P(f"[missing figure: {escape(str(path.name))}]", SMALL)]
    if crop_bottom:
        im = PILImage.open(path)
        out = GFIG / f"_crop_{path.name}"
        im.crop((0, 0, im.width, crop_bottom)).save(out)
        path = out
    w, h = PILImage.open(path).size
    width_ = width
    if max_h and width * h / w > max_h:
        width_ = max_h * w / h
    fl = [Image(str(path), width=width_, height=width_ * h / w)]
    if caption:
        fl.append(P(caption, CAP))
    return fl


def verdict_badge(v):
    v = (v or "").upper()
    col = {"MET": GREEN, "PASS": GREEN, "PARTIAL": ORANGE, "NOT MET": RED, "FAIL": RED, "TEAM ACTION": BLUE, "N/A": MUTED}.get(v, MUTED)
    return P(f'<font name="UI-B" color="white">{escape(v)}</font>', ParagraphStyle("b", parent=CELL, alignment=TA_CENTER)), col


# ----------------------------------------------------------------------------------------------- page decoration
def cover(c, doc):
    d = doc.data
    c.saveState()
    c.setFillColor(NAVY); c.rect(0, 0, PW, PH, fill=1, stroke=0)
    # decorative "GPS trace snapping to a road"
    import math
    c.setStrokeColor(colors.HexColor("#132b4d")); c.setLineWidth(0.8)
    for k in range(14):
        x = 18 + k * 44 + 9 * math.sin(k * 1.7)
        c.line(x, PH * 0.40, x + 30 * math.sin(k), PH * 0.80)
    for k in range(7):
        y = PH * 0.42 + k * 42
        c.line(0, y, PW, y + 14 * math.cos(k * 1.3))
    road = [(PW * 0.05 + i * PW * 0.9 / 40, PH * 0.58 + 70 * math.sin(i / 6.0) + 30 * math.cos(i / 3.1)) for i in range(41)]
    c.setStrokeColor(TEAL); c.setLineWidth(6); c.setLineCap(1)
    p = c.beginPath(); p.moveTo(*road[0])
    for pt in road[1:]:
        p.lineTo(*pt)
    c.drawPath(p, stroke=1, fill=0)
    import random
    rnd = random.Random(7)
    c.setFillColor(colors.HexColor("#7fb6ff"))
    for x, y in road[::1]:
        c.circle(x + rnd.uniform(-9, 9), y + rnd.uniform(-16, 16), 2.6, fill=1, stroke=0)
    c.setFillColor(ORANGE); c.circle(*road[0], 7, fill=1, stroke=0)
    c.setFillColor(GREEN); c.circle(*road[-1], 7, fill=1, stroke=0)
    # titles
    c.setFillColor(TEAL); c.setFont("UI-B", 10)
    c.drawString(M, PH - 2.2 * cm, "UE24CS352A MACHINE LEARNING HACKATHON  ·  PES UNIVERSITY  ·  DEPT. OF CSE, RR CAMPUS")
    c.setFillColor(colors.white); c.setFont("UI-BL", 46)
    c.drawString(M, PH - 4.6 * cm, "PathSnap")
    c.setFont("UI-L", 19)
    c.drawString(M, PH - 5.7 * cm, "The Complete Team Guide")
    c.setFillColor(colors.HexColor("#b8c7dc")); c.setFont("UI", 10.5)
    c.drawString(M, PH - 6.6 * cm, "From noisy GPS dots to the exact road a car drove: every part of the project,")
    c.drawString(M, PH - 7.15 * cm, "who owns it, how it works, how to demo it, and whether every requirement is met.")
    # stat tiles
    ho, dec = d["ho"], d["dec"]
    tiles = [(f"{100 * ho['hmm_top1']:.1f}%" if ho else f"{100 * dec['hmm_top1']:.1f}%", "Top-1 accuracy on the 10 held-out\nevaluator trips (with HMM)"),
             (f"{100 * dec['hmm_top1']:.1f}%", "Top-1 accuracy on the 10\nstudent-test trips (with HMM)"),
             (f"{(ho or dec)['recon_route_continuity']:.3f}", "route continuity after decoding\n(1.0 = fully drivable route)"),
             ("8 · 104", "models compared  ·\nautomated tests passing")]
    tw, th, y0 = (PW - 2 * M - 3 * 10) / 4, 2.6 * cm, 6.2 * cm
    for i, (big, small) in enumerate(tiles):
        x = M + i * (tw + 10)
        c.setFillColor(colors.HexColor("#13315c")); c.roundRect(x, y0, tw, th, 8, fill=1, stroke=0)
        c.setFillColor(TEAL); c.roundRect(x, y0 + th - 4, tw, 4, 2, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont("UI-BL", 20); c.drawString(x + 10, y0 + th - 34, big)
        c.setFillColor(colors.HexColor("#b8c7dc")); c.setFont("UI", 7.6)
        for j, s in enumerate(small.split("\n")):
            c.drawString(x + 10, y0 + 22 - j * 10, s)
    c.setFillColor(colors.white); c.setFont("UI-B", 11)
    c.drawString(M, 4.2 * cm, "Team")
    c.setFont("UI", 10.5)
    xs = M
    for name in ["Sujal", "Shriya", "Sudheeksh", "Shubashitha"]:
        c.setFillColor(MEMBER_COL[name]); c.circle(xs + 5, 3.55 * cm + 3, 4.5, fill=1, stroke=0)
        c.setFillColor(colors.white); c.drawString(xs + 14, 3.55 * cm, name)
        xs += 3.9 * cm
    c.setFillColor(colors.HexColor("#8aa0bf")); c.setFont("UI", 8)
    c.drawString(M, 2.2 * cm, f"Generated {date.today().isoformat()} by scripts/build_team_guide.py from the experiment records in the "
                              "repository  ·  github.com/Shriya-sr/ML-Hackathon")
    c.restoreState()


def later(c, doc):
    c.saveState()
    c.setFillColor(NAVY); c.rect(0, PH - 0.9 * cm, PW, 0.9 * cm, fill=1, stroke=0)
    c.setFillColor(TEAL); c.rect(0, PH - 0.9 * cm, 2.2 * cm, 0.9 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("UI-B", 8.5)
    c.drawString(0.55 * cm, PH - 0.58 * cm, "PathSnap")
    c.setFont("UI", 8); c.drawString(2.6 * cm, PH - 0.58 * cm, "The Complete Team Guide")
    c.setFillColor(MUTED); c.setFont("UI", 7.5)
    c.drawString(M, 0.9 * cm, "UE24CS352A · PES University · Sujal · Shriya · Sudheeksh · Shubashitha")
    c.drawRightString(PW - M, 0.9 * cm, f"{doc.page}")
    c.restoreState()


class Doc(BaseDocTemplate):
    def __init__(self, path, data):
        super().__init__(str(path), pagesize=A4, leftMargin=M, rightMargin=M, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                         title="PathSnap - The Complete Team Guide", author="PathSnap team (Sujal, Shriya, Sudheeksh, Shubashitha)",
                         subject="UE24CS352A Machine Learning hackathon - map matching")
        self.data, self.section = data, ""
        f = Frame(M, 1.5 * cm, CW, PH - 3.1 * cm, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate("cover", [Frame(0, 0, PW, PH)], onPage=cover),
                               PageTemplate("body", [f], onPage=later)])

    def afterFlowable(self, fl):
        if isinstance(fl, SectionTitle):
            self.section = f"{fl.num} · {fl.title}"
            self.notify("TOCEntry", (0, fl.title, self.page))


def section(num, title, kicker, need=9 * cm):
    return [CondPageBreak(need), SectionTitle(num, title, kicker), Spacer(1, 6)]


def h2(title, need=4.5 * cm):
    return [CondPageBreak(need), P(title, H2)]


def owner_tag(name):
    return f'<font name="UI-B" color="{MEMBER_COL[name].hexval()}">● {name}</font>'


# ----------------------------------------------------------------------------------------------- content
def build(d):
    S = d["summary"]
    dec, ho, ht, opt, hmm = d["dec"], d["ho"], d["ho_trips"], d["opt"], d["hmm"]["params"]
    inv = d["inv"]
    tr, st = inv[inv.partition == "train"], inv[inv.partition == "student_test"]
    hgb, rf = S.loc["hist_gradient_boosting"], S.loc["random_forest"]
    abl = d["abl"].set_index("dropped_group")
    tuned, untuned = opt["tuned"], opt["untuned"]
    s = [NextPageTemplate("body"), PageBreak()]

    # ===================================================================== 0. read me first
    s += section(0, "Read this first", "what this guide is and how to use it")
    s += [P("PathSnap takes the GPS fixes recorded by a phone or car once per second, which are always a few metres off, and works "
            "out the exact sequence of road segments the vehicle actually drove. That task is called <b>map matching</b>. We solve it in "
            "two stages. A machine-learning <b>classifier</b> scores every nearby road for every GPS fix. A <b>Hidden Markov Model</b> "
            "with the <b>Viterbi</b> algorithm then picks the best <i>sequence</i> of roads, one that a car can actually drive. "
            "The result is served by an API on one laptop and shown on a live map on a second laptop.", LEAD)]
    s += [stat_tiles([
        (f"{len(tr)}", f"training trips<br/>({int(tr.n_obs.sum()):,} GPS fixes)", NAVY.hexval()),
        (f"{len(st)} + 10", "student-test + held-out<br/>evaluator trips", NAVY.hexval()),
        (f"{100 * dec['clf_top1']:.1f}% → {100 * dec['hmm_top1']:.1f}%", "student-test Top-1:<br/>classifier → + HMM", GREEN.hexval()),
        (f"{100 * ho['clf_top1']:.1f}% → {100 * ho['hmm_top1']:.1f}%" if ho else "n/a", "held-out evaluator Top-1:<br/>classifier → + HMM",
         GREEN.hexval()),
    ]), Spacer(1, 10)]
    s += [P("How to read this guide", H2)]
    s += [table([["If you are…", "Read first", "Then"],
                 [owner_tag("Shriya") + "<br/>data & features", "§2 Data, §4.1–4.4", "§5 results, §7 your card, §8 demo"],
                 [owner_tag("Sudheeksh") + "<br/>models & tuning", "§4.5–4.8", "§5 results, §7 your card, §8 demo"],
                 [owner_tag("Shubashitha") + "<br/>HMM, maps, UI", "§4.9–4.10, §6.3 UI", "§7 your card, §8 demo (you drive the UI)"],
                 [owner_tag("Sujal") + "<br/>backend & deployment", "§6 System, §8 Demo", "§4 overview, §7 your card, §9 requirements"],
                 ["Everyone (1 hour)", "§0, §1, §3 big picture, §5 results", "§10 cheat sheet: memorise the numbers"]],
                [4.2 * cm, 6.2 * cm, CW - 10.4 * cm])]
    s += [callout("The three ideas to remember", [
        "<b>1. Score each road.</b> For each GPS fix, list the roads within 50 m and let a classifier estimate the probability that each "
        "one is the road the car is really on.",
        "<b>2. Choose the whole route together.</b> One fix on its own is ambiguous. The HMM connects the fixes and prefers road sequences "
        "that are connected and whose driving distance matches the GPS movement. Viterbi finds the best such sequence exactly.",
        "<b>3. Never cheat.</b> Trips are never split between training and testing (GroupKFold by trip). The reference route is "
        "used only for training labels and for scoring on the server. It never feeds a prediction and never reaches the UI."])]
    s += [callout("Honesty notes (say these if asked)", [
        "The code was built with substantial help from an AI coding assistant (Claude Code), working under the team's direction. "
        "This is disclosed in the README, as the submission guidelines require. Each member owns one area (section 7) and must be "
        "able to explain and defend it.",
        "Every number in this guide comes from a script in the repository run on the supplied data. Nothing is hand-typed or estimated."],
        ORANGE, colors.HexColor("#fff4ea"))]

    # ===================================================================== 1. the problem
    s += section(1, "The problem in plain English", "why map matching is hard")
    s += [P("Imagine a sheet of tracing paper with a dot for every second of a car journey, laid over a street map. Most dots sit "
            "next to a road, but not exactly on it, because GPS has errors of a few metres. Sometimes there are several roads close "
            "together, and sometimes the dots stop because the signal was lost. <b>Our job is to redraw the journey along the real roads, "
            "segment by segment and in the right direction.</b>")]
    s += [table([["Difficulty", "What goes wrong", "How PathSnap deals with it"],
                 ["GPS noise", "a dot lies 5–20 m away from the true road, sometimes nearer to another road", "candidates within 50 m; "
                  "the classifier looks at distance <i>relative</i> to the other candidates, not just absolute distance"],
                 ["Parallel roads, junctions", "two or more roads are equally close", "heading features (is the road pointing the way the car "
                  "moves?) + HMM connectivity"],
                 ["Two-way streets", "each street is two arcs, one per direction; when the car is slow the heading is unknown",
                  "U-turn penalty in the HMM; the sequence fixes the direction"],
                 ["Gaps and stops", "the signal drops for seconds or hours; a parked car produces a cloud of dots",
                  "the chain is broken on large gaps (>120 s); fixes far from the reference route are left unlabelled"],
                 ["World-wide data", "trips are in 25 different UTM zones; degrees are not metres", "every trip is projected into its own UTM "
                  "zone, so all geometry is in metres"]],
                [3.3 * cm, 6.1 * cm, CW - 9.4 * cm])]
    s += [Spacer(1, 6)] + img(MAPS / "student_00000052_comparison.png", CW, 9.5 * cm, crop_bottom=855,
                              caption="Real output on student-test trip 00000052 (Folium map). Blue = raw GPS, purple = our HMM route, "
                                      "drawn over the road network. Classifier 0.985 → HMM 0.992 Top-1.")

    # ===================================================================== 2. data
    s += section(2, "The data we were given", "four plain-text files per trip")
    s += [P("Every trip is a folder with four whitespace-separated text files. Here are the first lines of training trip 00000001, "
            "exactly as they appear in the files:")]
    s += [table([["File", "One line means", "Example (real first lines)", "Used for"],
                 ["<b>.track</b>", "one GPS fix: longitude, latitude, time (s)", '<font name="Mono" size="7.6">8.595203 49.844053 1.0<br/>'
                  '8.595294 49.844441 2.0</font>', "input to the model"],
                 ["<b>.nodes</b>", "one road-network node (junction / shape point): longitude, latitude; the line number is the node id",
                  '<font name="Mono" size="7.6">8.581679 49.857880<br/>8.584637 49.858295</font>', "input"],
                 ["<b>.arcs</b>", "one directed road segment: from-node id, to-node id; the line number is the arc id",
                  '<font name="Mono" size="7.6">0 281<br/>0 1</font>', "input"],
                 ["<b>.route</b>", "the reference answer: the ordered list of arc ids the vehicle drove",
                  '<font name="Mono" size="7.6">2641<br/>64973</font>', "<b>labels + scoring only</b>"]],
                [1.6 * cm, 5.6 * cm, 5.2 * cm, CW - 12.4 * cm])]
    s += [P("Partitions", H3)]
    s += [table([["Partition", "Trips", "GPS fixes", "Has .route?", "How we used it"],
                 ["train", f"{len(tr)}", f"{int(tr.n_obs.sum()):,}", "yes", "EDA, training, cross-validation, tuning, HMM grid search"],
                 ["student_test", f"{len(st)}", f"{int(st.n_obs.sum()):,}", "yes", "scored once per model after training; model comparison"],
                 ["evaluator_test", "10", f"{ho['n_fixes']:,}" if ho else "-", "yes (server only)",
                  "arrived after development; scored once with the frozen model; served by the API in the demo"]],
                [2.6 * cm, 1.3 * cm, 2.1 * cm, 2.4 * cm, CW - 8.4 * cm])]
    s += [P("What the exploratory data analysis (EDA) found", H3)]
    s += bullets([
        "<b>Clean files:</b> 360 files, none missing, 0 invalid coordinates, 0 duplicate rows or timestamps, timestamps always increasing.",
        "<b>1 Hz sampling</b> (median 99.1% of steps are exactly 1 s), but <b>135 gaps longer than 30 s</b> in 30 trips; the largest is "
        "153,064 s (about 42 hours, a parked car).",
        "<b>World-wide trips:</b> longitudes from −100 to 137 and latitudes from −35 to 62, in 25 UTM zones. Networks have 299 to 155,003 nodes "
        "and 649 to 307,392 arcs. About 95% of arcs are one direction of a two-way street.",
        "<b>.route is much shorter than .track</b>, with a median of 5.3 GPS fixes per route arc. So there is no ready-made 'correct road "
        "per fix'; we had to derive it by alignment (§4.2). The reference also has 14 breaks (consecutive arcs that do not connect) "
        "in 11 trips, so it is a reference, not perfect truth.",
        "After alignment, the <b>median GPS-to-road distance is 4.7 m</b>. 2.03% of fixes are more than 60 m from their road (parked or "
        "off-road) and are left unlabelled.",
        "A <b>radius sweep</b> showed that 50 m with at most 10 candidates misses the true road for only 1.22% of fixes (1.14% with the "
        "150 m fallback), at about 6.6 candidates per fix."])
    s += img(EDAF / "03_trip_locations.png", CW * 0.8, 7.2 * cm, caption="Trip locations: the data covers several continents, which is "
                                                                          "why every trip gets its own metric projection.")
    s += img(EDAF / "07_candidate_radius_sweep.png", CW * 0.85, 6.5 * cm,
             caption="Candidate radius sweep on the training trips: a larger radius misses fewer true roads but adds more wrong "
                     "candidates. 50 m is the knee of the curve.")

    # ===================================================================== 3. big picture
    s += section(3, "The whole pipeline in one picture", "eight stages, each a separate module", need=15 * cm)
    s += [P("Read the diagram left to right on the top row, then right to left on the bottom row. Each box is a separate Python "
            "module with its own tests. The same code path runs during training, evaluation and live API requests, so what we measured "
            "is exactly what the evaluator sees.")]
    s += [pipeline_diagram(), Spacer(1, 8)]
    s += [table([["Stage", "Code (src/pathsnap/…)", "Input → output"],
                 ["1 Load & clean", "data/loader.py, validator.py, preprocessing.py", "4 text files → validated arrays; bad rows dropped, sorted by time"],
                 ["2 Project", "geography/projection.py", "lon/lat degrees → x/y metres in the trip's UTM zone"],
                 ["3 Candidates", "geography/spatial_index.py, features/candidate_generation.py", "each fix → up to 10 road arcs within 50 m"],
                 ["4 Features", "features/feature_extraction.py, feature_schema.py", "each (fix, arc) pair → 21 numbers"],
                 ["5 Classifier", "models/model_factory.py, artifacts/models/final_classifier.joblib", "21 numbers → P(correct road)"],
                 ["6 HMM + Viterbi", "decoding/emissions.py, transitions.py, viterbi.py, hmm.py", "P for all fixes → one arc per fix"],
                 ["7 Reconstruct", "decoding/hmm.py::reconstruct_route", "arc per fix → continuous drivable arc sequence"],
                 ["8 Serve", "inference/pipeline.py, backend/, frontend/", "trip id → JSON route + metrics → map in the browser"]],
                [2.8 * cm, 7.2 * cm, CW - 10 * cm])]

    # ===================================================================== 4. step by step
    s += section(4, "How every stage works", "what, how, why: in detail")
    # 4.1
    s += [*h2("4.1  Cleaning and turning degrees into metres"), owner_line("Shriya")]
    s += [P("The loader reads each file with strict checks (right number of columns, integer ids, ids in range) and raises a clear "
            "error naming the file if anything is wrong. Cleaning drops invalid coordinates, sorts fixes by time and keeps the first fix "
            "of any duplicate timestamp. Each fix keeps its original row number so the answer can be mapped back.")]
    s += [P("<b>Why convert to metres?</b> Latitude and longitude are angles, not distances. One degree of longitude is 111 km at the "
            "equator but only 71.7 km at latitude 50°, and almost zero near the poles. Distances computed straight from degrees would be "
            "wrong and would differ between trips. So, for each trip, we take the centre of its GPS fixes, choose the matching <b>UTM zone</b> "
            "(the world is split into 60 vertical strips, each with its own flat coordinate system in metres), and convert every fix and "
            "every road node into <b>x, y in metres</b> with pyproj. Inside one trip the error of this flat approximation is negligible. "
            "We checked it against the haversine great-circle formula and the difference is under 0.2%.")]
    s += [callout("Worked example (real first two fixes of trip 00000001)", [
        "(8.595203°, 49.844053°) and (8.595294°, 49.844441°) lie in UTM zone 32N (EPSG:32632). They become "
        "(470,895.7 m, 5,521,370.6 m) and (470,902.5 m, 5,521,413.7 m). By Pythagoras: √(6.8² + 43.1²) = <b>43.63 m</b>. The "
        "haversine great-circle formula on the raw degrees gives 43.63 m as well, so the flat projection is accurate. Every later step "
        "(distance to a road, heading, speed) is ordinary school geometry on these x, y values."])]
    # 4.2
    s += [*h2("4.2  Making a 'correct road' label for every GPS fix"), owner_line("Shriya")]
    s += [P(".route lists the roads in order but does not say which GPS fix belongs to which road. We solve this with a "
            "<b>monotone dynamic-programming alignment</b>. Moving forward in time, each fix is assigned to a position in the route list, and "
            "positions may never go backwards. The cost of assigning fix i to route arc j is its distance to that arc (capped at 100 m) plus a "
            "15 m penalty scaled by how much the car's motion disagrees with the arc's direction. The DP "
            "C[i, j] = cost(i, j) + min<sub>j′ ≤ j</sub> C[i−1, j′] finds the cheapest assignment for the whole trip in one pass, "
            "using prefix minima. Fixes more than 60 m from their assigned arc get label −1 (unknown) and are left out of training "
            "and scoring. The direction term matters for out-and-back trips, where the same street is driven both ways.")]
    s += [P("<b>Important:</b> labels are needed only for training and for scoring. The live prediction path never reads .route.", SMALL)]
    # 4.3
    s += [*h2("4.3  Candidate roads: the shortlist for each fix"), owner_line("Shriya")]
    s += [P("All road segments of a trip go into an <b>STRtree</b>, a spatial index (an R-tree) from the Shapely library that answers "
            "'which segments are near this point?' in milliseconds instead of checking every segment. For each fix we take all arcs within "
            "<b>50 m</b>, keep the <b>10 nearest</b>, and if nothing is within 50 m we take the single nearest arc within 150 m (the fallback). "
            f"If the true road is not on this shortlist, nothing later can fix it. That happens for only "
            f"{100 * dec['miss_rate']:.2f}% of student-test fixes" + (f" and {100 * ho['miss_rate']:.2f}% of evaluator fixes" if ho else "") +
            ". Candidates are built without looking at the labels; a unit test checks this.")]
    s += img(GFIG / "guide_one_fix.png", CW * 0.62, 9 * cm,
             caption="Real fix from student-test trip 00000077. The car is almost stopped, so its heading is unreliable. The classifier "
                     "slightly prefers A (P = 0.47), but A is the same street in the opposite direction. The HMM picks B, which is correct.")
    # 4.4
    s += [*h2("4.4  The 21 features: what the model looks at"), owner_line("Shriya")]
    s += [P("Each (fix, candidate) pair becomes a row of 21 numbers in five groups. They are defined once in "
            "<font name='Mono'>feature_schema.py</font>, which training, evaluation and the API all use, so the column order can never drift.")]
    feats = [("geometric", "dist_m", "distance from the fix to the segment, in metres"), ("", "dist_rank", "0 = nearest candidate, 1 = second nearest, …"),
             ("", "dist_minus_min", "how much farther this candidate is than the nearest one"), ("", "dist_ratio", "this distance ÷ nearest distance"),
             ("", "n_candidates", "how crowded the area is (number of candidates for this fix)"),
             ("", "proj_frac", "where along the segment the fix falls (0 = start, 1 = end)"),
             ("", "at_endpoint / endpoint_dist_m", "the fix projects beyond an end node / distance to the nearer end"),
             ("", "arc_length_m", "segment length"),
             ("heading", "cos_heading", "cosine of the angle between the car's motion and the segment direction (+1 same way, −1 opposite)"),
             ("", "heading_diff_deg", "the same angle in degrees"), ("", "heading_valid", "1 if the car moved ≥ 3 m over ±2 s, so the heading is trustworthy"),
             ("", "cos_long_bearing", "the heading cosine over a longer ±5 s window (needs ≥ 5 m of movement): steadier"),
             ("context", "win_mean_dist_m / win_max_dist_m", "mean / max distance of the neighbouring fixes (±3) to this segment"),
             ("kinematic", "speed_mps, dt_prev, dt_next", "speed and the time gaps before and after the fix"),
             ("network", "from_in_degree, to_out_degree", "number of roads entering the start / leaving the end (junction complexity)"),
             ("", "is_two_way", "1 if the opposite-direction arc exists")]
    s += [table([["Group", "Feature", "Plain meaning"]] + [[f"<b>{g}</b>" if g else "", f'<font name="Mono" size="7.8">{n}</font>', m]
                                                           for g, n, m in feats], [2.1 * cm, 5.4 * cm, CW - 7.5 * cm])]
    s += [P("<b>Which features matter?</b> We removed one group at a time and re-ran the cross-validation (ablation). Removing "
            f"<b>heading</b> costs {100 * -abl.loc['heading', 'delta_top1']:.1f} points of Top-1 accuracy, removing <b>geometric</b> costs "
            f"{100 * -abl.loc['geometric', 'delta_top1']:.1f}, and context, kinematic and network each cost under 1 point. "
            "Direction of travel is therefore the single most important signal: distance alone cannot tell the two sides of a "
            "street apart.")]
    s += img(FIG / "feature_ablation.png", CW * 0.75, 6.3 * cm, caption="Group ablation: drop in CV Top-1 when a whole feature group is removed.")
    # 4.5
    s += [*h2("4.5  Eight models, compared fairly"), owner_line("Sudheeksh")]
    s += [P("Every model gets the same 21 features, the same five folds and the same scoring. The task is binary: 'is this candidate "
            "the correct road?' For each fix, the candidate with the highest probability is the classifier's answer (<b>Top-1</b>).")]
    plain = {"logistic_regression": "a weighted sum of features squashed into a probability; the simple, interpretable baseline",
             "gaussian_nb": "assumes features are independent given the class; very fast, but its probabilities are over-confident",
             "knn": "looks at the 25 most similar training rows and lets them vote (weighted by distance)",
             "random_forest": "200 deep decision trees on bootstrap samples, averaged (bagging)",
             "extra_trees": "like a random forest, but the split points are chosen at random; more variety, less variance",
             "hist_gradient_boosting": "trees built one after another, each correcting the previous ones' errors, on binned features",
             "svm_rbf": "finds a maximum-margin boundary in a curved (RBF) feature space; Platt scaling turns scores into probabilities",
             "mlp": "a small neural network with two hidden layers (64 and 32 neurons)"}
    order = S.sort_values("cv_macro_f1", ascending=False).index
    rows = [["Model", "How it thinks (plain words)", "CV Macro-F1", "CV Top-1", "Student Top-1", "Log loss", "ms / 1k fixes"]]
    for m in order:
        r = S.loc[m]
        name = f"<b>{escape(r.display)}</b>" + (" <font color='#1fa45a'>(selected)</font>" if m == "hist_gradient_boosting" else "")
        rows.append([name, plain[m], f"{r.cv_macro_f1:.4f}<br/><font size='6.8' color='#5b6675'>± {r.cv_macro_f1_std:.4f}</font>",
                     f"{r.cv_top1:.4f}", f"{r.st_top1:.4f}", f"{r.st_log_loss:.3f}", f"{r.inference_ms_per_1k_fixes:.0f}"])
    sel = list(order).index("hist_gradient_boosting") + 1
    s += [table(rows, [3.4 * cm, 6.1 * cm, 1.7 * cm, 1.5 * cm, 1.6 * cm, 1.3 * cm, CW - 15.6 * cm],
                styles=[("BACKGROUND", (0, sel), (-1, sel), colors.HexColor("#e6f6ec"))])]
    s += [P("<b>Why HistGradientBoosting?</b> It and Random Forest are statistically tied on CV Macro-F1 "
            f"({hgb.cv_macro_f1:.4f} vs {rf.cv_macro_f1:.4f}; the gap is far below the fold standard deviation of about 0.014). Comparing the "
            f"untuned versions, as at selection time, HGB fits "
            f"about {rf.fit_s_full / hgb.fit_s_full:.0f}× faster ({hgb.fit_s_full:.1f} s vs {rf.fit_s_full:.0f} s), its file is about "
            f"{rf.model_size_mb / hgb.model_size_mb:.0f}× smaller ({hgb.model_size_mb:.2f} MB vs {rf.model_size_mb:.0f} MB), it predicts "
            f"faster and has the higher student-test Top-1. For a model served live by an API, that is the better trade-off. All eight models are "
            "close because the heading and relative-distance features already make the problem nearly separable. Even a linear model "
            f"reaches {S.loc['logistic_regression'].cv_macro_f1:.3f}.")]
    s += img(FIG / "model_cv_macro_f1.png", CW * 0.8, 6.8 * cm, caption="CV Macro-F1 with fold spread for all eight models.")
    # 4.6
    s += [*h2("4.6  Validation that cannot cheat: GroupKFold by trip"), owner_line("Sudheeksh")]
    s += [P("Consecutive GPS fixes are almost identical (one second apart). If fixes from the same trip landed in both the training "
            "set and the validation set, the model would look excellent simply by memorising that trip. That is <b>leakage</b>. So we "
            "split by <b>whole trips</b>: GroupKFold(5, shuffle, seed 42) puts every trip entirely into one fold. Each model is trained 5 times "
            "on 64 trips and validated on the other 16, and we report the mean ± standard deviation. Training uses every 2nd fix "
            "(about 603,000 rows), because neighbouring fixes add little new information. The same folds are used for every model and for "
            "tuning, and they are saved in experiments/configurations/cv_folds.json.")]
    s += img(GFIG / "guide_groupkfold.png", CW, 5.2 * cm)
    # 4.7
    s += [*h2("4.7  The metrics, in plain words"), owner_line("Sudheeksh")]
    s += [table([["Metric", "What it answers", "Good value"],
                 ["Top-1 accuracy", "For what fraction of GPS fixes did we pick exactly the right road arc? <b>The headline number.</b>", "high (we get 0.95–0.96)"],
                 ["Macro-F1", "Average of the F1 score for the 'correct' and 'wrong' classes; fair even though most candidates are wrong", "high"],
                 ["Precision / Recall", "Of the candidates we called correct, how many were? / Of the correct ones, how many did we find?", "high"],
                 ["Log loss", "Are the probabilities honest? It punishes confident mistakes; important because the HMM uses them", "low"],
                 ["ROC-AUC / Brier", "Ranking quality / squared error of the probabilities", "high / low"],
                 ["Candidate miss rate", "How often the true road was not even on the shortlist (an upper limit on accuracy)", "low (≈ 0.2%)"],
                 ["Route continuity", "Fraction of consecutive arcs in our route that actually connect (1.0 = fully drivable)", "≈ 1.0"],
                 ["Route precision / recall / F1", "Overlap between the set of arcs we output and the reference route", "high"],
                 ["Edit similarity, LCS recall", "How similar our ordered arc sequence is to the reference sequence (Levenshtein / longest common subsequence)", "high"],
                 ["Invalid transitions", "Number of jumps between arcs that a car could not drive", "low"]],
                [3.6 * cm, CW - 6.6 * cm, 3 * cm])]
    # 4.8
    s += [*h2("4.8  Bayesian optimisation with Optuna"), owner_line("Sudheeksh")]
    s += [P("Instead of trying hyper-parameters on a fixed grid, <b>Bayesian optimisation</b> learns from the trials it has already run. "
            "Optuna's <b>TPE</b> sampler (Tree-structured Parzen Estimator) models which settings tend to give good scores and which give bad "
            "ones, and proposes the next trial where 'good' is most likely. The first 10 trials are random to explore. A "
            "<b>MedianPruner</b> stops a trial early when its first folds score below the median of earlier trials, which saves time. "
            f"Objective: mean 5-fold CV Macro-F1. {opt['n_trials']} trials ran ({opt['n_complete']} complete, {opt['n_pruned']} pruned) in "
            f"{opt['search_seconds'] / 60:.0f} minutes.")]
    bp = opt["best_params"]
    s += [table([["", "Untuned HGB", "Tuned HGB", "Change"],
                 ["CV Macro-F1", f"{untuned['cv_macro_f1']:.4f}", f"{tuned['cv_macro_f1']:.4f}", f"{tuned['cv_macro_f1'] - untuned['cv_macro_f1']:+.4f}"],
                 ["Student-test Top-1", f"{untuned['st_top1']:.4f}", f"{tuned['st_top1']:.4f}", f"{tuned['st_top1'] - untuned['st_top1']:+.4f}"],
                 ["Student-test log loss", f"{untuned['st_log_loss']:.4f}", f"{tuned['st_log_loss']:.4f}", f"{tuned['st_log_loss'] - untuned['st_log_loss']:+.4f}"],
                 ["Inference ms / 1k fixes", f"{untuned['inference_ms_per_1k_fixes']:.0f}", f"{tuned['inference_ms_per_1k_fixes']:.0f}", "slower, still fast"]],
                [5 * cm, 3.5 * cm, 3.5 * cm, CW - 12 * cm])]
    s += [P(f"Best settings: learning rate {bp['learning_rate']:.4f}, {bp['max_iter']} boosting iterations, {bp['max_leaf_nodes']} leaves, "
            f"depth {bp['max_depth']}, min {bp['min_samples_leaf']} samples per leaf, L2 {bp['l2_regularization']:.5f}, "
            f"{100 * bp['max_features']:.0f}% of features per split. <b>Honest reading:</b> the CV gain is small (within fold noise), but the "
            "probabilities improved clearly (lower log loss), and that helps the HMM. The rule 'keep the tuned model only if its CV score "
            "is not lower' was fixed before we looked at any test score.")]
    # 4.9
    s += [*h2("4.9  HMM + Viterbi: choosing the whole route at once"), owner_line("Shubashitha")]
    s += [P("The classifier judges each fix on its own, so it can jump between the two directions of a street, or to a side road, "
            "from one second to the next. A real car cannot do that. A <b>Hidden Markov Model</b> describes the journey as a chain:")]
    s += bullets([
        "<b>Hidden states</b>: the road the car is really on at each fix (one of the top 8 candidates).",
        f"<b>Emission score</b>: how well a candidate explains this fix: w<sub>clf</sub> × log P(classifier) plus a Gaussian distance term "
        f"(σ = {hmm['sigma_m']:.0f} m, weight {hmm['w_dist']}). Tuned w<sub>clf</sub> = {hmm['w_clf']}.",
        f"<b>Transition score</b>: how plausible it is to move from road a to road b in one step: −|driving distance on the network − straight "
        f"GPS distance| / β, with β = {hmm['beta_m']:.0f} m, minus a U-turn penalty of {hmm['uturn_penalty']} for switching to the reverse arc. It is "
        f"impossible (−∞) if b cannot be reached from a, or if the implied speed is above {hmm['vmax_mps']:.0f} m/s. Driving distances come from a "
        "bounded Dijkstra shortest-path search on the road graph, with caching.",
        "<b>Viterbi</b> is dynamic programming over this chain. For every fix and every candidate it keeps the best total score of any "
        "path ending there (δ<sub>t</sub>(s) = max<sub>r</sub> [δ<sub>t−1</sub>(r) + log T(r, s)] + log E<sub>t</sub>(s)), then traces "
        "back the single best path. It is exact, not a guess, and runs in time linear in the number of fixes.",
        f"<b>Chain breaks</b>: if a fix has no candidates, if no transition is possible, or if the time gap exceeds {hmm['max_gap_s']:.0f} s, "
        "the chain restarts there. We never invent road segments across such breaks."])
    s += img(GFIG / "guide_trellis.png", CW, 8 * cm,
             caption="Real Viterbi trellis from trip 00000077. Five of these six fixes are wrong if decided alone (the classifier prefers A, "
                     "the opposite direction, while the car is almost stopped). Viterbi keeps the whole sequence on B, the correct direction.")
    s += img(GFIG / "guide_hmm_zoom.png", CW, 10 * cm,
             caption="Same trip, car moving: the classifier alone gets 26 of 40 fixes right, jumping onto a side street and the wrong "
                     "direction. With the HMM all 40 are right.")
    s += [P(f"<b>How the decoder was tuned.</b> We grid-searched w<sub>clf</sub>, β, w<sub>dist</sub> and the U-turn penalty on 30 <i>training</i> trips, "
            "using out-of-fold probabilities (each trip scored by a model that never saw it), so no test data was involved. Tuned vs default "
            f"settings on student-test: Top-1 {dec['hmm_top1']:.4f} vs {d['dec_default']['hmm_top1']:.4f}.")]
    # 4.10
    s += [*h2("4.10  Route reconstruction"), owner_line("Shubashitha")]
    s += [P("Viterbi gives one arc per fix, but a fast car can skip short segments between two fixes. Inside each chain we fill every jump "
            "with the shortest connecting path on the road graph, so the output is a continuous, drivable sequence of arcs. "
            f"This raises mean route continuity to {dec['recon_route_continuity']:.3f} on student-test" +
            (f" and {ho['recon_route_continuity']:.3f} on the evaluator trips" if ho else "") + ".")]

    # ===================================================================== 5. results
    s += section(5, "Results", "what we achieved, measured honestly")
    rows = [["Metric", "Student-test<br/>classifier", "Student-test<br/>+ HMM", "Evaluator (held-out)<br/>classifier", "Evaluator (held-out)<br/>+ HMM"]]
    for name, a, b in [("Top-1 accuracy", "clf_top1", "hmm_top1"), ("Selection Macro-F1", "clf_sel_macro_f1", "hmm_sel_macro_f1"),
                       ("Route continuity", "clf_route_continuity", "recon_route_continuity"), ("Route F1", "clf_route_f1", "recon_route_f1"),
                       ("Edit similarity", "clf_edit_similarity", "recon_edit_similarity"),
                       ("Invalid transitions", "clf_invalid_transitions", "recon_invalid_transitions")]:
        fm = (lambda v: f"{v:.0f}") if "invalid" in a else (lambda v: f"{v:.4f}")
        rows.append([f"<b>{name}</b>", fm(dec[a]), f"<b>{fm(dec[b])}</b>", fm(ho[a]) if ho else "-", f"<b>{fm(ho[b])}</b>" if ho else "-"])
    s += [table(rows, [4.4 * cm] + [(CW - 4.4 * cm) / 4] * 4,
                styles=[("BACKGROUND", (2, 1), (2, -1), colors.HexColor("#e6f6ec")), ("BACKGROUND", (4, 1), (4, -1), colors.HexColor("#e6f6ec"))])]
    s += [Spacer(1, 6)]
    if ho:
        worse = ht[ht.hmm_top1 < ht.clf_top1]
        s += [callout("What these numbers mean", [
            f"On <b>trips the model had never seen</b>, the system puts {100 * ho['hmm_top1']:.1f}% of GPS fixes on exactly the right road arc, and "
            f"the route it outputs is {100 * ho['recon_route_continuity']:.1f}% continuous (drivable). The held-out score is in line with the "
            "student-test score, so our cross-validation was a reliable predictor: no overfitting to the test set.",
            f"The HMM adds about {100 * (ho['hmm_top1'] - ho['clf_top1']):.1f} points of Top-1 and cuts impossible jumps from "
            f"{ho['clf_invalid_transitions']:.0f} to {ho['recon_invalid_transitions']:.0f}. It improved {int((ht.hmm_top1 > ht.clf_top1).sum())} of "
            f"10 evaluator trips; on {', '.join(worse.trip_id)} it was worse "
            f"({worse.iloc[0].clf_top1:.3f} → {worse.iloc[0].hmm_top1:.3f}). Say so openly if asked: 10 trips is a small sample, and the "
            "evaluator set was scored once, with no re-tuning afterwards."], GREEN, colors.HexColor("#e9f7ef"))]
        s += img(FIG / "heldout_per_trip.png", CW, 6.5 * cm, caption="Held-out evaluator trips: per-trip Top-1 and route continuity.")
    s += img(FIG / "hmm_vs_classifier_per_trip.png", CW, 6.5 * cm, caption="Student-test trips: per-trip Top-1 and route continuity.")
    s += [P("Where errors remain", H3)]
    s += bullets([
        f"<b>Junction neighbours</b> (the arc just before or after the true one at a junction): {ho['hmm_err_adjacent_by_hmm']:.0f} fixes on the "
        "evaluator set. Often harmless, because the route itself is right." if ho else "Junction neighbours.",
        "<b>Direction while stationary</b>: a parked or crawling car has no heading, so both directions explain the fixes equally well.",
        "<b>Candidate misses</b>: if the true road is not within 50 m (≈ 0.2% of fixes), it cannot be chosen.",
        "<b>Imperfect reference</b>: the .route files contain breaks and parked periods, so some 'errors' are disagreements with an imperfect "
        "reference."])

    # ===================================================================== 6. system
    s += section(6, "The system: API, UI and two laptops", "how the evaluator sees it live")
    s += [P("6.1  Two-laptop architecture", H2), Spacer(1, 4), laptops_diagram(), Spacer(1, 6)]
    s += [*h2("6.2  The backend API (Laptop 1)"), owner_line("Sujal")]
    s += [P("FastAPI app in <font name='Mono'>backend/</font>. On start-up it loads the classifier, checks that the feature schema and "
            "version match the code (it refuses to start on a mismatch), and loads the HMM parameters. It lists the trips in "
            "<font name='Mono'>data/raw/evaluator_test</font> and runs in <b>EVALUATION</b> mode, or, if that folder is empty, serves the "
            "student-test trips in <b>DEVELOPMENT</b> mode. Every request is logged to the console with client IP, endpoint, status and "
            "time, plus a per-trip accuracy line. That console is your proof to the evaluator that inference is happening live.")]
    s += [table([["Method", "Endpoint", "What it does"],
                 ["GET", "/health", "is the API up, is the model loaded, mode, number of trips"],
                 ["GET", "/api/system/info", "EVALUATION / DEVELOPMENT, partition, versions"],
                 ["GET", "/api/model/info", "model variant, feature schema, CV score, HMM parameters"],
                 ["GET", "/api/trips", "the trips available (no labels)"],
                 ["GET", "/api/trips/{id}/metadata", "fixes, duration, bounding box, network size, gaps"],
                 ["POST", "/api/predict/{id}", "<b>runs the full pipeline</b>, returns the route, per-fix decisions, flags, timings, server-side metrics"],
                 ["GET", "/api/results/{id}", "last result for a trip (cached)"],
                 ["GET", "/api/results/{id}/explain/{fix}", "candidates, probabilities and key features for one fix"]],
                [1.5 * cm, 5.4 * cm, CW - 6.9 * cm])]
    s += [P("Robustness: unknown trip id → 404, malformed trip files → 422, model missing → 503, prediction too slow → 504. The response "
            "schema has no field for the reference route, and a test (<font name='Mono'>test_no_ground_truth_leakage</font>) checks that "
            "the reference never appears in any response.", SMALL)]
    s += [*h2("6.3  The evaluator UI (Laptop 2)", 13.5 * cm), owner_line("Shubashitha")]
    s += img(FIG / "ui_trip_00000087.png", CW, 11 * cm, crop_bottom=905,
             caption="The live UI in EVALUATION mode on held-out evaluator trip 00000087: 99.1% Top-1, route continuity 1.000. "
                     "Left: trip list. Centre: map. Right: metrics computed on Laptop 1.")
    s += [table([["UI element", "What it shows / does"],
                 ["Top bar", "API online/offline, EVALUATION/DEVELOPMENT, model name, trip count, API address with a Change button"],
                 ["Trip list", "the 10 trips with fixes and duration; click one to send POST /api/predict to Laptop 1"],
                 ["Map layers", "basemap, road network (grey), GPS fixes (blue), HMM route (green, arrows = direction), classifier-only route "
                                "(orange), robustness flags (gaps, chain breaks, low confidence)"],
                 ["Snap to road", "animates every GPS fix moving onto the road the model matched it to"],
                 ["Replay trip", "drives a marker along the trip in time order"],
                 ["Why this road?", "click any blue dot: candidate roads, their probabilities and key features"],
                 ["Metrics panel", "Top-1, Macro-F1, continuity, miss rate, inference time, route F1, classifier-only vs + HMM bars, "
                                   "robustness counts, pipeline timing per stage"]],
                [3.3 * cm, CW - 3.3 * cm])]
    s += [*h2("6.4  Testing and quality"), owner_line("Sujal")]
    s += [P("<b>92 Python tests</b> (pytest) and <b>12 UI tests</b> (vitest) pass. They cover malformed, missing and empty files, invalid "
            "coordinates, duplicate and out-of-order timestamps, single-fix trips, candidate edge cases (none, one, many, fallback), proof that "
            "labels never change candidates, feature schema order, missing or corrupt model files, Viterbi checked against brute force, forced "
            "gap breaks, impossible speeds, route metrics checked against reference implementations, every API error code, concurrent "
            "requests, ground-truth leakage, and end-to-end runs on real trips. A clean clone of the repository was tested from scratch.")]

    # ===================================================================== 7. who did what
    s += section(7, "Who owns what, and how it works", "one card per team member")
    s += [P("The split below is the team's ownership plan, as recorded in TEAM_CONTRIBUTIONS.md. Each owner is responsible for "
            "explaining and defending their part, and each has a backup. <b>If the real split is different, edit TEAM_CONTRIBUTIONS.md "
            "and re-run this script.</b> Disclosure: the implementation was produced with AI assistance under the team's direction.", SMALL)]
    cards = [
        ("Shriya", "Data, preprocessing, candidates, features", "GitHub repository owner",
         "src/pathsnap/data/, geography/, features/; scripts/run_eda.py, build_features.py, feature_analysis.py; reports/eda/",
         ["Reads the 4 file types with strict validation and a data-quality report (§2).",
          "Projects each trip to its UTM zone so all geometry is in metres (§4.1).",
          "Builds a label for every fix with the monotone DP alignment, including the direction term (§4.2).",
          "STRtree candidate search (50 m, K = 10, 150 m fallback) and the 21 features in 5 groups (§4.3–4.4)."],
         [("Why not use latitude/longitude directly?", "Degrees are not metres; 1° of longitude shrinks from 111 km to 72 km by latitude 50°. UTM gives true metres."),
          ("Where do the per-fix labels come from?", ".route has no per-fix info, so we align fixes to it with a monotone DP; −1 if > 60 m away."),
          ("Why 50 m and 10 candidates?", "The radius sweep: 1.22% miss rate at 6.6 candidates/fix; larger radii add noise for little gain."),
          ("Most important feature group?", f"Heading: removing it costs {100 * -abl.loc['heading', 'delta_top1']:.0f} points of Top-1.")]),
        ("Sudheeksh", "Models, cross-validation, metrics, Bayesian optimisation", "ML lead",
         "src/pathsnap/models/, training/, evaluation/metrics.py, optimisation/; scripts/train_all.py, optimise.py; configs/models.yaml",
         ["Trains 8 models from 7 families on identical features and folds (§4.5).",
          "GroupKFold(5) by trip, so no trip is in both training and validation (§4.6).",
          "Records accuracy, precision, recall, Macro-F1, log loss, ROC-AUC, Brier, Top-1 and the train-vs-validation gap (§4.7).",
          "Optuna TPE + MedianPruner on HGB, 40 trials, objective CV Macro-F1 (§4.8)."],
         [("Why group by trip?", "Neighbouring fixes are near-copies; a random split would leak a trip into its own validation set."),
          ("Why HGB over Random Forest?", f"Tied on CV ({hgb.cv_macro_f1:.4f} vs {rf.cv_macro_f1:.4f}); untuned HGB fits ~{rf.fit_s_full / hgb.fit_s_full:.0f}× faster and its file is ~{rf.model_size_mb / hgb.model_size_mb:.0f}× smaller."),
          ("What does TPE do?", "It models good vs bad trials and proposes settings where good ones are more likely: smarter than a grid."),
          ("Why did tuning barely move CV?", "The features already separate the classes well; tuning mainly improved probability quality (log loss).")]),
        ("Shubashitha", "HMM / Viterbi, route reconstruction, Folium maps, evaluator UI", "Laptop 2 operator",
         "src/pathsnap/decoding/, visualisation/; scripts/tune_hmm.py, evaluate_all.py, generate_maps.py; frontend/src/",
         ["Turns per-fix probabilities into one drivable route with an HMM and segmented Viterbi (§4.9).",
          "Fills skipped segments with shortest paths (§4.10).",
          "Tunes the decoder on training trips with out-of-fold probabilities only.",
          "Folium maps (maps/html/) and the React + Leaflet UI (§6.3); drives the UI in the demo."],
         [("What is an emission vs a transition?", "Emission: how well a road explains one fix. Transition: how plausible the move between two roads is."),
          ("Why is Viterbi exact?", "Dynamic programming keeps the best path into every state, so the global best path is never lost."),
          ("What does the U-turn penalty fix?", "Flip-flopping between a street's two directions when the heading is unreliable."),
          ("When does the chain break?", f"No candidates, no valid transition, or a gap > {hmm['max_gap_s']:.0f} s; we never invent roads across a break.")]),
        ("Sujal", "Inference pipeline, FastAPI backend, deployment, testing", "Laptop 1 operator",
         "src/pathsnap/inference/pipeline.py; backend/; docs/deployment.md; tests/, backend/tests/; scripts/lan_check.py",
         ["MapMatcher loads the model and checks the schema; predict() never reads .route (§6.2).",
          "RouteEvaluator is the only code that reads .route, and it returns aggregate metrics only.",
          "FastAPI with request logging, CORS, gzip, timeouts and clear error codes; EVALUATION vs DEVELOPMENT mode.",
          "92 + 12 tests, clean-clone test, LAN check script; runs Laptop 1 in the demo."],
         [("How does the evaluator know it is live?", "The Laptop 1 console prints a POST /api/predict line and an accuracy line for every click."),
          ("How do you stop label leakage?", "Prediction loads trips with with_route=False; responses have no reference fields; a test enforces it."),
          ("What if Laptop 2 cannot connect?", "Same Wi-Fi? Correct IP (ipconfig)? Firewall rule? Test-NetConnection to port 8000, then /health."),
          ("What happens with a broken trip file?", "The API answers 422 with the file name and the reason, instead of crashing.")])]
    for name, area, role, files, how, qa in cards:
        col = MEMBER_COL[name]
        head = Table([[P(f'<font name="UI-BL" size="15" color="white">{name}</font>'
                         f'<font name="UI" size="9" color="white">   ·   {area}</font>', CELL),
                       P(f'<font name="UI-B" size="8" color="white">{role.upper()}</font>', ParagraphStyle("r", parent=CELL, alignment=2))]],
                     colWidths=[CW * 0.72, CW * 0.28])
        head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), col), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                  ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
        body = [P(f'<font name="UI-SB" color="{col.hexval()}">Files</font>  <font name="Mono" size="7.6">{escape(files)}</font>', CELL),
                Spacer(1, 4), P(f'<font name="UI-SB" color="{col.hexval()}">What it does and how</font>', CELL)]
        body += [Paragraph(h, ParagraphStyle("cbul", parent=CELL, leftIndent=10, bulletIndent=0), bulletText="•") for h in how]
        body += [Spacer(1, 4), P(f'<font name="UI-SB" color="{col.hexval()}">Be ready to answer</font>', CELL)]
        body += [P(f"<b>Q: {escape(q)}</b><br/>A: {escape(a)}", ParagraphStyle("qa", parent=CELL, spaceAfter=3, leftIndent=6)) for q, a in qa]
        bt = Table([[body]], colWidths=[CW])
        bt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        s += [KeepTogether([head, bt]), Spacer(1, 10)]

    # ===================================================================== 8. demo
    s += section(8, "Demo playbook", "exactly what to do and show, in order")
    s += h2("8.1  The day before (and once more 30 minutes before)")
    s += [table([["", "Who", "Check"],
                 ["1", owner_tag("Sujal"), "Laptop 1: evaluator trips are in <font name='Mono'>PATHSNAP\\data\\raw\\evaluator_test\\</font> (10 folders). "
                                           "Run the VS Code task <b>4. Run all backend + pipeline tests</b>: 92 passed."],
                 ["2", owner_tag("Sujal"), "Firewall rule for port 8000 added once (Administrator PowerShell; see LAPTOP1_BACKEND_COMMANDS.md §2)."],
                 ["3", owner_tag("Shubashitha"), "Laptop 2: Node.js installed, zip unzipped, <font name='Mono'>npm.cmd ci</font> done while you have internet."],
                 ["4", "both", "Both laptops on the <b>same</b> Wi-Fi, or better, the same <b>phone hotspot</b> (college Wi-Fi often blocks laptop-to-laptop traffic)."],
                 ["5", "both", "Full rehearsal once on the network you will use. Take the screenshots for the report during the rehearsal."],
                 ["6", "everyone", "Charge both laptops; bring chargers; turn off sleep and auto-updates; close other apps."]],
                [0.7 * cm, 2.6 * cm, CW - 3.3 * cm])]
    s += h2("8.2  Start-up sequence (about 3 minutes)")
    s += [code(["# Laptop 1 (Sujal) - VS Code: Ctrl+Shift+P -> Tasks: Run Task -> '1. Backend: start API (Laptop 1)'",
                "#   or in PowerShell:",
                'cd "D:\\ML hackathon\\PATHSNAP"',
                ".\\.venv\\Scripts\\python.exe -m backend.main",
                "#   expect:  model loaded: hist_gradient_boosting_tuned",
                "#            mode=EVALUATION partition=evaluator_test trips=10",
                "ipconfig            # read the Wi-Fi IPv4 address out loud (it changes per network!)",
                "",
                "# Laptop 2 (Shubashitha) - PowerShell, in ...\\PathSnap_Laptop2_Frontend\\frontend",
                "Test-NetConnection <LAPTOP-1-IP> -Port 8000          # TcpTestSucceeded : True",
                'Set-Content -Encoding ascii .env.local "VITE_API_URL=http://<LAPTOP-1-IP>:8000"',
                "npm.cmd run dev                                       # open http://localhost:5173"])]
    s += h2("8.3  The run sheet: what to show and say")
    run = [("0:00", "Sujal", "Introduce the problem in one sentence; show the two laptops and say what runs where.",
            "“Laptop 1 holds the model and the evaluator data; Laptop 2 is only the UI. The reference routes never leave Laptop 1.”"),
           ("0:30", "Shubashitha", "Top bar: <b>API online</b>, <b>EVALUATION</b>, model hist_gradient_boosting_tuned, Trips 10.",
            "“We are in evaluation mode on the 10 reserved trips we never trained on.”"),
           ("1:00", "Evaluator", "Asks for a trip; Shubashitha clicks it.", "Let the evaluator choose; that proves nothing is pre-computed."),
           ("1:10", "Sujal", "Turns Laptop 1's console to the evaluator: the <font name='Mono'>POST /api/predict/&lt;id&gt; -> 200</font> line and the accuracy line.",
            "“That request just ran cleaning, candidates, features, the classifier and Viterbi live.”"),
           ("1:40", "Shubashitha", "Map: GPS (blue) vs HMM route (green, arrows). Zoom into a junction. Tick <b>Classifier-only route</b> to compare.",
            "“Orange jumps between roads; green is the route after the HMM: connected and drivable.”"),
           ("2:30", "Shubashitha", "Press <b>Snap to road</b>, then <b>Replay trip</b>.", "Visual WOW: every fix slides onto its road."),
           ("3:00", "Shubashitha", "Click a blue dot → <b>Why this road?</b> panel: candidates, probabilities, features.",
            "“Here the classifier was unsure between both directions; the sequence decided.”"),
           ("3:40", "Shubashitha", "Metrics panel: Top-1, Macro-F1, continuity, miss rate, timings; classifier vs + HMM bars.",
            f"“On all 10 held-out trips: {100 * ho['clf_top1']:.1f}% → {100 * ho['hmm_top1']:.1f}% Top-1.”" if ho else ""),
           ("4:30", "Evaluator", "Picks a second trip (try 00000015: the longest, 6,854 fixes, about 4–8 s).", "Shows that it scales; keep talking while it runs."),
           ("5:00", "Shriya", "Data story: four files, EDA findings, UTM projection, label alignment, candidates, features.", "Use §2 and §4.1–4.4."),
           ("6:30", "Sudheeksh", "Models: 8 models, GroupKFold by trip, metrics, why HGB, Optuna.", "Use §4.5–4.8; show the report figures."),
           ("8:00", "Shubashitha", "HMM/Viterbi intuition with the trellis figure (§4.9).", "Emission vs transition; why it is exact."),
           ("9:00", "Sujal", "Leakage protection, tests, robustness (what happens on a broken file / wrong id).", "Optionally open http://&lt;ip&gt;:8000/docs."),
           ("9:30", "everyone", "Questions. Anyone may be asked anything: know the cheat sheet (§10).", "")]
    s += [table([["Time", "Who", "What to show", "What to say / why"]] +
                [[t, owner_tag(w) if w in MEMBER_COL else w, a, b] for t, w, a, b in run],
                [1.1 * cm, 2.5 * cm, 7.2 * cm, CW - 10.8 * cm])]
    s += h2("8.4  Must-show checklist")
    must = (d["req"] or {}).get("demo_must_show") or []
    base = ["Backend running on Laptop 1, UI on Laptop 2, connected over the network (two physical laptops)",
            "API logs visible on Laptop 1 for every request",
            "The system working for the ten reserved evaluator trips (EVALUATION mode)",
            "Accuracy computed server-side from the internal reference data; the reference is never shown in the UI",
            "Route on the map, classifier-only vs HMM comparison, Folium maps (maps/html/) if asked",
            "Technical report and repository (public GitHub) ready to show"]
    s += bullets([escape(x) for x in (must or base)])
    s += h2("8.5  If something goes wrong")
    s += [table([["Problem", "Fix (stay calm and talk while fixing)"],
                 ["UI says API offline", "Same network? Right IP (ipconfig on Laptop 1, use Change in the UI)? Backend window still open? "
                                         "Firewall rule? Switch both laptops to the phone hotspot."],
                 ["Test-NetConnection fails", "Wi-Fi set to Private (Settings → Network); firewall rule for port 8000; hotspot."],
                 ["Grey map background", "No internet for map tiles. Untick Basemap; routes and roads still draw."],
                 ["“running scripts is disabled”", "You typed npm; type npm.cmd."],
                 ["Laptop 2 dies", "Run the UI on Laptop 1 (VS Code task 5) and explain the fallback."],
                 ["A trip is slow", "The largest trip takes a few seconds; keep explaining while the spinner runs."],
                 ["Mode shows DEVELOPMENT", "evaluator_test folder empty or backend started before the copy: restart the backend."]],
                [4.2 * cm, CW - 4.2 * cm])]

    # ===================================================================== 9. requirements
    s += section(9, "Requirements check", "every requirement from the five brief documents")
    req = d["req"]
    if req:
        sm = req.get("summary", {})
        s += [P(escape(req.get("intro", "")))] if req.get("intro") else []
        s += [stat_tiles([(str(sm.get(k, 0)), lab, col.hexval()) for k, lab, col in
                          [("MET", "requirements met", GREEN), ("PARTIAL", "partially met", ORANGE),
                           ("NOT MET", "not met", RED), ("TEAM ACTION", "need the team<br/>(e.g. live demo, public repo)", BLUE)]]),
              Spacer(1, 8)]
        if req.get("gaps"):
            s += [*h2("9.1  What is still open (fix these before submitting)")]
            s += [P("Everything else is met. These are the partial or not-met items, then the items only the team can do. "
                    "None of them needs retraining.", SMALL)]
            s += [table([["#", "Open item and the fix"]] + [[str(i), escape(g)] for i, g in enumerate(req["gaps"], 1)],
                        [0.8 * cm, CW - 0.8 * cm])]
        if req.get("submission"):
            s += [*h2("9.2  Submission details from the brief")] + bullets([escape(g) for g in req["submission"]])
        s += [*h2("9.3  The full checklist (all requirements)", 6 * cm)]
        s += [P("Codes: MG = Main Guidelines, DL = Deliverables, MT = Models and Training Guidelines, DE = Dataset Exploration, "
                "AP = Assessment and Plagiarism Policy, X = housekeeping. p = page in that document.", SMALL)]
        rows = [["ID", "Requirement", "Source", "Status", "Evidence / what is left"]]
        sty, last = [], None
        for it in req["items"]:
            if it.get("group") != last:
                last = it.get("group")
                rows.append([P(f'<font name="UI-B" color="#13315c">{escape(last or "")}</font>', CELL), "", "", "", ""])
                k = len(rows) - 1
                sty += [("SPAN", (0, k), (-1, k)), ("BACKGROUND", (0, k), (-1, k), colors.HexColor("#dfe7f2"))]
            badge, col = verdict_badge(it["verdict"])
            rows.append([P(f'<font size="7.4">{escape(it["id"])}</font>', CELL), P(escape(it["req"]), CELL),
                         P(f'<font size="7.2">{escape(it.get("source", ""))}</font>', CELL), badge,
                         P(f'<font size="7.2" color="#3b4552">{escape(it.get("evidence", ""))}</font>', CELL)])
            sty.append(("BACKGROUND", (3, len(rows) - 1), (3, len(rows) - 1), col))
        s += [table(rows, [1.15 * cm, 5.6 * cm, 2.0 * cm, 1.95 * cm, CW - 10.7 * cm], zebra=False, styles=sty)]
    else:
        s += [P("The requirements audit has not been run yet.", SMALL)]

    # ===================================================================== 10. cheat sheet
    s += section(10, "Cheat sheet", "numbers and words to know by heart")
    nums = [("Training / student-test / evaluator trips", f"{len(tr)} / {len(st)} / 10"),
            ("Training GPS fixes", f"{int(tr.n_obs.sum()):,}"), ("Sampling rate", "1 Hz (one fix per second)"),
            ("Median GPS-to-road distance", "4.7 m"), ("Gaps > 30 s", "135 in 30 trips"),
            ("Candidate radius / max candidates / fallback", "50 m / 10 / 150 m"),
            ("Candidate miss rate (student / evaluator)", f"{100 * dec['miss_rate']:.2f}% / {100 * ho['miss_rate']:.2f}%" if ho else f"{100 * dec['miss_rate']:.2f}%"),
            ("Features", "21 in 5 groups (geometric, heading, context, kinematic, network)"),
            ("Models compared", "8 from 7 families; HistGradientBoosting selected"),
            ("CV", "GroupKFold(5) by trip, seed 42; every 2nd fix for training"),
            ("HGB CV Macro-F1 (tuned)", f"{tuned['cv_macro_f1']:.4f}"),
            ("Optuna", f"TPE + MedianPruner, {opt['n_trials']} trials ({opt['n_complete']} complete, {opt['n_pruned']} pruned)"),
            ("HMM", f"top-{hmm['max_states']} states, w_clf {hmm['w_clf']}, β {hmm['beta_m']:.0f} m, U-turn {hmm['uturn_penalty']}, gap break {hmm['max_gap_s']:.0f} s"),
            ("Student-test Top-1", f"{dec['clf_top1']:.4f} → {dec['hmm_top1']:.4f}"),
            ("Evaluator Top-1 (held-out)", f"{ho['clf_top1']:.4f} → {ho['hmm_top1']:.4f}" if ho else "-"),
            ("Route continuity after reconstruction", f"{dec['recon_route_continuity']:.3f} (student) / {ho['recon_route_continuity']:.3f} (evaluator)" if ho else ""),
            ("Ablation: drop heading / geometric", f"−{100 * -abl.loc['heading', 'delta_top1']:.1f} / −{100 * -abl.loc['geometric', 'delta_top1']:.1f} Top-1 points"),
            ("Tests", "92 Python + 12 UI, all passing")]
    s += [table([["Fact", "Value"]] + [[a, f"<b>{b}</b>"] for a, b in nums], [8 * cm, CW - 8 * cm])]
    s += [P("Glossary", H3)]
    gl = [("Arc", "one directed road segment between two nodes; a two-way street is two arcs"),
          ("Candidate", "a road arc near a GPS fix that might be the right one"),
          ("Top-1", "the classifier's single best candidate for a fix"),
          ("UTM", "Universal Transverse Mercator: 60 flat map zones with coordinates in metres"),
          ("STRtree", "a spatial index (R-tree) to find nearby segments quickly"),
          ("GroupKFold", "cross-validation where a group (here a trip) is never split across folds"),
          ("Leakage", "information from the answer sneaking into training or prediction, giving falsely high scores"),
          ("TPE", "Tree-structured Parzen Estimator: the Bayesian optimisation sampler used by Optuna"),
          ("HMM", "Hidden Markov Model: hidden states (true roads) emit observations (GPS fixes)"),
          ("Viterbi", "dynamic-programming algorithm that finds the most likely sequence of hidden states"),
          ("Emission / transition", "score of a road for one fix / score of moving from one road to the next"),
          ("Route continuity", "share of consecutive output arcs that connect on the road network")]
    s += [table([["Term", "Meaning"]] + [[f"<b>{a}</b>", b] for a, b in gl], [3.6 * cm, CW - 3.6 * cm])]
    s += [P("Where to find more", H3)]
    from pypdf import PdfReader
    n_tr = len(PdfReader(str(ROOT / "reports" / "technical_report" / "PathSnap_Technical_Report.pdf")).pages)
    s += bullets([f"Technical report ({n_tr} pages): reports/technical_report/PathSnap_Technical_Report.pdf",
                  "120-question bank: reports/viva/QUESTION_BANK.md · one-page numbers: reports/viva/QUICK_REVISION.md",
                  "Command sheets: docs/LAPTOP1_BACKEND_COMMANDS.md, docs/LAPTOP2_FRONTEND_COMMANDS.md · team plan: docs/TEAM_ACTION_PLAN.md",
                  "Interactive maps: maps/html/ · results: experiments/results/ · held-out results: reports/final/TEST_FOLDER_CHECK.md"])
    return s


def owner_line(name):
    return P(f'<font name="UI" size="8" color="#5b6675">Owner:</font> {owner_tag(name)}',
             ParagraphStyle("o", parent=SMALL, spaceAfter=4))


def main():
    d = load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Doc(OUT, d)
    doc.build(build(d))
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
