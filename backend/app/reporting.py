from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether, Flowable
)
from reportlab.pdfbase.pdfmetrics import stringWidth


ORANGE = colors.HexColor("#F47735")
DARK = colors.HexColor("#17191C")
INK = colors.HexColor("#25292D")
MUTED = colors.HexColor("#6E747A")
PALE = colors.HexColor("#F3F4F5")
PALE_ORANGE = colors.HexColor("#FFF0E7")
GRID = colors.HexColor("#D9DDE0")
WHITE = colors.white


class MetricBars(Flowable):
    """Compact vector chart; values are read directly from the saved run."""
    def __init__(self, labels, values, title, suffix="%"):
        super().__init__()
        self.labels = labels
        self.values = values
        self.title = title
        self.suffix = suffix
        self.width = 174 * mm
        self.height = 45 * mm

    def draw(self):
        c = self.canv
        c.setFillColor(DARK)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(0, self.height - 5 * mm, self.title)
        chart_top = self.height - 11 * mm
        chart_bottom = 8 * mm
        chart_h = chart_top - chart_bottom
        maxv = max([float(v or 0) for v in self.values] + [1])
        maxv = max(100 if self.suffix == "%" else maxv, maxv)
        left = 4 * mm
        usable = self.width - 8 * mm
        gap = 5 * mm
        bw = (usable - gap * (len(self.values) - 1)) / max(1, len(self.values))
        for i, (label, value) in enumerate(zip(self.labels, self.values)):
            x = left + i * (bw + gap)
            h = chart_h * max(0, float(value or 0)) / maxv
            c.setFillColor(PALE)
            c.roundRect(x, chart_bottom, bw, chart_h, 2 * mm, fill=1, stroke=0)
            c.setFillColor(ORANGE if i == len(self.values) - 1 else colors.HexColor("#9CA3A8"))
            c.roundRect(x, chart_bottom, bw, h, 2 * mm, fill=1, stroke=0)
            c.setFillColor(DARK)
            c.setFont("Helvetica-Bold", 8)
            c.drawCentredString(x + bw / 2, min(chart_bottom + h + 2 * mm, chart_top + 1 * mm), f"{float(value or 0):.2f}{self.suffix}")
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 7)
            c.drawCentredString(x + bw / 2, 2 * mm, label)


class ConfusionMatrixGraphic(Flowable):
    """A compact, readable vector heatmap confusion matrix (not a table)."""
    def __init__(self, labels, values, width=174 * mm, max_height=82 * mm):
        super().__init__()
        self.labels = [str(x) for x in labels]
        self.values = values or []
        self.width = width
        n = max(1, len(self.labels))
        # Size the flowable to its true drawing bounds. The previous fixed
        # max-height clipped the layout reservation for 10-class matrices,
        # allowing subsequent matrices to overlap them.
        self.cell = min(7.0 * mm, (width - 42 * mm) / (n + 1), 58 * mm / (n + 1))
        self.height = (n + 1) * self.cell + 16 * mm

    def draw(self):
        c = self.canv
        labels, values = self.labels, self.values
        n = min(len(labels), len(values))
        if not n:
            return
        cell = self.cell
        left = 36 * mm
        bottom = 8 * mm
        top = self.height - 5 * mm
        maxv = max([float(v) for row in values[:n] for v in row[:n]] + [1.0])
        c.saveState()
        c.setFillColor(MUTED); c.setFont('Helvetica-Bold', 5.8)
        c.drawString(left, top + 1 * mm, 'PREDICTED →')
        c.saveState(); c.translate(5 * mm, bottom + (n * cell) / 2); c.rotate(90)
        c.drawCentredString(0, 0, 'ACTUAL →'); c.restoreState()
        for j, label in enumerate(labels[:n]):
            x = left + j * cell + cell / 2
            c.setFillColor(INK); c.setFont('Helvetica', 5.5)
            c.saveState(); c.translate(x, bottom + n * cell + 2 * mm); c.rotate(35)
            c.drawString(0, 0, label[:18]); c.restoreState()
        for i, label in enumerate(labels[:n]):
            y = bottom + (n - 1 - i) * cell
            c.setFillColor(INK); c.setFont('Helvetica', 5.5)
            c.drawRightString(left - 3 * mm, y + cell * .42, label[:20])
            for j in range(n):
                v = float(values[i][j]) if j < len(values[i]) else 0.0
                intensity = min(1.0, v / maxv)
                if i == j:
                    base = colors.HexColor('#F47735')
                    fill = colors.Color(1, .91 - .28 * intensity, .84 - .48 * intensity)
                else:
                    fill = colors.Color(.96, .95 - .12 * intensity, .93 - .2 * intensity)
                x = left + j * cell
                c.setFillColor(fill); c.setStrokeColor(colors.HexColor('#E3D8D0'))
                c.roundRect(x + .5, y + .5, cell - 1, cell - 1, 1.5 * mm, fill=1, stroke=1)
                c.setFillColor(INK if intensity < .58 else DARK)
                c.setFont('Helvetica-Bold', 5.8 if n <= 6 else 5.2)
                c.drawCentredString(x + cell / 2, y + cell / 2 - 2, f'{int(v):,}')
        c.restoreState()


def _logo(canvas, x, y, scale=1.0):
    """Draw the geometric SENTINEL mark as vector paths."""
    canvas.saveState()
    canvas.translate(x, y)
    canvas.scale(scale, scale)
    canvas.setStrokeColor(ORANGE)
    canvas.setLineWidth(2.5)
    p = canvas.beginPath()
    p.moveTo(30, 39); p.lineTo(14, 39); p.lineTo(5, 30); p.lineTo(5, 22)
    p.lineTo(13, 14); p.lineTo(28, 14); p.lineTo(34, 8); p.lineTo(34, 4)
    p.lineTo(28, -2); p.lineTo(10, -2)
    canvas.drawPath(p)
    canvas.setLineWidth(1.1)
    canvas.setStrokeColor(colors.HexColor("#F9A06D"))
    p2 = canvas.beginPath()
    p2.moveTo(12, 39); p2.lineTo(4, 31); p2.lineTo(4, 23); p2.lineTo(12, 15)
    canvas.drawPath(p2)
    canvas.restoreState()


def _page_brand(canvas, document):
    canvas.saveState()
    w, h = A4

    # Warm paper tint: near-white with a restrained orange header wash, not an orange page.
    canvas.setFillColor(colors.HexColor("#FFFCF9"))
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#FFF2E9"))
    canvas.rect(0, h - 29 * mm, w, 29 * mm, fill=1, stroke=0)
    canvas.setFillColor(ORANGE)
    canvas.rect(0, 0, 2.2 * mm, h, fill=1, stroke=0)

    # Low-opacity diagonal watermark behind the page content.
    canvas.saveState()
    canvas.translate(w / 2, h / 2)
    canvas.rotate(32)
    canvas.setFillColor(ORANGE)
    try:
        canvas.setFillAlpha(0.055)
    except Exception:
        pass
    canvas.setFont("Helvetica-Bold", 48)
    canvas.drawCentredString(0, 0, "SENTINEL")
    try:
        canvas.setFillAlpha(1)
    except Exception:
        pass
    canvas.restoreState()

    _logo(canvas, 17 * mm, h - 20 * mm, 0.55)
    canvas.setFillColor(DARK)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(29 * mm, h - 15.8 * mm, "SENTINEL")
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawString(29 * mm, h - 20 * mm, "ADAPTIVE NETWORK INTELLIGENCE")
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.55)
    canvas.line(17 * mm, h - 27 * mm, w - 17 * mm, h - 27 * mm)
    canvas.line(17 * mm, 15 * mm, w - 17 * mm, 15 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawString(17 * mm, 10 * mm, "MODEL RUN · HELD-OUT TEST EVALUATION")
    canvas.drawRightString(w - 17 * mm, 10 * mm, f"{document.page:02d}")
    canvas.restoreState()


def build_pdf_report(model: dict) -> bytes:
    """Build a downloadable report for the currently persisted multiclass model run only."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=30 * mm, bottomMargin=19 * mm,
        title=f"SENTINEL Model Test Report — {model.get('dataset', 'Dataset')}",
        author="SENTINEL — Adaptive Network Intelligence"
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="BrandTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=27, leading=30, textColor=DARK, alignment=TA_LEFT, spaceAfter=2
    ))
    styles.add(ParagraphStyle(
        name="BrandSub", parent=styles["Normal"], fontName="Helvetica",
        fontSize=7, leading=10, textColor=MUTED, tracking=2, spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name="EyebrowX", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=7.5, leading=10, textColor=ORANGE, spaceBefore=3, spaceAfter=5
    ))
    styles.add(ParagraphStyle(
        name="SectionX", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=14, leading=17, textColor=DARK, spaceBefore=3, spaceAfter=5
    ))
    styles.add(ParagraphStyle(
        name="SubSectionX", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=9, leading=11, textColor=INK, spaceBefore=5, spaceAfter=3
    ))
    styles.add(ParagraphStyle(
        name="BodyX", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=7.6, leading=10, textColor=INK, spaceAfter=3
    ))
    styles.add(ParagraphStyle(
        name="SmallX", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=6.5, leading=8, textColor=MUTED, spaceAfter=2
    ))
    styles.add(ParagraphStyle(
        name="TinyFeatureX", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=5.7, leading=6.8, textColor=INK, spaceAfter=0
    ))
    styles.add(ParagraphStyle(
        name="MetricValueX", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=17, leading=19, textColor=DARK, alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name="MetricLabelX", parent=styles["Normal"], fontName="Helvetica",
        fontSize=6.5, leading=8, textColor=MUTED, alignment=TA_CENTER
    ))

    def P(text, style="BodyX"):
        return Paragraph(str(text), styles[style])

    def fmt_pct(value):
        return "—" if value is None else f"{float(value) * 100:.2f}%"

    def fmt_num(value, digits=3):
        return "—" if value is None else f"{float(value):.{digits}f}"

    def styled_table(rows, widths, font=7.0, highlight_col=None, repeat=1):
        t = Table(rows, colWidths=widths, repeatRows=repeat)
        commands = [
            ("GRID", (0, 0), (-1, -1), 0.35, GRID),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("FONTSIZE", (0, 0), (-1, -1), font),
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
        if len(rows) > 1:
            commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]))
        if highlight_col is not None:
            commands += [
                ("BACKGROUND", (highlight_col, 1), (highlight_col, -1), PALE_ORANGE),
                ("FONTNAME", (highlight_col, 1), (highlight_col, -1), "Helvetica-Bold"),
            ]
        t.setStyle(TableStyle(commands))
        return t

    def metric_strip(items):
        cells = []
        for value, label in items:
            cells.append([P(value, "MetricValueX"), P(label, "MetricLabelX")])
        t = Table([cells], colWidths=[174 * mm / len(cells)] * len(cells))
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PALE),
            ("BOX", (0, 0), (-1, -1), 0.5, GRID),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        return t

    keys = ["all", "fixed", "dynamic"]
    names = {"all": "A · All features", "fixed": "B · Fixed subset", "dynamic": "C · Dynamic selection"}
    results = model.get("results") or {}
    labels = model.get("labels") or results.get("dynamic", {}).get("labels") or []
    feature_names = model.get("feature_names") or model.get("feature_columns") or []
    feature_total = len(feature_names) if feature_names else "—"
    dynamic = results.get("dynamic", {})
    dynamic_features = dynamic.get("feature_count", "—")

    story = [
        Spacer(1, 8 * mm),
        P("SENTINEL", "BrandTitle"),
        P("ADAPTIVE NETWORK INTELLIGENCE", "BrandSub"),
        P("MODEL RUN / EXPERIMENT REPORT", "EyebrowX"),
        P(f"{model.get('dataset', 'Dataset')} · {model.get('kind', 'Classifier')}", "SectionX"),
        P("Measured training and held-out test results from the model currently saved in this SENTINEL installation.", "BodyX"),
        Spacer(1, 3 * mm),
        metric_strip([
            (str(model.get("kind", "—")), "CLASSIFIER"),
            (str(model.get("seed", "—")), "RANDOM SEED"),
            (f"{int(model.get('test_records', 0)):,}", "HELD-OUT TEST ROWS"),
            (str(feature_total), "INPUT FEATURES"),
        ]),
        Spacer(1, 5 * mm),
        P("RUN DETAILS", "EyebrowX"),
    ]

    run_rows = [
        ["Run field", "Saved value"],
        ["Dataset", model.get("dataset", "—")],
        ["Task", "Multiclass intrusion-category classification"],
        ["Model / classifier", model.get("kind", "—")],
        ["Training timestamp", model.get("created_at", "Not recorded")],
        ["Random seed", model.get("seed", "—")],
        ["Evaluation protocol", model.get("split", "Provided / official test file")],
        ["Training records", f"{int(model.get('train_records', 0)):,}"],
        ["Held-out test records", f"{int(model.get('test_records', 0)):,}"],
        ["Total dataset records", f"{int(model.get('records', 0)):,}"],
        ["Input feature columns", str(feature_total)],
        ["Predicted classes", ", ".join(labels) if labels else "Not recorded"],
    ]
    story += [
        styled_table(run_rows, [48 * mm, 126 * mm], font=7.2),
        Spacer(1, 5 * mm),
        P("TRAINING & EVALUATION SUMMARY", "EyebrowX"),
    ]

    metric_rows = [["Metric", "A · All", "B · Fixed", "C · Dynamic"]]
    metric_specs = [
        ("Accuracy", "accuracy", "pct"),
        ("Macro precision", "macro_precision", "pct"),
        ("Macro recall", "macro_recall", "pct"),
        ("Macro F1", "macro_f1", "pct"),
        ("Weighted F1", "weighted_f1", "pct"),
        ("Training duration (s)", "training_seconds", "num"),
        ("Test inference (s)", "inference_seconds", "num"),
        ("Average features used", "feature_count", "num"),
    ]
    for label, key, kind in metric_specs:
        row = [label]
        for approach in keys:
            value = results.get(approach, {}).get(key)
            row.append(fmt_pct(value) if kind == "pct" else fmt_num(value, 4))
        metric_rows.append(row)
    story += [
        styled_table(metric_rows, [55 * mm, 39 * mm, 39 * mm, 41 * mm], font=7.0, highlight_col=3),
        Spacer(1, 4 * mm),
        MetricBars(
            ["All", "Fixed", "Dynamic"],
            [results.get(k, {}).get("accuracy", 0) * 100 for k in keys],
            "HELD-OUT TEST ACCURACY"
        ),
        P("All three approaches are evaluated on the same held-out test records for this saved run. The comparison is specific to the dataset, split, classifier and random seed shown above.", "SmallX"),
        Spacer(1, 2 * mm),
    ]

    # Per-class report for each approach, not just the dynamic approach.
    story += [
        P("CLASS-WISE TEST RESULTS", "EyebrowX"),
        P("Precision, recall and F1 by predicted attack family", "SectionX"),
        P("Support is the number of held-out test records belonging to each actual class.", "SmallX"),
    ]
    for approach in keys:
        result = results.get(approach, {})
        per_class = result.get("per_class", {})
        approach_rows = [["Class", "Precision", "Recall", "F1-score", "Support"]]
        for label in labels:
            item = per_class.get(label, {})
            approach_rows.append([
                label, fmt_pct(item.get("precision")), fmt_pct(item.get("recall")),
                fmt_pct(item.get("f1-score")), f"{int(item.get('support', 0)):,}"
            ])
        story += [
            P(names[approach], "SubSectionX"),
            styled_table(approach_rows, [42 * mm, 32 * mm, 32 * mm, 32 * mm, 36 * mm], font=6.8),
            Spacer(1, 3 * mm),
        ]
    story += [PageBreak(), P("CONFUSION MATRICES", "EyebrowX"),
              P("Held-out test predictions", "SectionX"),
              P("Rows represent actual classes; columns represent model predictions. Diagonal values are correct predictions; off-diagonal values are misclassifications.", "SmallX")]

    for matrix_index, approach in enumerate(keys):
        if matrix_index:
            story.append(PageBreak())
        result = results.get(approach, {})
        matrix_values = result.get("confusion_matrix") or []
        matrix_labels = result.get("labels") or labels
        if not matrix_values or not matrix_labels:
            continue
        story += [
            P(names[approach], "SubSectionX"),
            ConfusionMatrixGraphic(matrix_labels, matrix_values),
            Spacer(1, 3 * mm),
        ]
    # Feature audit begins separately so it cannot be squeezed below the final heatmap.
    story += [PageBreak(),
        P("FEATURE SELECTION RECORD", "EyebrowX"),
        P("What the trained approaches used", "SectionX"),
        P("Feature selection information below is taken from the saved model artifact. Feature importance is a model-derived relevance score, not a causal explanation.", "BodyX"),
        P("Dynamic architecture: a compact multiclass router predicts a family; that family’s specialist then distinguishes Normal traffic from that family using its own training-selected feature subset. Test labels are never used for routing or feature selection.", "SmallX"),
    ]
    fixed_features = model.get("fixed_features") or model.get("selected_features") or []
    router_features = model.get("router_features") or []
    family_features = model.get("family_features") or {}
    if fixed_features:
        story += [P(f"Fixed subset · {len(fixed_features)} features", "SubSectionX"),
                  P(", ".join(map(str, fixed_features)), "SmallX")]
    if router_features:
        story += [P(f"Dynamic router · {len(router_features)} features", "SubSectionX"),
                  P(", ".join(map(str, router_features)), "SmallX")]
    if family_features:
        story += [P("Dynamic family experts", "SubSectionX")]
        expert_rows = [[P("Predicted family", "SmallX"), P("Expert feature subset", "SmallX")]]
        for family, selected in family_features.items():
            # Paragraph wrapping keeps long feature lists inside the printable column.
            expert_rows.append([P(str(family), "SmallX"), P(", ".join(map(str, selected)), "TinyFeatureX")])
        expert_table = Table(expert_rows, colWidths=[34 * mm, 140 * mm], repeatRows=1, hAlign="LEFT")
        expert_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PALE_ORANGE),
            ("GRID", (0, 0), (-1, -1), .3, GRID),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(expert_table)
    importance = model.get("global_importance") or {}
    if importance:
        story += [Spacer(1, 4 * mm), P("Top training-derived global feature importance", "SubSectionX")]
        imp_rows = [["Feature", "Importance"]]
        for feature, score in list(importance.items())[:8]:
            imp_rows.append([str(feature), f"{float(score):.6f}"])
        story.append(styled_table(imp_rows, [125 * mm, 49 * mm], font=6.8))
    story += [
        Spacer(1, 5 * mm),
        P("RUN NOTES", "EyebrowX"),
        P("This document is generated from the currently saved model artifact selected in the Model Benchmark page. It reports the model type, run timestamp, data split, all/fixed/dynamic test results, class-wise metrics, confusion matrices and saved feature-selection details.", "BodyX"),
        P("If a field was not saved by the training pipeline, it is shown as unavailable rather than inferred or fabricated.", "SmallX"),
    ]

    doc.build(story, onFirstPage=_page_brand, onLaterPages=_page_brand)
    return buf.getvalue()


def build_binary_pdf_report(report: dict) -> bytes:
    """Build an independent UNSW-NB15 binary LightGBM report from saved metrics."""
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import mm
    from io import BytesIO
    from reportlab.lib import colors
    metrics = report.get('test_metrics', {})
    labels = metrics.get('labels', ['Normal', 'Attack'])
    matrix = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=25*mm, bottomMargin=20*mm, title='SENTINEL UNSW-NB15 Binary Model Test Report', author='SENTINEL')
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BinaryTitle', parent=styles['Title'], textColor=DARK, fontSize=23, leading=27, alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle(name='BinarySub', parent=styles['Normal'], textColor=MUTED, fontSize=9, leading=13, spaceAfter=12))
    styles.add(ParagraphStyle(name='BinarySection', parent=styles['Heading2'], textColor=ORANGE, fontSize=13, leading=16, spaceBefore=12, spaceAfter=7))
    story = [Paragraph('SENTINEL', styles['Heading2']), Paragraph('UNSW-NB15 Binary Intrusion Detection', styles['BinaryTitle']), Paragraph('Normal vs Attack · LightGBM · Official held-out test evaluation', styles['BinarySub'])]
    story += [Paragraph('Measured test metrics', styles['BinarySection'])]
    metric_rows = [['Metric', 'Measured value'], ['Accuracy', f"{metrics.get('accuracy',0)*100:.2f}%"], ['Macro precision', f"{metrics.get('macro_precision',0)*100:.2f}%"], ['Macro recall', f"{metrics.get('macro_recall',0)*100:.2f}%"], ['Macro F1', f"{metrics.get('macro_f1',0)*100:.2f}%"], ['Weighted F1', f"{metrics.get('weighted_f1',0)*100:.2f}%"], ['Inference / record', f"{metrics.get('inference_ms_per_record',0):.4f} ms"], ['Training + preprocessing', f"{metrics.get('training_seconds_including_preprocessing',0):.2f} s"]]
    t=Table(metric_rows, colWidths=[85*mm,75*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),DARK),('TEXTCOLOR',(0,0),(-1,0),colors.white),('TEXTCOLOR',(0,1),(-1,-1),INK),('GRID',(0,0),(-1,-1),.4,GRID),('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,PALE]),('PADDING',(0,0),(-1,-1),8)])); story.append(t)
    story += [Paragraph('Confusion matrix', styles['BinarySection']), Paragraph('Rows are actual classes; columns are predicted classes.', styles['BodyText'])]
    story.append(ConfusionMatrixGraphic(labels, matrix, width=174*mm, max_height=62*mm))
    story += [Paragraph('Per-class performance', styles['BinarySection'])]
    per=metrics.get('per_class',{})
    rows=[['Class','Precision','Recall','F1','Support']]
    for label in labels:
        v=per.get(label,{})
        rows.append([label, f"{v.get('precision',0)*100:.2f}%", f"{v.get('recall',0)*100:.2f}%", f"{v.get('f1-score',0)*100:.2f}%", f"{int(v.get('support',0)):,}"])
    pt=Table(rows,colWidths=[40*mm,30*mm,30*mm,30*mm,30*mm]); pt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),DARK),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.4,GRID),('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,PALE]),('PADDING',(0,0),(-1,-1),7)])); story.append(pt)
    story += [Spacer(1,10), Paragraph(f"Training records: {int(report.get('train_records',0)):,} · Test records: {int(report.get('test_records',0)):,} · Features: {int(report.get('feature_count',0))}", styles['BodyText']), Spacer(1,8), Paragraph('Scope: This is a binary UNSW-NB15 experiment using the official label field. It is separate from native UNSW-NB15 attack-category classification and from NSL-KDD five-family classification. Metrics are taken from the saved held-out test evaluation.', styles['BodyText'])]
    doc.build(story, onFirstPage=_page_brand, onLaterPages=_page_brand)
    return buf.getvalue()
