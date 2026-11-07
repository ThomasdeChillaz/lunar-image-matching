"""Render the research handout from the same checked evidence as the CLI."""
from pathlib import Path
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lunar_matching.evidence import LABELS, METHODS, load_evidence


def render_comparison(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Direct", "Single, no extra", "Previous active +32", "Single, full cap",
              "Beam, uniform", "Beam, geometry", "Beam, active"]
    colors = ["#8A9BA7"] * 4 + ["#4D8CAA", "#4D8CAA", "#087C75"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained", sharey=True)
    for axis, values, title, limit in (
        (axes[0], [result["methods"][m]["correct"] for m in METHODS], "Correct raw poses /4", 4.65),
        (axes[1], [result["methods"][m]["median_positive_online_seconds"] for m in METHODS], "Median positive seconds", 66)):
        axis.barh(range(7), values, color=colors)
        for index, value in enumerate(values):
            axis.text(value + limit * .025, index, str(int(value)) + "/4" if axis is axes[0] else f"{value:.1f}s",
                      va="center", fontsize=9)
        axis.set(xlim=(0, limit), title=title)
        axis.tick_params(labelsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(7), labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xticks(range(5))
    image = ROOT / "docs/figures/pose-runtime.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image, dpi=200)
    plt.close(fig)
    return image


def build(destination):
    result = load_evidence()
    navy, teal, gray = colors.HexColor("#152C3D"), colors.HexColor("#087C75"), colors.HexColor("#536674")
    body = ParagraphStyle("BriefBody", fontName="Helvetica", fontSize=9.3, leading=13.5,
                          textColor=navy, spaceAfter=7)
    title = ParagraphStyle("BriefTitle", parent=body, fontName="Helvetica-Bold", fontSize=22, leading=26, spaceAfter=10)
    section = ParagraphStyle("BriefSection", parent=body, fontName="Helvetica-Bold", fontSize=11,
                             leading=15, textColor=teal, spaceBefore=9, spaceAfter=6)
    small = ParagraphStyle("BriefSmall", parent=body, fontSize=8, leading=11, textColor=gray)
    width = A4[0] - 88
    story = []

    def paragraph(text, style=body):
        story.append(Paragraph(text, style))

    paragraph("INDEPENDENT RESEARCH PROTOTYPE  |  7 NOVEMBER 2026", small)
    paragraph("Lunar camera localization<br/>under changing illumination", title)
    paragraph("Thomas de Chillaz", section)
    paragraph("<b>Question.</b> Does retaining multiple camera and lighting hypotheses during native patch selection "
              "improve localization over selecting one hypothesis early?")
    paragraph("Experimental design", section)
    paragraph("I implemented a persistent set of up to eight camera/light branches, reference-only patch-disagreement "
              "selection, native measurements and sequential camera updates. Seven methods share the same eight cases. "
              "The new single-camera and beam controls share 128 total branch-specific field attempts, a reference "
              "candidate universe and the final AP3P/RANSAC estimator. Eight branches receive 16 rounds each.")
    paragraph("Inference receives reference geometry and imagery, camera intrinsics and a 24-light reference catalogue. "
              "Query true pose, depth and Sun are scoring labels only. All predictions are sealed before scoring.")
    paragraph("Measured development result", section)
    table_rows = [["Method", "Correct\nraw /4", "Wrong\npositive", "Nonoverlap\n/4", "Median\nseconds"]]
    for name, label in zip(METHODS, LABELS, strict=True):
        row = result["methods"][name]
        table_rows.append([label, str(row["correct"]), str(row["wrong_positive"]), str(row["raw_nonoverlap"]),
                           f"{row['median_positive_online_seconds']:.2f}"])
    table = Table(table_rows, colWidths=[218, 61, 69, 78, width-426], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("LEADING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F0F5F6"), colors.white]),
        ("LINEBELOW", (0, 0), (-1, 0), .8, teal)]))
    story.append(table)
    story.append(Spacer(1, 9))
    paragraph("<b>No recovery gain; no candidate promoted.</b> Every method recovers the same positive. Active beam "
              "takes 54.52s versus 21.77s for the existing no-extra control. It adds zero trainable parameters.")
    paragraph("Correct pose: rotation error &lt;=2 degrees and translation &lt;=1% of median Euclidean reference range. "
              "All eligible positives remain in the denominator. These are raw poses, not calibrated acceptances.", small)
    paragraph("Scope and reliability", section)
    paragraph("Four positives cover three already exposed terrain components; four nonoverlaps record both endpoints. "
              "All 56 method/case cells and 24 historical control identities independently replay. This small, shared-renderer "
              "study does not establish fresh-terrain performance, a 1% deployment risk, modern-model superiority or real "
              "orbital navigation accuracy. Calibration and final terrain remain unused.")
    story.append(PageBreak())
    paragraph("FINDINGS AND NEXT RESEARCH QUESTIONS", small)
    paragraph("Where the pipeline loses accuracy", title)
    paragraph("1. Lighting coverage and camera coverage are different", section)
    paragraph("The faint case now retains its correct catalogue illumination. Its closest retained initial camera is still "
              "364.51m and 12.11 degrees from truth. Native acquisition produces no correct final branch. A better "
              "camera proposal process needs to be tested separately from simply retaining more lights.")
    paragraph("2. Precise-looking matches can still yield inaccurate pose", section)
    paragraph("The strong single control produces 123 correspondences within two native pixels over the positive roster. "
              "Two final translations still miss the pose criterion: 17.42m versus a 13.13m limit, and 31.29m versus 15.88m. "
              "Its saved camera-update trajectory visits accurate intermediate states on those cases, but the final discrete "
              "matches and AP3P output do not preserve them. Oracle-selected intermediate states are not output successes.")
    image = render_comparison(result)
    story.append(Image(str(image), width=width, height=width*3.5/8))
    paragraph("Same-case, same-CPU development comparison. Published latency belongs to the frozen experiment; "
              "the new interface's validation and process startup are not included in these historical measurements.", small)
    paragraph("What I would investigate next", section)
    paragraph("Test camera/illumination search that preserves geometric coverage; allocate native measurements by their "
              "ability to resolve pose ambiguity; and study correspondence uncertainty and spatial conditioning at the final "
              "geometry handoff. Compare each mechanism against the strongest simple control at matched cost, then "
              "evaluate on genuinely fresh terrain. These are proposed experiments, not measured improvements or novelty claims.")
    paragraph('<b>Code and research summary:</b> <link href="https://github.com/ThomasdeChillaz/lunar-image-matching" '
              'color="#087C75">github.com/ThomasdeChillaz/lunar-image-matching</link><br/>'
              '<b>Reproducible measured artifacts:</b> <link href="https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-beam-v144-development" '
              'color="#087C75">persistent-hypothesis development release</link>', small)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CFDADD"))
        canvas.line(44, 36, A4[0]-44, 36)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(gray)
        canvas.drawString(44, 24, "Thomas de Chillaz | Research prototype and measured limitations")
        canvas.drawRightString(A4[0]-44, 24, str(document.page))
        canvas.restoreState()

    destination.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=44, leftMargin=44,
                      topMargin=39, bottomMargin=47, title="Lunar localization: research brief",
                      author="Thomas de Chillaz").build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build(ROOT / "output/pdf/lunar-localization-research-brief.pdf")
