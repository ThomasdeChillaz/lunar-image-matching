"""Presentation-only figures and PDF drafts from sealed v137 predictions.

No matching, model training, camera fitting, calibration or new image acquisition.
"""
from pathlib import Path
import json
import sys
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    KeepTogether,
)

ROOT = Path("reports/solar-active-v137")
RUN = Path("runs/solar-active-v137/comparison")
OUT = Path("output/pdf")
NAMES = ["direct", "joint_no_extra", "uniform", "appearance", "geometry", "active"]
LABELS = ["Direct", "Joint / no extra", "Uniform +32", "Appearance +32", "Geometry +32", "Active +32"]
BLUE, TEAL, RED, GREY = "#213f5b", "#087f8c", "#c25845", "#e5eaf0"
URL = "https://github.com/ThomasdeChillaz/lunar-image-matching"


def read(p):
    return json.loads(Path(p).read_text())


def figures(result, scores):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.labelcolor": BLUE,
        "text.color": BLUE, "axes.titleweight": "bold", "savefig.dpi": 180})
    stats = [result["methods"][n] for n in NAMES]
    fig, ax = plt.subplots(figsize=(9.2, 3.6), layout="constrained")
    correct = np.array([v["correct"] for v in stats])
    wrong = np.array([v["wrong_positive"] for v in stats])
    y = np.arange(6)
    ax.barh(y, correct, color=TEAL, label="Correct raw pose")
    ax.barh(y, wrong, left=correct, color=RED, label="Wrong raw pose")
    ax.barh(y, 16-correct-wrong, left=correct+wrong, color=GREY, label="No raw pose")
    for i, (c, w) in enumerate(zip(correct, wrong)):
        ax.text(16.2, i, f"{c}/16 correct; {w} wrong", va="center", fontsize=9)
    ax.set(yticks=y, yticklabels=LABELS, xlim=(0, 20), xticks=[0, 4, 8, 12, 16],
        xlabel="All 16 eligible positive pairs; failures retained", title="More recovery, modest isolated selection gain")
    ax.invert_yaxis(); ax.legend(loc="upper left", bbox_to_anchor=(0, -.18), ncol=3, fontsize=8, frameon=False)
    fig.savefig(ROOT/"recovery.png"); plt.close(fig)

    faint = [r for r in scores if r["case"]["positive"] and r["case"]["reference_index"] == 0]
    fig, ax = plt.subplots(figsize=(8.5, 3.5), layout="constrained")
    for name, label, color, marker in zip(NAMES[1:], LABELS[1:], [GREY, BLUE, RED, "#9b7d2c", TEAL], ["s", "o", "v", "D", "*"]):
        values = [np.nan if r["methods"][name]["pose_error"] is None else r["methods"][name]["pose_error"]["translation_m"] for r in faint]
        ax.plot(range(4), values, color=color, marker=marker, markersize=8 if name == "active" else 5, label=label)
    threshold = .01*faint[0]["methods"]["active"]["pose_error"]["reference_range_m"]
    ax.axhline(threshold, ls="--", color="#637286", lw=1)
    ax.text(3.08, threshold, "18.79 m limit", va="center", fontsize=8)
    ax.set(xticks=range(4), xticklabels=["ref0 / query0", "ref0 / query1", "ref1 / query0", "ref1 / query1"],
        xlim=(-.15, 3.8), ylim=(0, 30), ylabel="Absolute camera translation error (m)", title="Faint condition: four repeated rendering-noise pairings")
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper left")
    ax.text(.02, .02, "Missing points = no pose. Direct returns no pose on all four.", transform=ax.transAxes, fontsize=8)
    fig.savefig(ROOT/"faint-errors.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
    offsets = [(8, -14), (8, -14), (-74, -18), (-104, -8), (8, -18), (8, 9)]
    for n, label, record, offset in zip(NAMES, LABELS, stats, offsets):
        x, y = record["median_positive_online_seconds"], 100*record["correct"]/16
        ax.scatter(x, y, color=TEAL if n == "active" else BLUE, s=70 if n == "active" else 35)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=8)
    ax.set(xlim=(0, 15), ylim=(15, 60), xlabel="Median fresh online seconds per positive pair (same CPU)",
        ylabel="Raw pose recovery (%)", title="Accuracy costs time: no efficiency advantage over Direct")
    ax.text(.02, .96, "Reference preparation amortized; worker peak 1.38 GB is shared.", transform=ax.transAxes, va="top", fontsize=8)
    fig.savefig(ROOT/"accuracy-runtime.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.7, 3.4), layout="constrained")
    values = np.array([[result["by_condition"][str(i)][n]["correct"] for i in range(4)] for n in NAMES])
    ax.imshow(values, vmin=0, vmax=4, cmap="GnBu", aspect="auto")
    for y in range(6):
        for x in range(4): ax.text(x, y, f"{values[y,x]}/4", ha="center", va="center", color="white" if values[y,x] >= 3 else BLUE)
    ax.set(yticks=range(6), yticklabels=LABELS, xticks=range(4),
        xticklabels=["Faint\nnp31-r3-c3", "Bright\nnp31-r0-c4", "Nearly uniform A\nnp31-r1-c0", "Nearly uniform B\nnp31-r1-c0"],
        title="The gain is concentrated in one exposed terrain condition")
    fig.savefig(ROOT/"conditions.png"); plt.close(fig)


def demo(protocol, raw, scores):
    # Existing public renders only. Inference is never called by this builder.
    from PIL import Image as PILImage
    sys.path.insert(0, str(Path.cwd()))
    from replay_raw_acquisition_v109 import decode, luminance
    fig, axes = plt.subplots(3, 2, figsize=(8, 11), layout="constrained")
    for row, index, title in zip(range(3), [0, 1, 16], ["Recovered faint pose: 5.70 m / 0.177 deg", "Nonoverlap: no raw pose", "Remaining failure: nearly uniform images, no raw pose"]):
        entry = raw["rows"][index]; case = entry["case"]
        reference, query = (protocol["inputs"][case[n]] for n in ("reference_index", "query_index"))
        a = luminance(decode(Path(reference["reference_radiance"][case["reference_seed"]]["path"]).read_bytes(), exr_size=512))
        b = luminance(decode(Path(query["query_radiance"][case["query_seed"]]["path"]).read_bytes(), exr_size=512))
        finite = np.r_[a.ravel(), b.ravel()]
        lo, hi = np.quantile(finite, [.01, .995])
        if hi <= lo: hi = lo+1.
        for col, image in enumerate((a, b)):
            axes[row,col].imshow(image, cmap="gray", vmin=lo, vmax=hi)
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            axes[row,col].set_title(("Reference" if col == 0 else "Query")+" | "+title, fontsize=8)
        method = entry["result"]["methods"]["active"]
        for col, name in enumerate(("reference_xy", "query_xy")):
            endpoints = np.asarray(method[name], float).reshape(-1, 2)
            if len(endpoints): axes[row,col].scatter(*endpoints.T, s=22, facecolors="none", edgecolors="#65edd7", lw=.8)
    fig.suptitle("Saved-output demonstration | Active +32\nDisplay-only common contrast per pair; conditional matches are not independent landmarks.", fontsize=10)
    fig.savefig(ROOT/"demonstration.png"); plt.close(fig)


def make_pdfs(result):
    OUT.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title137", fontName="Helvetica-Bold", fontSize=23, leading=26, textColor=colors.HexColor(BLUE), spaceAfter=12))
    styles.add(ParagraphStyle(name="Deck137", fontName="Helvetica", fontSize=11, leading=15, textColor=colors.HexColor(TEAL), spaceAfter=11))
    styles.add(ParagraphStyle(name="Body137", fontName="Helvetica", fontSize=9.5, leading=13.6, spaceAfter=8, textColor=colors.HexColor(BLUE)))
    styles.add(ParagraphStyle(name="Small137", fontName="Helvetica", fontSize=8, leading=10.8, spaceAfter=7, textColor=colors.HexColor(BLUE)))
    styles.add(ParagraphStyle(name="H137", fontName="Helvetica-Bold", fontSize=12, leading=15, spaceBefore=7, spaceAfter=7, textColor=colors.HexColor(BLUE)))
    width = 7.0*inch
    def p(text, style="Body137"): return Paragraph(text, styles[style])
    def img(name, h): return Image(str(ROOT/name), width=width, height=h*inch, kind="proportional")
    def head(kicker, title, deck): return [p(kicker.upper(), "Small137"), p(title, "Title137"), p(deck, "Deck137")]
    def table():
        rows = [["Method", "Correct /16", "Wrong", "Strict /16", "Online (s)"]]
        for label,n in zip(LABELS,NAMES):
            v=result["methods"][n]
            rows.append([label,str(v["correct"]),str(v["wrong_positive"]),str(v["strict"]),f'{v["median_positive_online_seconds"]:.2f}'])
        t=Table(rows, colWidths=[2.8*inch,1.05*inch,.7*inch,1.05*inch,1.0*inch])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(BLUE)),("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(-1,-1),"Helvetica"),
            ("FONTSIZE",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),
            ("BACKGROUND",(0,-1),(-1,-1),colors.HexColor("#e5f4f2")),("LINEBELOW",(0,0),(-1,-1),.3,colors.HexColor("#d9e0e7"))]))
        return t
    def footer(canvas, doc):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor("#d9e0e7")); canvas.line(54,42,558,42)
        canvas.setFont("Helvetica",8); canvas.setFillColor(colors.HexColor(BLUE))
        canvas.drawString(54,29,"Lunar image matching | v137 development study | 5 November 2026")
        canvas.drawRightString(558,29,str(doc.page)); canvas.restoreState()
    def write(filename, story):
        doc=SimpleDocTemplate(str(OUT/filename), pagesize=(612,792), leftMargin=54,rightMargin=54,topMargin=42,bottomMargin=54,
            title="Active patch acquisition for lunar camera localization: preliminary development evidence", author="Thomas de Chillaz",
            subject="Reproducible development pilot, not a validated navigation model")
        doc.build(story,onFirstPage=footer,onLaterPages=footer)

    brief=head("Research brief / preliminary evidence", "Where should a lunar matcher look next?", "Active patch acquisition recovers 8/16 simulated camera poses; equal-budget controls recover 7/16.")
    brief += [p("Question", "H137"),p("Can native image measurements selected to distinguish camera hypotheses recover poses that fixed measurements miss? The specific candidate combines a shared-camera correlation fit with a terrain-uncertainty-aware patch-selection proxy. It adds zero trainable parameters."),
        p("Matched development result", "H137"),table(),Spacer(1,9),
        p("Correct: rotation &lt;=2 degrees and translation &lt;=1% of median reference range. Strict: &lt;=0.5 degrees and &lt;=0.1%. Every positive remains in the denominator. All six methods return no raw pose on 16 nonoverlaps. These are raw proposals, not calibrated acceptances.","Small137"),
        img("recovery.png",2.3),
        p("Interpretation: +25 points over Direct, but only +6.25 points (one pose) over uniform/geometry selection. The latter is the isolated policy comparison. Most improvement is shared extra acquisition, not evidence unique to active selection.")]
    brief += [PageBreak(),*head("What the evidence supports", "A useful lead, with clear limits", "An independently replayable development finding suitable for technical feedback; no superiority or flight-readiness claim."),
        p("Mechanism and failure", "H137"),p("Begin with 36 native correlation fields, fit one camera, and acquire 32 further patches sequentially. At each step, compare seven camera hypotheses with conservative reference-height envelopes (0, +/-5, +/-10, +/-20 m), obtain the selected query measurement, then update the camera. Finish with the original six-match, three-pixel AP3P solver."),
        img("faint-errors.png",2.25),
        p("On the faint condition, active selection recovers all four noise pairings; uniform and geometry each recover three. On the first pairing, translation error is 5.70 m for active versus 23.75 m for uniform (18.79 m limit). Bright cases already succeed; all eight nearly uniform positives still fail."),
        p("Evidence limits", "H137"),p("The 16 positives contain only four exposed acquisition/lighting conditions on three terrain components. Repeated rendering noise is not independent geography. Matching reference illumination is supplied. The descriptive terrain interval for active minus uniform/geometry is [0, 19.58] points and includes zero. Selection uses height-envelope perturbations; full terrain rerender robustness is untested."),
        p("Active costs 11.31 s versus 1.36 s for Direct on the same CPU. The 50% raw recovery misses the original 70% accepted-pose target. Unknown-Sun, fresh-terrain, learned-response, transfer and calibrated-risk benefits remain open. Existing LoMa/RoMa comparisons use a different cohort and cannot rank this method."),
        p("Reproduction and next decisive test", "H137"),p(f'All 832 method/case scores independently replay. Frozen inference source: 48c6e9a; no thresholds changed. Code, manifest and saved-output replay: <link href="{URL}/tree/main/reports/solar-active-v137" color="#087f8c">GitHub research package</link>. The next funded/admitted study should test the unchanged policy on additional development terrain, lighting mismatch and full terrain perturbations, against the same equal-budget controls. Calibration/final terrain stays closed.',"Small137"),
        p('Related work: Davison, <i>Active Search for Real-Time Vision</i> (ICCV 2005), establishes uncertainty-guided image search. Ansar et al., <i>Illumination Invariant Image Matching for Lunar TRN</i> (2025), motivates the application. This pilot does not establish novelty against either work.',"Small137")]
    write("lunar-active-acquisition-v137-brief.pdf",brief)

    paper=head("Working paper draft / development only", "Active patch acquisition for lunar camera localization", "A controlled pilot of hypothesis-conditioned native image measurement under weak image evidence")
    paper += [p("Abstract","H137"),p("We investigate whether choosing additional image measurements can repair a lunar camera-localization pipeline that produces plausible camera proposals but too few reliable correspondences. A shared-camera correlation fit is followed by sequential patch acquisition using a reference-only, height-envelope prediction-separation proxy. In a source-frozen experiment with 16 positive and 16 nonoverlapping image pairs, the active policy recovers eight raw poses, versus four for reference-side render-and-match and seven each for equal-budget uniform and geometry selection. No wrong positive or nonoverlap poses are returned by the active policy. However, the policy-specific gain is one pose, its terrain-group interval includes zero, and runtime is 8.3 times Direct's. Only three exposed terrain components and supplied matching reference illumination are tested. These results motivate a larger test; they do not establish an accepted-pose, novel learned-model or state-of-the-art advantage."),
        p("1. Scope and research question","H137"),p("Lunar terrain-relative navigation must handle geometric viewpoint changes and strong lighting effects. Our narrower question is whether spending a fixed additional image-processing budget at hypothesis-discriminating locations improves final camera estimation. We deliberately distinguish a raw pose proposal, a geometrically correct proposal in simulation, and a calibrated accepted output. This study measures the first two only."),
        p("The original project failed to recover any poses on this diagnostic. Earlier shared-camera repairs restored four of 16 but did not beat Direct. A joint soft-correlation pilot improved a faint camera proposal while leaving only five reliable image matches, below the fixed six-match final gate. Additional measurements are therefore a testable mechanism, without relaxing the solver or fitting a rejection rule."),
        p("Status of this draft","H137"),p("The experiment and independent replay are complete. This is a preliminary working draft, not a submitted paper. The broader compact learned A/B/C model, fresh-terrain test, unknown-Sun evaluation, seed repetitions and calibrated final performance remain unfinished. No NASA/JPL affiliation is claimed.")]
    paper += [PageBreak(),*head("Method", "Measure, update, then choose again", "No new learned weights or acceptance calibration."),
        p("2. Shared-camera image evidence","H137"),p("Recompute the frozen geometry-based proposal bank and its eight-candidate, four-scale photometric initializer from the image pair. Form 36 supported masked native correlation fields. Optimize a common six-dimensional camera against smoothed correlation potentials; this remains a proposal. Query depth, true query pose and true query Sun are unavailable to inference. Reference geometry, pose and intrinsics are supplied."),
        p("3. Sequential acquisition policy","H137"),p("Construct 256 spatial reference candidates, excluding missing geometry, unsupported locations and points within 12 pixels of an existing anchor. Around the current camera, form seven bounded hypotheses along three weak directions of the acquired points' projection Jacobian. For each candidate, predict nine cheap reference-preview intensity changes using a first-order gradient/displacement approximation."),
        p("Perturb reference points by 0, +/-5, +/-10 and +/-20 metres along local up converted into the reference-camera frame. The active proxy ranks separation between predicted-observation envelopes. Unknown hypothesis pairs supply zero support and retain their denominator weight. This conservative score is neither a calibrated covariance nor Shannon mutual information."),
        p("Select one candidate within the fixed common spatial slot, acquire its actual query correlation field, refit the shared camera and regenerate hypotheses. Repeat for 32 additional searches. Unselected extra candidates never receive expensive query-field extraction; failed attempts still consume their slot. Reference previews, hypothesis management and input decoding are charged in online time."),
        p("4. Final measurement and pose","H137"),p("Choose camera-conditioned modes on the actual image grid, fit local fractional peaks, and retain the frozen appearance, ambiguity and independent reverse checks. Use the original AP3P/refit with at least six matches and a three-native-pixel threshold. The soft camera is never substituted for a final failure. Camera-conditioned endpoints are not independent landmarks."),
        p("Confound controlled explicitly","H137"),p("Additional templates use reference 3D support without the old fitted-plane intersection restriction. Every additional-acquisition control shares this path. Thus comparisons with the no-extra method combine extra measurements and extra-template support; only comparisons between equal-budget policies isolate selection.")]
    paper += [PageBreak(),*head("Evaluation", "A frozen, bounded diagnostic", "Four exposed camera/lighting conditions; three terrain components."),
        p("5. Data and controls","H137"),p("Use all 32 image pairs from the prior native diagnostic: 16 positives and 16 certified nonoverlaps. Four acquisition conditions each have two independent reference-rendering noise realizations crossed with two query realizations. They comprise one faint, one bright and two nearly uniform conditions; the latter two share a terrain component. These repetitions are not 16 independent terrain samples."),
        p("The reference image is rendered at the query-matching illumination, and Direct also searches its frozen 24-light reference library. This is a supplied-light diagnostic, not the primary estimated-Sun track. No additional geography is acquired. Four calibration and twelve final terrain groups remain untouched."),
        p("Fresh arms: Direct; joint fitting without extra acquisition; and joint fitting plus 32 searches using uniform, reference-gradient appearance, geometry-only, or active predicted-observation selection. All four policies share eligible candidates, spatial slots, local search bounds, update rule and final solver. Policy order rotates over noise combinations; Direct runs first/last alternately. Twenty historical methods are copied unchanged for identity controls, not retimed."),
        p("6. Metrics and audit","H137"),p("Correct raw pose: rotation error <=2 degrees and absolute translation error <=1% of median reference range. Strict: <=0.5 degrees and <=0.1%. Include every positive, including failures. Report native correspondence errors at 1/2/4/8/12 pixels; no matcher-derived coordinates become ground truth. Negative analysis accounts for both terrain endpoints."),
        p("Seal predictions before loading query scoring geometry. Independently reconstruct geometry and score all 26 methods x32 pairs =832 cells without importing the matcher or pose estimator. Fresh initializer/Direct outputs and all 20 historical controls must match exactly; eight available no-extra rows must match the prior joint pilot. All checks pass."),
        p("7. Resources","H137"),p("Native 512-pixel images; two CPU threads per serial job, OpenCV one; no paid compute. The explicitly approved 2,400-second supplement reserves 120 seconds for full-budget generated controls and 1,800 for comparison/replay. Actual elapsed is 66.43+499.38 seconds. Fixed charges remain; 480 seconds is an unused recovery envelope, not new-study authority. Original accounting is unchanged.")]
    paper += [PageBreak(),*head("Results", "The extra measurements help", "The active policy adds one pose over the strongest equal-budget controls."),
        table(),Spacer(1,9),p("No raw nonoverlap output occurs for any fresh arm (0/16 each). Strict recovery is four for active and five for geometry. Online medians include fresh initialization and input decoding, with reference-only caches permitted. Whole-worker peak RSS is 1.38 GB including shared caches; this cannot rank per-method memory.","Small137"),
        img("conditions.png",2.15),
        p("8. What improves","H137"),p("Active recovers all four faint-condition noise pairings and all four bright cases. Uniform and geometry each recover three faint cases; appearance recovers two. Every method fails all eight nearly uniform positives. Against Direct the active gain is 25 percentage points; against no-extra it is 18.75 points. The isolated policy gain over uniform/geometry is 6.25 points, below the original ten-point improvement target."),
        p("The prespecified early-signal rule passes: at least two more correct positives than historical/no-extra/Direct controls, strictly more than each equal-budget policy, and no additional wrong/nonoverlap outputs versus Direct. This modest diagnostic rule does not replace the original accepted-pose success criteria."),
        p("9. Uncertainty","H137"),p("Enumerating 27 three-component bootstrap draws gives descriptive 95% active-minus-Direct interval [0,78.33] points and active-minus-uniform/geometry [0,19.58]. Both include zero; three components are too few for a stable population interval. Nine endpoint-weighted negative draws contain no edges. All-zero empirical negative rates cannot certify a 1% deployment risk.")]
    paper += [PageBreak(),*head("Failure analysis and prior art", "Recovery is not yet robust localization", "Promising mechanism evidence, with substantial unresolved failure modes."),
        img("accuracy-runtime.png",2.05),
        p("10. Precision and cost","H137"),p("On the first faint noise pairing, active reduces final translation error to 5.70 m versus uniform 23.75 m (criterion 18.79 m), so this difference is not a tiny threshold crossing. Yet active does not uniformly minimize error: geometry reaches 1.86 m on another pairing versus active 11.08 m. Total active matches within one/two pixels are 224/243; uniform has 228/244. More correct local matches alone do not explain the final pose count."),
        p("The nearly uniform images fail initialization and remain in the denominator. The method cannot create evidence in a textureless view. The active latency is 11.31 s versus Direct 1.36 s, about 8.33 times slower; there is no measured efficiency advantage. Full rerendered terrain/occluder perturbations remain untested: the current envelope changes local point projections only."),
        p("11. Related work and comparison boundaries","H137"),p("Davison [1] establishes uncertainty-guided active image search and information/cost reasoning. PixLoc [2] provides learned featuremetric camera refinement. Our proxy combines a particular masked native correlation pipeline with height envelopes; novelty against broader active localization literature is unproven. Ansar and colleagues' lunar TRN work [3] makes illumination-invariant matching a directly relevant problem, but we do not reproduce or outperform that paper."),
        p("LoMa [4] and RoMa v2 [5] are strong modern matchers with different standard benchmarks and metrics. Our separate eight-development-terrain/48-positive benchmark records Direct 30/48, LoMa-R 15/48 and RoMa v2 15/48. The new active method has not run that cohort; 8/16 here cannot be ranked against 15/48 there, or against their published terrestrial scores. No new state-of-the-art claim is supported.")]
    paper += [PageBreak(),*head("Limitations, reproduction and next test", "What would make the claim stronger?", "Keep the current policy frozen while testing whether the one-pose advantage generalizes."),
        p("12. Remaining success criteria","H137"),p("The original aim of >=70% correctly accepted poses is unmet (50% raw here, no calibration). The >=10-point accepted gain and a positive terrain interval are also unmet. Empirical accepted-error/nonoverlap constraints require independent calibration and adequate evaluation. Unknown-Sun, fresh-terrain, full learned A/B/C ablations, three training seeds, sensor transfer and modern accuracy/efficiency comparisons remain open. Preserve the untouched final reserve until development supports a candidate."),
        p("13. Proposed next discriminating experiment","H137"),p("Before another scientific job, publish an admitted, bounded schedule on additional available development terrain. Keep all current methods and gates frozen. Cross reference/query lighting mismatch with viewpoint change; compare all four acquisition policies at equal 32/64/128 budgets and rerender reference terrain/occluders at declared perturbations. A gain over Direct without a gain over uniform is a measurement-budget result, not an active-policy contribution. Test original nonoverlaps with both endpoints recorded. If no consistent gain remains, publish that negative result."),
        p("14. Reproducibility","H137"),p(f'Frozen inference source: 48c6e9ac4f15a1787a3bfdce6a5b84de54090763. Protocol SHA256: c39533b10b16e9ba0cc48c5c2aa8d671f1c037bbcc8dc89670a5d3c5cfce49b7. The public replay package includes saved outputs, seals, independent arithmetic, environment pins and complete case lists. Source and replay instructions: <link href="{URL}/tree/main/reports/solar-active-v137" color="#087f8c">repository research directory</link>. No private compute ledger, credentials or correspondence is distributed.',"Small137"),
        p("References","H137"),
        p('[1] A. J. Davison. Active Search for Real-Time Vision. ICCV, 2005. <link href="https://www.doc.ic.ac.uk/~ajd/Publications/davison_iccv2005.pdf">Author PDF</link>.<br/>[2] P.-E. Sarlin et al. Back to the Feature: Learning Robust Camera Localization from Pixels to Pose. CVPR, 2021. <link href="https://openaccess.thecvf.com/content/CVPR2021/html/Sarlin_Back_to_the_Feature_Learning_Robust_Camera_Localization_From_Pixels_CVPR_2021_paper.html">CVF paper</link>.<br/>[3] N. Rothenberger, G. Georgakis, Y. Cheng, A. Ansar. Illumination Invariant Image Matching for Lunar TRN. AIAA SCITECH, 2025. <link href="https://doi.org/10.2514/6.2025-2073">DOI</link>.<br/>[4] D. Nordstrom et al. LoMa: Local Feature Matching Revisited. ECCV, 2026. <link href="https://www.davnords.com/loma">Project and results</link>.<br/>[5] J. Edstedt et al. RoMa v2: Harder Better Faster Denser Feature Matching. ECCV, 2026. <link href="https://arxiv.org/abs/2511.15706">arXiv:2511.15706</link>.',"Small137"),
        p("Conclusion","H137"),p("Sequential additional acquisition repairs some missing-measurement failures. A reference-only active selection proxy supplies a small extra development gain, with no evidence yet of a general, learned, calibrated or efficient lunar localization advance.")]
    write("lunar-active-acquisition-v137-working-paper.pdf",paper)


if __name__ == "__main__":
    result=read(ROOT/"results.json"); scores=read(RUN/"scores.json")
    if not result["passed"] or result["method_case_cells"] != 832:
        raise ValueError("Complete independent replay required before presentation")
    figures(result,scores)
    demo(read(ROOT/"protocol.json"),read(RUN/"predictions.json"),scores)
    make_pdfs(result)
