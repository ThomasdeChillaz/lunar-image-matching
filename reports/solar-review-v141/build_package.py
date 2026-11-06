"""Presentation-only research update from saved evidence and primary-source notes.

No renderer, training, matching, camera fitter or confidence calibration is used.
The older v137 papers and every frozen scientific dependency remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/'reports/solar-review-v141'
OUT = ROOT/'output/pdf'
URL = 'https://github.com/ThomasdeChillaz/lunar-image-matching'
BLUE, TEAL, GREY = '#213f5b', '#087f8c', '#e8eef2'
METHODS = ('direct', 'joint_no_extra', 'uniform', 'appearance', 'geometry', 'active')
LABELS = ('Direct', 'Joint / no extra', 'Uniform +32', 'Appearance +32', 'Geometry +32', 'Active +32')


def read(name):
    return json.loads((ROOT/name).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    result = read('reports/solar-active-v137/results.json')
    cost = read('reports/solar-blind-v139/timing.json')
    gate = read('reports/solar-blind-v139/result.json')
    inventory = read('reports/solar-blind-v138/data-preparation.json')
    protocol = read('reports/solar-blind-v140/protocol.json')
    for name, digest in {**protocol['sources'], **protocol['inputs']}.items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen scientific dependency changed: '+name)
    if (not result['passed'] or result['method_case_cells'] != 832
            or result['accepted_pose_study'] or gate['lunar_inference_run']
            or gate['new_lunar_results'] != 0):
        raise ValueError('Preserve the measured/unrun study distinction')
    if (result['methods']['active']['correct'], result['methods']['uniform']['correct'],
            result['methods']['geometry']['correct']) != (8, 7, 7):
        raise ValueError('Saved acquisition result changed')
    forecast = 12*max(s['elapsed_seconds'] for s in cost['samples'])+30
    if abs(forecast-cost['comparison_forecast_seconds']) > 1e-8 or forecast <= 480:
        raise ValueError('Cost-gate evidence changed')
    sources = ['reports/solar-active-v137/results.json', 'reports/solar-active-v137/RESULTS.md',
        'reports/solar-blind-v139/timing.json', 'reports/solar-blind-v139/result.json',
        'reports/solar-blind-v138/data-preparation.json', 'reports/solar-blind-v140/protocol.json',
        'reports/SOLAR_DITHER_FINGERPRINT_V105.md', 'reports/SOLAR_CLEAN_RADIANCE_RESULTS_V106.md',
        'reports/solar-five-arm-results-v123/matched_comparison_v115.json',
        'reports/solar-review-v141/literature.json']
    for name in ('recovery.png', 'conditions.png', 'accuracy-runtime.png', 'faint-errors.png'):
        sources.append('reports/solar-active-v137/'+name)
    snapshot = dict(schema='presentation_evidence_v141', scientific_inference=False,
        model_training=False, calibration_or_final_access=False, source_hashes={n:sha(ROOT/n) for n in sources},
        protected_scientific_source_count=len(protocol['sources']), protected_input_count=len(protocol['inputs']),
        positive_pairs=result['positive_pairs'], terrain_components=result['terrain_components'],
        methods={n:result['methods'][n] for n in METHODS},
        isolated_policy_gain_pp=result['comparisons']['uniform']['active_minus_control_pp'],
        isolated_policy_descriptive_interval_pp=result['comparisons']['uniform']['descriptive_terrain95_interval_pp'],
        unknown_light_lunar_results=0, unknown_light_forecast_seconds=forecast,
        clean_modern_catalogues_available=len(inventory['historical_modern_reference_cameras_with_clean_catalogue']),
        clean_modern_reference_cameras=inventory['historical_modern_reference_cameras'],
        original_requirements_unchanged=True, goal_complete=False, superiority_claim=False)
    (HERE/'evidence.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    return result, cost, read('reports/solar-review-v141/literature.json')


def make_documents(result, cost, literature):
    OUT.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    for name, size, leading, face, color in (
        ('Title141', 23, 26, 'Helvetica-Bold', BLUE),
        ('Deck141', 11, 15, 'Helvetica', TEAL),
        ('Body141', 9.4, 13.2, 'Helvetica', BLUE),
        ('Small141', 8.1, 10.5, 'Helvetica', BLUE),
        ('Head141', 12, 15, 'Helvetica-Bold', BLUE)):
        styles.add(ParagraphStyle(name=name, fontName=face, fontSize=size, leading=leading,
            textColor=colors.HexColor(color), spaceAfter=8, spaceBefore=5 if name=='Head141' else 0,
            keepWithNext=name=='Head141'))
    def p(text, style='Body141'):
        # Normalize prose number spacing without touching HTML links or paths.
        parts = re.split(r'(<[^>]*>)', text)
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(r'\b([A-Za-z]{2,})(?=\d)', r'\1 ', parts[i])
            parts[i] = re.sub(r'\ba(?=\d)', 'a ', parts[i])
            parts[i] = re.sub(r'(?<=\d)(s|m)\b', r' \1', parts[i])
            parts[i] = re.sub(r':(?=\d)', ': ', parts[i])
            for original in ('MASt3R', 'MOONSt3R', 'DINOv2', 'DINOv3', 'AP3P', 'RGB16', 'RGB8', 'RoMaV2'):
                spaced = re.sub(r'\b([A-Za-z]{2,})(?=\d)', r'\1 ', original)
                parts[i] = parts[i].replace(spaced, original)
        return Paragraph(''.join(parts), styles[style])
    def page(kicker, title, deck):
        return [p(kicker.upper(), 'Small141'), p(title, 'Title141'), p(deck, 'Deck141')]
    def image(name, height):
        return Image(str(ROOT/'reports/solar-active-v137'/name), width=504, height=height,
                     kind='proportional')
    def table(rows, widths):
        wrapped = [[p(str(c), 'Small141') for c in row] for row in rows]
        t = Table(wrapped, colWidths=widths, hAlign='LEFT')
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor(GREY)),
            ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#d3dde4')),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        return t
    def results_table():
        rows = [['Method', 'Correct /16', 'Wrong', 'Strict /16', 'Seconds']]
        for name, label in zip(METHODS, LABELS, strict=True):
            v = result['methods'][name]
            rows.append([label, v['correct'], v['wrong_positive'], v['strict'],
                         f"{v['median_positive_online_seconds']:.2f}"])
        return table(rows, [190, 80, 60, 80, 94])
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d3dde4')); canvas.line(54,42,558,42)
        canvas.setFont('Helvetica', 7.5); canvas.setFillColor(colors.HexColor(BLUE))
        canvas.drawString(54,29,'Thomas de Chillaz | Working research update | 6 November 2026')
        canvas.drawRightString(558,29,str(doc.page)); canvas.restoreState()
    def write(filename, story):
        doc = SimpleDocTemplate(str(OUT/filename), pagesize=(612,792), leftMargin=54,
            rightMargin=54, topMargin=42, bottomMargin=54, author='Thomas de Chillaz',
            title='Sequential patch acquisition for lunar localization: evidence and comparison limits',
            subject='Preliminary development study; no new lunar inference or validated navigation claim')
        doc.build(story, onFirstPage=footer, onLaterPages=footer)

    brief = page('Research brief / measured pilot', 'A lunar localization lead<br/>and the next decisive test',
        'Active patch selection recovers 8/16 raw poses versus 7/16 for the strongest equal-search controls.')
    brief += [p('What was tested', 'Head141'), p('A classical native-resolution pipeline fits one camera to correlation fields, then sequentially chooses 32 further image measurements. The active score favors patches that distinguish camera hypotheses despite bounded reference-height uncertainty. It adds zero trainable parameters; no learned-model upgrade is demonstrated.'),
        results_table(), p('Correct: rotation &lt;=2 degrees and translation &lt;=1% of median reference range. Strict: &lt;=0.5 degrees and &lt;=0.1%. Every eligible positive remains in the denominator; all fresh methods return zero raw poses on 16 nonoverlaps. Seconds are median positive-pair online time on the same CPU.', 'Small141'),
        image('recovery.png',170), p('<b>Interpretation.</b> The isolated policy gain is one pose, or 6.25 points; its descriptive terrain interval is [0,19.58] points. Active is about8.33x slower than Direct. This is a development lead, not evidence of a superior or efficient model.')]
    brief += [PageBreak(), *page('Research brief / scope and related work', 'What would make the result useful?',
        'The next contribution must survive unknown lighting, stronger controls and independent terrain.'),
        p('Three limits that matter', 'Head141'),
        p('The 16 positives reuse four exposed acquisition conditions on three terrain components. Matching reference illumination is supplied. All eight nearly uniform positives still fail; strict precision is better for geometry selection (5/16) than active (4/16). Raw recovery is not calibrated correct acceptance.'),
        p('The historical48-positive modern comparison used default-dither RGB8. Fixed detector-coordinate dither can create false image matches in dark renders; the clean control removed three wrong poses without recovering the dark cases. The newer active pilot uses dither-free EXR. These pipelines cannot be pooled into a clean state-of-the-art ranking.'),
        p('Prior art changes the claim', 'Head141'),
        p('Ansar and colleagues already demonstrate <link href="https://doi.org/10.2514/6.2025-2073" color="#087f8c">lunar illumination-aware matching</link>. <link href="https://www.doc.ic.ac.uk/~ajd/Publications/davison_iccv2005.pdf" color="#087f8c">Active search</link> and <link href="https://psarlin.com/pixloc/" color="#087f8c">learned geometric refinement</link> are established. <link href="https://arxiv.org/abs/2510.18172" color="#087f8c">Lunar MASt3R adaptation</link> and <link href="https://arxiv.org/abs/2607.01851" color="#087f8c">up-to-sevenfold lunar-model distillation</link> also exist. Small size, refinement or lunar adaptation alone is insufficient novelty.'),
        p('A narrower hypothesis remains worth testing: <b>selecting actual measurements that distinguish competing camera/light explanations under reference-map uncertainty improves correct camera localization at a fixed search budget.</b> Establish that gain over uniform and geometry controls before making a learned or novelty claim.'),
        p('The primary unknown-light study remains unrun', 'Head141'),
        p('A new reference-only catalogue/camera initializer passes full six-method generated controls. Their pair times are46.06/51.88s; the frozen full eight-case forecast is652.51s, exceeding its480s cap. It stopped before lunar inference. The clean roster contains four positives and four nonoverlaps, with three positive terrain components. It is a diagnostic, not fresh validation.'),
        p('Original targets remain unchanged: &gt;=70% correctly accepted poses, &gt;=10-point gain with a positive paired terrain interval, &lt;=1% empirical accepted error/nonoverlap acceptance, and a modern accuracy or efficiency advantage. Full A/B/C ablations, seeds, transfer and the untouched final test remain open.', 'Small141'),
        p(f'<link href="{URL}/tree/main/reports/solar-review-v141" color="{TEAL}">Full8-page working paper, primary references and source-bound evidence</link>. <link href="{URL}/releases/tag/solar-active-v137-development" color="{TEAL}">Measured pilot</link>; <link href="{URL}/releases/tag/solar-blind-v139-cost-gate" color="{TEAL}">clean inputs and cost-gate replay</link>. No email or paper submission has been made.', 'Small141')]
    write('lunar-localization-v141-research-brief.pdf', brief)

    paper = page('Working paper / preliminary development evidence', 'Sequential patch acquisition\nfor lunar camera localization',
        'A small measured lead, a clean-image evaluation gap, and a precise contribution hypothesis.')
    paper += [p('Thomas de Chillaz', 'Small141'), p('Abstract', 'Head141'),
        p('We study whether sequential native-image measurements can recover lunar camera poses that a fixed measurement set misses. A classical shared-camera correlation pipeline with a reference-height-aware selection proxy recovers8/16 raw poses, compared with7/16 for uniform and geometry policies at the same32-extra-search budget. The isolated6.25-point gain has a descriptive terrain interval including zero. Active selection is about8.33 times slower than Direct. The evidence reuses four supplied-light acquisition conditions on three terrain components; it is not a changing-light generalization result.'),
        p('A separate unknown-light catalogue/camera initializer passes generated controls, but its measured cost forecast prevents lunar inference within the frozen cap. Historical modern comparisons use a different, default-dither image pipeline and cannot establish a clean ranking for the new method. We therefore present an audited diagnostic study and an unproven mechanism hypothesis, rather than a superior learned matcher.'),
        p('1. Research question', 'Head141'),
        p('At a fixed native patch-search budget, can measurements selected to distinguish globally consistent camera and illumination hypotheses improve correct absolute camera localization despite reference-map uncertainty? The eventual compact learned system remains constrained to fewer than100,000 total parameters; the presently measured acquisition component adds no learned parameters.'),
        p('2. What the present evidence contributes', 'Head141'),
        p('The completed result separates extra measurement support from selection policy, retains initialization failures, compares all fresh methods with a common pose solver and independently replays every saved score. The code also exposes a remaining primary-track gap: removing supplied matching light requires a larger camera/light search whose lunar recovery has not been measured.'),
        p('No flight-readiness, orbital accuracy, calibrated1% risk, new-terrain generalization or novelty claim follows. The candidate contribution concerns a specific physical measurement policy; broad active vision, learned refinement and compact lunar reconstruction have relevant precedents.'),
        p('Keywords: lunar terrain-relative navigation; active measurement; camera pose; illumination; reproducibility.', 'Small141')]
    paper += [PageBreak(), *page('Method', 'Select measurements, then update',
        'The measured policy remains frozen; the unknown-light extension is a separate untested lunar configuration.'),
        p('3. Native correlation and camera fitting', 'Head141'),
        p('Inputs are native512-pixel reference/query images, reference depth, known reference camera-to-world pose and both camera intrinsics. Reference geometry lifts image anchors into3D. The query true pose, depth and Sun are scoring-only. The measured pipeline begins with36 native correlation fields and a shared camera fit to their ambiguity-preserving image evidence.'),
        p('4. Sequential acquisition', 'Head141'),
        p('Form256 spatial reference candidates with eligible geometry, excluding locations too close to acquired anchors. Around the current camera, construct seven bounded hypotheses along weak projection-Jacobian directions. Cheap first-order reference previews predict observation changes under these hypotheses. For each anchor, repeat prediction under0,+/-5,+/-10,+/-20m reference-height changes along local up expressed in the reference frame.'),
        p('The active score favors separable prediction envelopes. Unknown evidence gives no supporting discrimination and retains its denominator weight. This is a conservative proxy; it is not calibrated mutual information or a probability of correctness. Select a candidate within the shared spatial slot, acquire its actual query correlation field, update the camera and repeat. Unselected candidates receive no expensive query-field measurement. Failed attempts consume searches.'),
        p('5. Equal-search controls and final pose', 'Head141'),
        p('Uniform, reference-appearance, geometry-only and active policies share candidate eligibility, spatial slots, extra-search limits, extra-template construction, camera updates and final estimation. Only comparisons among these policies isolate selection. The no-extra comparison also changes the measurement and template path, so its full gain is not attributable to the active score.'),
        p('Use camera-conditioned native-grid modes, local peak refinement and frozen appearance/ambiguity/reverse checks. The final AP3P/refit requires six matches and a three-native-pixel reprojection threshold. The soft camera proposal never replaces a failed final solve. Public uncalibrated poses remain null; raw proposals are explicit research outputs.'),
        p('6. Unknown-light extension', 'Head141'),
        p('A separate initializer ranks up to24 reference illuminations across2,048 reference-derived camera hypotheses using full3D z-buffer projections and common photometric support. It refines at most eight seeds and chooses one shared catalogue light from image evidence. This is discrete illumination selection, not continuous Sun estimation or independent solar-visibility validation. The retained catalogue light conditions the same native acquisition controls.')]
    paper += [PageBreak(), *page('Evaluation and data provenance', 'Keep the image pipelines separate',
        'A changed input encoding or shared terrain can change what a comparison establishes.'),
        p('7. Completed acquisition cohort', 'Head141'),
        p('The measured v137 cohort contains16 positives and16 original nonoverlaps. Four exposed acquisition conditions cross two independent reference rendering-noise draws with two query draws. There is one faint, one bright and two nearly uniform conditions; the two uniform conditions share one terrain component. The effective geography comprises only three components. Matching reference illumination is supplied; historical light-index identifiers do not make this an unknown-light experiment.'),
        table([['Evidence', 'Input / scope', 'What may be concluded'],
            ['Historical modern comparison', '48 positives;8 terrains; default-dither RGB8', 'Recorded performance on that encoding only'],
            ['Active acquisition pilot', '16 positives;3 terrains; supplied-light dither-free EXR', 'A small within-pilot policy difference'],
            ['Clean unknown-light roster', '4 positives +4 nonoverlaps; RGB16;104 images', 'Inputs prepared; no lunar predictions'],
            ['Full-budget generated controls', 'Flat/layered pattern fixtures;24 lights;2,048 cameras', 'Software/cost evidence, not physical sunlight robustness']], [136,178,190]),
        p('8. Why default dithering matters', 'Head141'),
        p('Earlier audits found detector-coordinate image structure from Blender default dithering in dark renders. A featureless emission-plane control produced2,025 same-coordinate feature matches with default dithering and zero after disabling it. A targeted four-positive paired clean-image study retained the bright success, removed all three wrong low-light poses and recovered none of the three dark positives. This identifies an encoding failure mechanism; it does not determine the full clean modern ranking.'),
        p('The old48-positive modern benchmark remains a reproducibility record: Direct30/48, LoMa-R15/48 and RoMa v2 15/48. The new active pipeline has not run that roster. Only3/16 reference cameras in the larger roster currently have complete clean24-light catalogues. Thirteen need clean acquisition. Neither those historical counts nor heterogeneous paper percentages can rank the current active method.'),
        p('9. Metrics and protected data', 'Head141'),
        p('Correct raw pose requires rotation error&lt;=2 degrees and absolute translation error&lt;=1% of median Euclidean reference range. Strict pose uses0.5 degrees and0.1%. Retain all eligible positives, including no-initializer/no-pose cases. Native correspondence thresholds are1/2/4/8/12 original pixels. Four calibration and twelve final terrain groups remain closed; there is no independent orbital pose reference or human landmark review.')]
    paper += [PageBreak(), *page('Measured acquisition result', 'One extra pose over equal-search controls',
        'The prospective early signal passes; the original accepted-pose research targets do not.'),
        results_table(), p('Seconds are positive-pair online medians including fresh initialization and decoding, with reference-only caches permitted. All six methods return zero raw poses on16 nonoverlaps. Worker memory includes shared caches and cannot rank individual methods.', 'Small141'),
        image('conditions.png',180),
        p('10. Effect size and uncertainty', 'Head141'),
        p('Raw gains are25/18.75/6.25 points over Direct/no-extra/best policy; only the last isolates selection. The27 three-component bootstrap draws give descriptive intervals[0,78.33] points over Direct and[0,19.58] over uniform/geometry. Both include zero; population evidence remains limited.'),
        p('Active recovers all four faint noise pairings versus three for uniform/geometry. Bright cases already succeed; all eight nearly uniform positives fail. Geometry leads strict recovery5/16 versus active4/16. Active has243 two-pixel matches versus uniform244; more precise matches alone do not explain the pose result.'),
        p('Negative resampling weights both terrain endpoints; nine of27 draws have no edges. Zero observed errors and an all-zero bootstrap cannot establish a1% deployment risk.'),
        p('11. Verification', 'Head141'),
        p('All832 saved method/case cells independently replay. The fresh initializer and Direct match saved controls exactly; all20 historical controls remain unchanged, and eight available no-extra rows repeat the prior pilot. These are reproducibility checks on consumed data, not independent terrain validation.')]
    paper += [PageBreak(), *page('Unknown-light control and failure mechanism', 'The primary experiment is still unrun',
        'Successful generated controls establish wiring; their cost prevented the full lunar comparison.'),
        p('12. Full-budget control result', 'Head141'),
        table([['Control', 'Complete pair time', 'Configuration'],
            ['Flat geometry', f"{cost['samples'][0]['elapsed_seconds']:.2f}s", 'All six methods;24 references;2,048 camera proposals'],
            ['Layered geometry', f"{cost['samples'][1]['elapsed_seconds']:.2f}s", 'Same configuration;32 extra searches per acquisition policy'],
            ['Full8-case forecast', f"{cost['comparison_forecast_seconds']:.2f}s", '1.5 x8 x worst pair +30s margin']], [125,125,254]),
        p('Every generated arm recovered its known camera within the frozen control tolerance. The fixtures contain arbitrary image patterns, not a physical sunlight simulation. The measured forecast exceeds even the full480-second recovery cap before the pilot runtime. The controller stopped before lunar prediction, so this is not0/4 or0/8 lunar recovery. No smaller substitute was run.'),
        p('RGB16 is decoded without RGB8 truncation and converted from Standard/sRGB to linear luminance. Geometry support is independent of brightness. Camera/light scores and query measurements are fresh; only reference projections/descriptors are cached. The shared illumination output remains a catalogue proposal, not a certified Sun estimate. The full24-light grid also contains the tested query-light configurations; off-grid robustness is not established.'),
        p('13. What the zero-pose work taught us', 'Head141'),
        p('Earlier native measurements failed through weak candidate support and inadequate reliable correspondences. Prior reference-frame and measurement repairs restored4/16, tying Direct. A later faint camera proposal came within10.02m but supplied only five reliable matches, below the six-match gate. Sequential extra acquisition can cross that support threshold on the faint condition. This is a measured pathway, not proof that the full prior0/16 failure had a single cause.'),
        p('14. Remaining mechanisms to distinguish', 'Head141'),
        p('The present policy may help by measurement count, spatial arrangement, camera conditioning or local image ambiguity. Equal-budget controls are essential. A learned response model must add a gain beyond deterministic native refinement, not merely obtain more searches or more reference information. Height-envelope projections do not substitute for rerendering altered terrain/shadows or for validating solar visibility under missing map coverage.'),
        p('No new training, calibration, final evaluation or sensor-transfer result is reported by this document update. Earlier software and data failures, frozen deadlines and resource charges remain preserved.', 'Small141')]
    paper += [PageBreak(), *page('Prior art and state-of-the-art context', 'Compare the task before the number',
        'Published scores describe different datasets, geometry access and error definitions.'),
        table([['Source', 'Published result / method', 'Relationship to this study'],
            ['Lunar TRN [1]', 'Fine-tuned LoFTR unseen-region RA-1/RA-5:70.6/85.9%', 'Match precision, not absolute pose success; lunar adaptation already exists'],
            ['LoMa [4]', 'LoMa-G HardMatch54.3; WxBS73.4 mAA@10px', 'Not the LoMa-R profile in our CPU study'],
            ['RoMa v2 [5]', 'MegaDepth86.6; ScanNet73.8 AUC@20deg', 'Relative-pose AUC; paper uses800/1024px matching/refinement'],
            ['RoMa-Omega [6]', 'Its Table2: WxBS72.9 vs64.8 for RoMa v2', 'Within-paper comparator; A100 throughput is not our CPU latency'],
            ['Lunar distillation [8]', 'Up to7x smaller than688M teacher; reduced student94.9M', 'Compact lunar reconstruction is prior art; far above our100k constraint']], [99,210,195]),
        p('15. What is already established', 'Head141'),
        p('LIMA derives illumination-aware evidence from co-registered references; the same paper separately adapts LoFTR to lunar imagery [1]. RoMa v2 supplies subpixel refinement and predicted covariance [5]. CoRe explicitly models coarse-assignment failures separately from fine noise and uses reliability-weighted refitting [9]. Active uncertainty-guided search [2] and learned featuremetric localization against3D maps [3] are established ideas.'),
        p('StereoLunar adapts MASt3R to lunar reconstruction/relative pose and releases MOONSt3R [7]. Recent lunar foundation-model distillation [8] uses SVD-based decoder initialization and feature transfer. Thus neither lunar fine-tuning, compression nor a reliability/refinement head alone is a sufficient new contribution.'),
        p('16. Candidate distinction', 'Head141'),
        p('The remaining hypothesis couples one physically consistent illumination explanation to camera-conditioned native measurements, then chooses observations robust to reference-map uncertainty. It is a candidate integration, not established novelty. A defensible claim needs a gain uniquely attributable to that coupling, beyond common reference information and equal-search deterministic controls. The current one-pose supplied-light lead is insufficient.'),
        p('This is a targeted review, not an exhaustive novelty search. No published terrestrial matching score is converted into our lunar accepted-pose rate. No new untested method replaces a required baseline.', 'Small141')]
    paper += [PageBreak(), *page('Research contract and next evidence', 'A falsifiable contribution,<br/>with the original targets',
        'The project remains active; documentation quality does not satisfy the model-performance requirements.'),
        p('17. Necessary discriminating comparisons', 'Head141'),
        p('Keep the prepared clean eight-case study frozen: Direct; joint without extras; uniform, appearance, geometry and active with32 extra searches. Estimate query light from images; do not add a true-query matching template. Retain all failures and original nonoverlaps. Seal every prediction before loading scoring geometry and independently replay all48 cells. This tiny study diagnoses the primary-track change; it cannot establish generalization or1% risk.'),
        p('For the full contribution, evaluate all eight A/B/C on/off combinations and the pooled-response control with frozen checkpoints. Separate data, response representation, globally shared Sun and active measurement effects. Include an independent-Sun-per-correspondence control. Repeat the selected learned configuration with the three predefined seeds and preserve every outcome.'),
        p('18. Original success targets', 'Head141'),
        table([['Requirement', 'Current evidence'],
            ['>=70% correctly accepted poses', '50% raw supplied-light recovery; no accepted-pose study'],
            ['>=10-point accepted gain; positive terrain interval', '6.25-point raw gain over strongest equal-search policy; interval includes0'],
            ['<=1% empirical accepted error/nonoverlap', 'Zero raw errors on a small dependent pilot; no independent calibration'],
            ['Modern pose advantage or useful accuracy/speed tradeoff', 'No matched clean modern ranking; active slower than Direct'],
            ['Fresh test, transfer and robustness', 'Untouched reserve preserved; unknown-light, full terrain and sensor claims open']], [226,278]),
        p('19. Fair modern comparison', 'Head141'),
        p('Complete the declared native comparisons against frozen/recalibrated CNN, deterministic refinement, SIFT, MatchAnything-EfficientLoFTR, LoMa-R, RoMa v2 and MOONSt3R where pretrained controls and resources qualify. Preserve author profiles and common-input profiles separately. Use identical original image content, reference geometry and pose estimation; map all endpoints to native pixels. Query depth/pose/Sun remain labels, except the separately declared supplied-Sun track.'),
        p('Clean acquisition must precede a broader clean ranking. Record training-data overlap, both negative endpoints, reference preparation, latency and memory. Keep calibration/final groups closed until development supports a qualifying frozen candidate. An inconclusive or negative learned result remains a finding; it does not become a superior model through threshold adjustment.')]
    paper += [PageBreak(), *page('Reproduction and references', 'Inspect the evidence,<br/>keep the claims bounded',
        'Presentation generated from unchanged saved results and a dated primary-source review.'),
        p('20. Reproduction', 'Head141'),
        p(f'<link href="{URL}/releases/tag/solar-active-v137-development" color="{TEAL}">V137 release</link> contains the832-cell measured acquisition replay. <link href="{URL}/releases/tag/solar-blind-v139-cost-gate" color="{TEAL}">V139 release</link> contains two ordinary clean-input ZIPs and checksums; extract both into a fresh directory and run <font name="Courier">python reports/solar-blind-v139/replay_cost.py</font>. This verifies101 source/125 input hashes and cost arithmetic, not a saved generated-pose replay.'),
        p('This paper builder imports no matcher, pose fitter, renderer or training code. The companion evidence file hashes the saved statistics, figures, inventory and primary-source notes. The protected104-source/130-input continuation closure remains unchanged. The v137 brief and paper remain historical artifacts. No email or paper submission is automatic.'),
        p('References', 'Head141')]
    order = ['LIMA','ActiveSearch','PixLoc','LoMa','RoMaV2','RoMaOmega','StereoLunar','LunarDistillation','CoRe']
    by_id = {s['id']:s for s in literature['sources']}
    for i, key in enumerate(order, 1):
        item = by_id[key]
        paper.append(p(f'[{i}] {escape(item["authors"])} ({item["year"]}). {escape(item["title"])}. <link href="{escape(item["url"])}" color="{TEAL}">Primary source</link>.', 'Small141'))
    paper += [p('20.1 Source and metric limitations', 'Head141'),
        p('The LIMA numerical table was checked in the existing author copy (p.11); the JPL publication list verifies the DOI. The publisher endpoint was unavailable during this update. Other entries were checked against primary papers or author project pages on6 November2026. Website results and model profiles may differ from published paper configurations; the review records each specific source and metric.'),
        p('Conclusion: additional native measurements repair some low-light support failures, while the active policy supplies only a small, inconclusive supplied-light gain. The next useful result must demonstrate the globally consistent illumination/measurement mechanism on clean data, with matched controls and preserved uncertainty limits.')]
    write('lunar-localization-v141-working-paper.pdf', paper)


if __name__ == '__main__':
    if Path.cwd() != ROOT:
        raise ValueError('Run presentation builder from repository root')
    make_documents(*evidence())
