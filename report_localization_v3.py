"""Render measured development diagnostics; does not run or tune a matcher."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def report():
    root = Path('reports')
    run = Path('runs/localization-diagnostics-v3-validated')
    summary = json.loads((run/'summary.json').read_text())
    gate = json.loads((root/'localization_data_gate_v3.json').read_text())
    refit = json.loads((root/'localization_pose_refit_v3.json').read_text())
    with gzip.open(run/'results.json.gz', 'rt') as f:
        rows = json.load(f)
    for filename in ('localization_replay_audit_v3.json', 'localization_pose_refit_audit_v3.json'):
        if not json.loads((root/filename).read_text())['passed']:
            raise ValueError('Replay audit required before reporting')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    s = summary['results']; r = refit['results']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
    for method, label, color in [('coarse', 'Coarse', '#52728e'), ('deterministic_native', 'Native refinement', '#107d83')]:
        thresholds = [1, 2, 4, 8, 12]
        axes[0].plot(thresholds, [100*s[method]['precision_at_native_px'][str(t)] for t in thresholds],
                     marker='o', label=label, color=color)
    axes[0].set(xlabel='Correspondence tolerance (original pixels)', ylabel='Precision among co-visible points (%)',
                ylim=(0, 100), title='Native detail improves localization precision')
    axes[0].legend(frameon=False)
    names = ['Coarse\nEPNP', 'Native\nEPNP', 'Coarse\nEPNP + LM', 'Native\nEPNP + LM']
    values = [s['coarse']['correct_poses'], s['deterministic_native']['correct_poses'],
              r['coarse']['refined_correct'], r['deterministic_native']['refined_correct']]
    axes[1].bar(names, values, color=['#52728e', '#107d83', '#52728e', '#107d83'])
    for x, y in enumerate(values):
        axes[1].text(x, y+.8, f'{y}/72', ha='center')
    axes[1].set(ylabel='Poses within criterion (all 72 pairs)', ylim=(0, 76),
                title='Precision and pose refinement both matter')
    fig.suptitle('Development only: 8 geometries / 4 footprint components', fontsize=12)
    fig.savefig(root/'localization_precision_pose_v3.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout='constrained')
    experiments = ['oracle', 'grid_1', 'grid_2', 'grid_4', 'grid_8', 'grid_16']
    x = np.arange(len(experiments)); width = .36
    axes[0].bar(x-width/2, [s[e]['correct_poses'] for e in experiments], width, label='Original EPNP', color='#52728e')
    axes[0].bar(x+width/2, [r[e]['refined_correct'] for e in experiments], width, label='EPNP + LM', color='#107d83')
    axes[0].set_xticks(x, ['Exact', '1', '2', '4', '8', '16'])
    axes[0].set(xlabel='Query-coordinate rounding grid (original pixels)', ylabel='Correct poses / 8 geometries', ylim=(0, 9))
    axes[0].legend(frameon=False)
    strata = defaultdict(lambda: [0, 0, 0])
    for refined in refit['rows']:
        original = rows[refined['source_row']]
        if original['experiment'] != 'deterministic_native':
            continue
        label = 'Same lighting ID' if original['light_a'] == original['light_b'] else 'Different lighting IDs'
        c = strata[label]; c[0] += 1; c[1] += original['pose_correct']; c[2] += refined['pose_correct']
    labels = list(strata)
    for j, (label, counts) in enumerate(strata.items()):
        for offset, index, color in [(-width/2, 1, '#52728e'), (width/2, 2, '#107d83')]:
            axes[1].bar(j+offset, 100*counts[index]/counts[0], width, color=color)
            axes[1].text(j+offset, 100*counts[index]/counts[0]+2, f'{counts[index]}/{counts[0]}', ha='center')
    axes[1].set_xticks(range(len(labels)), labels)
    axes[1].set(ylabel='Correct pose rate (%)', ylim=(0, 100), title='Native refinement with viewpoint change')
    fig.suptitle('Development sensitivity; lighting IDs are not measured Sun angles', fontsize=12)
    fig.savefig(root/'localization_geometry_lighting_v3.png', dpi=180); plt.close(fig)
    table = '\n'.join(f'| {name} | {s[key]["correct_poses"]}/72 | {r[key]["refined_correct"]}/72 |'
                      for key, name in [('coarse', 'Frozen coarse matcher'), ('deterministic_native', 'Deterministic native refinement')])
    note = f'''# Localization development diagnosis - iteration 7

This is a completed development diagnosis and a failed fresh-data gate, not a trained compact-v3 model or a held-out improvement claim. The compact-v2 release and all earlier results remain unchanged.

## Measured findings

Eight model-selection geometries from four selected footprint components give 72 stereo/light combinations. Repeated lighting cases are dependent. All eligible cases, including missing poses, remain in denominators. Correct pose means rotation <=2 degrees and translation <=1% of median reference ray range; this is a simulated diagnostic, not a spacecraft requirement.

| Frontend | Original EPNP/RANSAC | Same initialization + inlier LM refit |
|---|---:|---:|
{table}

The original-estimator experiment was specified before inference. The LM experiment is a separate, post-diagnostic development ablation. It is standard nonlinear optimization, not a new learned method. Neither modifies the historical v2 estimator.

Exact projected correspondences recover 8/8 poses, passing the 95% software gate. With the original estimator, rounding correct query coordinates to a 16-original-pixel grid gives 0/8. Native descriptor search improves precision within 2 original pixels from {100*s['coarse']['precision_at_native_px']['2']:.2f}% to {100*s['deterministic_native']['precision_at_native_px']['2']:.2f}% among co-visible reference points. Both return {s['coarse']['returned_matches']:,} matches, of which {s['coarse']['scorable_matches']:,} reference points are co-visible. Conditional correspondence precision does not count non-visible predictions as correct.

Native refinement plus LM recovers 12/24 same-light and 14/48 changing-light poses. The stricter 0.5-degree/0.1%-range criterion is met in only 1/72 predicted cases. The current system is still weak. LM also loses an oracle-noise case at each of sigma=1 and sigma=2 pixels, so it is not universally better. These findings support investigating both coordinate precision and pose conditioning; neither is the sole established cause.

![Precision and pose factors](localization_precision_pose_v3.png)
![Geometry sensitivity and lighting](localization_geometry_lighting_v3.png)

## Fresh-data gate: failed before training

The complete released SouthPole inventory contains {gate['camera_files']:,} NPZ camera files, 12,974 stereo geometries and 12,454 complete three-light geometries. Every NPZ contains only intrinsics and cam2world. There are no released trajectory, generation-batch or Sun-angle fields. The outer StereoGeo archive was checked again: TychoCrater is an empty directory.

The conservative fallback keeps each complete mode/folder together. All three fallback groups were already exposed in v2; zero qualify as fresh under that grouping, versus 12 required. Modes are not asserted to be actual trajectory identities. This does not prove that no unused spatial terrain exists: it establishes that the requested batch/trajectory-conservative fresh split cannot be certified from this release. No full-depth fresh-footprint count or trajectory independence is claimed.

Per the accepted plan, compact-v3 training, seed trials, new final-test inference, modern pose rankings, transfer evaluation and a superiority paper are deferred. No model-version alias or untrained checkpoint is published as a working compact-v3 model. LuSNAR scenes7-9 remain the intended transfer reserve; their files/cases have not been acquired or frozen, and no transfer result is claimed.

## Reproduction and audit

Use the validated requirements-lock environment. Large original archives, v2 checkpoints and depth caches are provided through their existing provenance and compact-v2 release assets. The new measurement capsule contains both diagnostic attempts, protocols, saved correspondences, the full compressed camera inventory and replay results. The first attempt predates a non-numerical lazy-import cleanup; the validated run is the one reported here. Both have identical count outcomes. No numerical method or threshold was selected between those executions.

```bash
python localization_data_v3.py --output reports/new_data_gate.json --records runs/new_inventory.jsonl.gz
python localization_diagnostics_v3.py --output runs/new_localization_diagnosis
python localization_pose_refit_v3.py --source runs/new_localization_diagnosis --output reports/new_pose_refit.json
python verify_localization_v3.py --run runs/new_localization_diagnosis --gate reports/new_data_gate.json --output reports/new_localization_audit.json
python verify_localization_pose_refit_v3.py --input reports/new_pose_refit.json --output reports/new_refit_audit.json
```

Independent projection/visibility/pose-error replay checks 312 rows and 15 aggregate tables. A separate refit replay checks another 312 saved poses and 15 tables. The full 76,802-record camera inventory is checked. Repository verification: 137 tests passed, four optional tests skipped; a fresh procedural smoke produced 20 summaries.

Runtime includes separate native feature extraction: 48 native feature maps were measured, with total cost preserved in summary.json. Per-case refinement timings exclude that cache-building cost and must not be advertised as end-to-end speed. No accepted-error or risk-coverage claim is available: no new acceptance policy or nonoverlap pose experiment was fitted/run.

## What permits the next phase

Obtain verified generation-batch/trajectory membership and sufficient untouched footprint groups, or additional independently characterized 3D relighting data satisfying the same split contract. Re-audit the data before training. Then compare the learned module against BOTH native descriptor refinement and pose refitting, train using true 3D visibility/correspondences, and reserve all new final cases until model/policy freeze. No human review or paid compute is required by the diagnostic stage.

## Related work

- [Rothenberger et al., Illumination Invariant Image Matching for Lunar TRN](https://doi.org/10.2514/6.2025-2073): related motivation; no LIMA reproduction or direct paper-percentage comparison is claimed.
- [RoMa v2](https://arxiv.org/html/2511.15706): subpixel refinement and predictive covariance already exist.
- [Semi-Dense Matching Uncertainty Is Not Just Local Confidence](https://arxiv.org/abs/2608.08685): coarse-failure uncertainty is existing work.
- [MOONSt3R](https://github.com/clementinegrethen/StereoLunar): relevant lunar-adapted comparison for the future study; not executed here.
- [LunarGeo release](https://zenodo.org/records/18415119): source cameras, renders and ray-depth.
'''
    (root/'LOCALIZATION_DIAGNOSTICS_V3.md').write_text(note)
    write_pdf(root, table, s, r, gate)


def write_pdf(root, table, s, r, gate):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyLocal', fontName='Helvetica', fontSize=10, leading=14, spaceAfter=9))
    styles.add(ParagraphStyle(name='SmallLocal', fontName='Helvetica', fontSize=8, leading=11, spaceAfter=7))
    styles['Title'].fontSize = 21; styles['Title'].leading = 25; styles['Title'].alignment = TA_LEFT
    story = []
    def p(text, style='BodyLocal'):
        story.append(Paragraph(text, styles[style]))
    p('Lunar localization: what limits the current matcher?', 'Title')
    p('Development diagnostic | Thomas de Chillaz | 31 October 2026', 'SmallLocal')
    p('<b>Result:</b> recovering useful camera pose needs finer correspondences and a well-refined pose estimate. Standard engineering changes recover 26/72 development poses. A new learned compact-v3 model has not been trained.')
    p('Question and protocol', 'Heading2')
    p('Eight development stereo geometries, four selected footprint components, three lighting observations per camera: 72 dependent stereo/light combinations. Exact ray-depth and calibrated cameras define the simulated reference. Query depth and true query pose enter scoring only. Pose success requires rotation error at most 2 degrees and translation error at most 1% of median reference range.')
    data = [['Matcher frontend', 'Original pose fit', 'With LM refit'],
            ['Frozen coarse matcher', '0/72', '3/72'],
            ['Native descriptor refinement', '7/72', '26/72']]
    t = Table(data, colWidths=[2.8*inch, 1.3*inch, 1.3*inch])
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#173b52')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9), ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8), ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#edf3f5'), colors.white])]))
    story.append(t); story.append(Spacer(1, 10))
    p('The original-estimator experiment was specified before inference. LM is a separately disclosed post-diagnostic development ablation: standard nonlinear optimization on the original RANSAC inliers, without changing the 3-pixel threshold. No old result or model was adjusted.', 'SmallLocal')
    story.append(Image(str(root/'localization_precision_pose_v3.png'), width=6.4*inch, height=2.24*inch))
    p('Limits: only 1/72 refined predictions meets the stricter 0.5-degree / 0.1%-range criterion. These are raw pose outcomes, without a calibrated rejection policy. They establish neither reliable navigation nor state-of-the-art superiority.', 'SmallLocal')
    story.append(PageBreak())
    p('What the diagnosis establishes', 'Title')
    p('<b>Precision matters:</b> exact correspondences recover 8/8 poses. Rounding correct query locations to a 16-original-pixel grid gives 0/8 with the original pose fit. Native descriptor refinement raises correspondence precision within 2 original pixels from 4.06% to 48.57% among co-visible points.')
    p('Both frontends return 22,697 matches; 14,938 reference points are co-visible. Native refinement plus LM succeeds on 12/24 same-light and 14/48 changing-light pairs. Lighting IDs are not measured Sun angles. LM is not universally better: it loses cases at some noise levels. Performance remains too weak for a navigation claim.')
    story.append(Image(str(root/'localization_geometry_lighting_v3.png'), width=6.4*inch, height=2.112*inch))
    p('Fresh-data gate: failed before training', 'Heading2')
    p('All 76,802 released SouthPole camera files were inspected: 12,974 stereo geometries, with only intrinsics and camera-to-world matrices. No trajectory or generation-batch IDs are released. The advertised Tycho directory is empty. Conservatively retaining entire mode/folders leaves zero fresh fallback groups; 12 are required. This does not mean all unused terrain is exhausted; the requested split cannot be certified from the available metadata.')
    p('Consequently, training, modern pose rankings, transfer evaluation and a superiority paper remain deferred. Verified batch/trajectory grouping plus untouched spatial data, or an additional suitable 3D relighting source, is needed before the learned-method phase.')
    p('Reproducibility and next comparison', 'Heading2')
    p('Saved correspondences, native coordinates, protocols and the full camera inventory are checksummed. Independent replay checks 312 diagnostic rows and 15 tables, plus 312 refit poses and 15 tables. Repository checks: 137 tests passed, four optional tests skipped; fresh procedural smoke passed. Native feature-extraction costs are reported separately; no end-to-end speed claim is made.', 'SmallLocal')
    p('Next: train pair-specific reliability and native refinement on true 3D correspondences; compare against deterministic refinement plus pose refitting, RoMa v2, LoMa-R, EfficientLoFTR and lunar-adapted MOONSt3R. Real orbital navigation accuracy remains unverified.', 'SmallLocal')
    p('Sources: MoonAnything, doi:10.5281/zenodo.18415119; Rothenberger et al., doi:10.2514/6.2025-2073; RoMa v2, arXiv:2511.15706. Full methods, caveats and commands are in the repository diagnostic report.', 'SmallLocal')
    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#50616e'))
        canvas.drawString(42, 24, 'Independent development study - no NASA/JPL affiliation or flight-readiness claim')
        canvas.drawRightString(570, 24, str(doc.page))
    doc = SimpleDocTemplate(str(root/'localization_diagnostic_brief_v3.pdf'), pagesize=(612, 792),
                            leftMargin=42, rightMargin=42, topMargin=36, bottomMargin=40)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == '__main__':
    report()
