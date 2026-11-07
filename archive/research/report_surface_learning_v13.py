"""Report the complete matched surface-supervision cohort without best-seed claims."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new
from report_coarse_training_v5 import paired_interval


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return {r['pair_id']: r for r in json.load(stream)}


def count(rows):
    return dict(cases=len(rows), correct=sum(r['pose_correct'] for r in rows.values()),
        strict=sum(r['strict_pose_correct'] for r in rows.values()),
        wrong=sum(r['estimated_cam2world'] is not None and not r['pose_correct'] for r in rows.values()),
        estimated=sum(r['estimated_cam2world'] is not None for r in rows.values()),
        returned=sum(r['returned'] for r in rows.values()), scorable=sum(r['scorable'] for r in rows.values()),
        correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in rows.values()))


def report(root, audit_path, controls_path, prefix):
    root = Path(root); prefix = Path(prefix)
    audit = json.loads(Path(audit_path).read_text()); controls = json.loads(Path(controls_path).read_text())
    if not audit['passed'] or not controls['passed'] or audit['summary_sha256'] != sha256(root/'summary.json'):
        raise ValueError('Completed cohort audit and actual selected-model controls required')
    summary = json.loads((root/'summary.json').read_text()); results = []
    base = read_rows(root/'initial_encoder_3d_results.json.gz')
    if len(base) != 72:
        raise ValueError('Full development denominator required')
    sift_path = Path('runs/localization-v5-axial-correction/results.json.gz')
    sift_audit = json.loads(Path('reports/localization_v5_rescore_audit.json').read_text())
    if not sift_audit['passed'] or sift_audit['results_sha256'] != sha256(sift_path):
        raise ValueError('Audited corrected SIFT outputs required')
    with gzip.open(sift_path, 'rt') as stream:
        sift = {r['pair_id']: r for r in json.load(stream) if r['method'] == 'sift_native' and r['solver'] == 'ap3p_reclassify'}
    if set(sift) != set(base):
        raise ValueError('Missing common SIFT cases')
    for key, row in sift.items():
        row['component'] = base[key]['component']; row['estimated_cam2world'] = row['pose']
    fallback = lambda rows: {i: sift[i] if sift[i]['pose'] is not None else rows[i] for i in rows}
    baseline_fallback = fallback(base)
    for r in summary['results']:
        folder = root/f'{r["experiment"]}-{r["seed"]}'; rows = read_rows(folder/'development_results.json.gz')
        meta = json.loads((folder/'training.json').read_text())
        if set(rows) != set(base):
            raise ValueError('Different eligible cases')
        mixed = fallback(rows)
        results.append(dict(r, raw=count(rows), sift_fallback=count(mixed), total_parameters=meta['total_parameters'],
            raw_gain_pp=100*(count(rows)['correct']-count(base)['correct'])/72,
            raw_interval_pp=paired_interval(rows, base), fallback_interval_pp=paired_interval(mixed, baseline_fallback),
            raw_gained=[i for i in base if rows[i]['pose_correct'] and not base[i]['pose_correct']],
            raw_lost=[i for i in base if base[i]['pose_correct'] and not rows[i]['pose_correct']],
            per_component={g: dict(cases=sum(r['component'] == g for r in rows.values()),
                correct=sum(r['pose_correct'] for r in rows.values() if r['component'] == g))
                for g in sorted({r['component'] for r in rows.values()})},
            checkpoint_sha256=sha256(folder/'selected.pt'), results_sha256=sha256(folder/'development_results.json.gz')))
    selected = [r for r in results if r['experiment'] == summary['winner']]
    output = dict(scope='Matched development supervision study; no fresh final, calibrated acceptance, global SOTA or orbital accuracy claim.',
        frozen_surface=count(base), frozen_surface_sift=count(baseline_fallback), winner=summary['winner'],
        selected_seed_correct=[r['raw']['correct'] for r in selected], runs=results,
        training_points=10340, training_geometries=18, training_components=12,
        selection_geometries=8, selection_components=4, source_sha256=sha256(__file__),
        audit_sha256=sha256(audit_path), controls_sha256=sha256(controls_path), sift_results_sha256=sha256(sift_path),
        limits=['Only four exposed terrain components; descriptive intervals after hypothesis formation.',
            'Normalization changes geometry, interpolation, support and effective search extent together.',
            'All72 exposed selection pairs choose new checkpoints; older16-case cohorts are historical.',
            'No initial pose means no recovered pose; raw estimates are not accepted poses.',
            'No calibrated nonoverlap/risk, fresh final, transfer or full modern comparison.'])
    write_new(prefix.with_suffix('.json'), output)
    lines = ['# Geometry-normalized supervision: complete development results', '',
        f'Five runs completed {sum(r["completed_steps"] for r in results)} optimizer steps. The predefined selected configuration is `{summary["winner"]}`; its three seed results are **'+
        '/'.join(str(r['raw']['correct']) for r in selected)+' correct raw poses out of72**, versus **54/72** for frozen surface refinement. The unchanged SIFT-first baseline gives **57/72**.', '',
        '| Experiment | Seed | Selected step | Correct /72 | Strict /72 | Wrong raw poses | Same SIFT fallback | Parameters, total |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["experiment"]} | {r["seed"]} | {r["selected_step"]} | {r["raw"]["correct"]} | {r["raw"]["strict"]} | {r["raw"]["wrong"]} | {r["sift_fallback"]["correct"]} | {r["total_parameters"]} |')
    lines += ['', 'Each ablation receives2000 updates, eight visible examples per batch and the same learning rate, component/point/light/jitter sampling and sensor perturbation draws. Common support contains10340 exact visible integer-reference points across18 training geometries/12 components. Query truth constructs training normalization/visibility and scoring only; inference receives reference depth and an image-estimated query pose. Darkness does not create occlusion labels. Raw and normalized training differ in geometry, resampling and effective search extent.', '',
        'Checkpoint selection uses all72 exposed model-selection pairs, then2px-correct point count, then earlier step. Initialization is eligible. This new selection rule was frozen before fitting; earlier16-case results are preserved. No seed is substituted for a better-looking headline. The no-match head is absent from this precision-only ablation; reliable rejection remains unfinished.', '',
        'For each run, the JSON retains strict/primary counts, native correspondence support, four-component intervals, per-component results and all gained/lost cases. A paired interval containing zero remains inconclusive. These are four exposed components, not72 independent terrains. SIFT-first routing uses estimated-pose availability only, never correctness. The speed ratio measured for the frozen v11 model cannot be assigned to these new networks.', '',
        'Verification:193 repository tests/four optional skips, a corrected fresh100-row software smoke, actual initial equivalence on every historical case, independent train projection/visibility labels, all checkpoint native metrics and final pose-estimator replay, selected/final/optimizer/source hashes, exact sampler RNG replay and actual selected-weight quarter-turn/blank controls. This is saved-study reproducibility, not independent retraining or geographic validation.', '',
        'Earlier zero-step preparation with the feature-batch bug remains preserved. All failed checkpoints and seed outcomes are retained. Fresh-final, low-error calibrated acceptance/nonoverlap, required remaining native baselines, transfer and orbital navigation targets remain unmet. No paid compute, human review or automatic email.', '']
    path = prefix.with_suffix('.md')
    if path.exists():
        raise FileExistsError(path)
    path.write_text('\n'.join(lines))
    labels = ['Frozen surface']+[r['experiment'].replace('_', '\n')+'\n'+str(r['seed']) for r in results]
    raw = [count(base)]+[r['raw'] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
    for ax, metric, title in zip(axes, ('correct', 'strict', 'wrong'), ('Correct poses /72', 'Strict poses /72', 'Wrong raw poses')):
        values = [r[metric] for r in raw]; colors = ['#777777']+['#336d9d']*len(results)
        ax.bar(np.arange(len(values)), values, color=colors)
        ax.set_xticks(np.arange(len(values)), labels, rotation=40, ha='right', fontsize=8)
        ax.set_title(title); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.text(i, value+.3, str(value), ha='center', fontsize=9)
        ax.set_ylim(0, max(values)*1.17+1)
    fig.suptitle('Geometry-normalized training • exposed development only • four terrain components', fontsize=12)
    fig.savefig(prefix.with_suffix('.png'), dpi=160); plt.close(fig)
    print(json.dumps(output, indent=2)); return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', required=True)
    parser.add_argument('--audit', required=True); parser.add_argument('--controls', required=True)
    parser.add_argument('--prefix', required=True); args = parser.parse_args()
    report(args.run, args.audit, args.controls, args.prefix)
