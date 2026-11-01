"""Report native-window ablations, all seeds and exposed-development limitations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new
from report_coarse_training_v5 import paired_interval
from report_surface_learning_v13 import count, read_rows


def report(root, audit_path, prefix):
    root, prefix = Path(root), Path(prefix)
    audit = json.loads(Path(audit_path).read_text()); protocol = json.loads((root/'protocol.json').read_text())
    if not audit['passed'] or audit['protocol_sha256'] != sha256(root/'protocol.json'):
        raise ValueError('Executed cohort audit required')
    base = read_rows('runs/localization-v11-surface-warp-development/results.json.gz')
    if len(base) != 72 or protocol['pilot']:
        raise ValueError('Principal report requires all72 exposed selection pairs')
    sift_path = Path('runs/localization-v5-axial-correction/results.json.gz')
    sift_audit = json.loads(Path('reports/localization_v5_rescore_audit.json').read_text())
    if not sift_audit['passed'] or sift_audit['results_sha256'] != sha256(sift_path):
        raise ValueError('Audited corrected SIFT baseline required')
    sift = {i:r for i,r in read_rows_sift(sift_path).items()}
    for pair_id, row in sift.items():
        row['component'] = base[pair_id]['component']; row['estimated_cam2world'] = row['pose']
    if set(sift) != set(base):
        raise ValueError('SIFT/common cases differ')
    fallback = lambda rows: {i:sift[i] if sift[i]['pose'] is not None else r for i,r in rows.items()}
    base_fallback = fallback(base); records = []
    for saved in audit['runs']:
        folder = root/f'{saved["variant"]}-{saved["seed"]}'
        meta = json.loads((folder/'training.json').read_text())
        rows = read_rows(folder/f'selection_step_{meta["selected_step"]}.json.gz')
        if set(rows) != set(base):
            raise ValueError('Different eligible pair denominator')
        mixed = fallback(rows)
        records.append(dict(saved, raw=count(rows), sift_fallback=count(mixed),
            total_parameters=meta['total_parameters'], trainable_parameters=meta['trainable_parameters'],
            raw_gain_pp=100*(count(rows)['correct']-count(base)['correct'])/72,
            raw_interval_pp=paired_interval(rows,base), fallback_interval_pp=paired_interval(mixed,base_fallback),
            gained=[i for i in base if rows[i]['pose_correct'] and not base[i]['pose_correct']],
            lost=[i for i in base if base[i]['pose_correct'] and not rows[i]['pose_correct']],
            per_component={g:count({i:r for i,r in rows.items() if r['component'] == g}) for g in sorted({r['component'] for r in base.values()})},
            per_lighting_id={str((a,b)):count({i:r for i,r in rows.items() if (r['light_a'],r['light_b']) == (a,b)}) for a in range(3) for b in range(3)},
            validation=meta['validation'],checkpoint_sha256=sha256(folder/'selected.pt'),
            selected_rows_sha256=sha256(folder/f'selection_step_{meta["selected_step"]}.json.gz')))
    summary = None if audit['partial'] else json.loads((root/'summary.json').read_text())
    result = dict(scope='Exposed development-only native-window training; no calibrated acceptance, fresh generalization or SOTA claim.',
        partial=audit['partial'], frozen_surface=count(base), frozen_surface_sift=count(base_fallback),
        winner=None if summary is None else summary['winner'], runs=records,
        source_sha256=sha256(__file__), audit_sha256=sha256(audit_path),
        protocol_sha256=sha256(root/'protocol.json'),
        intervals='Descriptive paired bootstrap across four previously exposed components; checkpoint/configuration selection reuse these cases. Not confirmatory inference.',
        limits=['All72 pairs are dependent views of eight geometries in four exposed components.',
            'Wrong raw estimates are not calibrated accepted poses;1% error/nonoverlap is not established.',
            'Frozen native CNN caches accelerate repeated evaluation; fitting seconds are not isolated inference latency.',
            'Shared renderer/batch/trajectory dependence and failed fresh-final data gate remain.',
            'Required remaining native baselines, transfer, reliable acceptance and full paper remain incomplete.'])
    write_new(prefix.with_suffix('.json'),result)
    lines = ['# Native-window supervision results','',
        ('**Partial cohort: later configurations/repeat seeds are not yet included.**' if audit['partial'] else '**Complete predefined ablations and repeat seeds.**'),'',
        'All readouts are exposed development results:72 pairs from four terrain components. Frozen surface gives54/72 correct raw poses; the stronger same SIFT-first baseline gives57/72. Raw estimates are not calibrated accepted navigation poses.','',
        '| Variant | Seed | Selected update | Correct /72 | Strict /72 | Wrong raw | Same SIFT fallback | Total parameters |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["variant"]} | {r["seed"]} | {r["selected_step"]} | {r["raw"]["correct"]} | {r["raw"]["strict"]} | {r["raw"]["wrong"]} | {r["sift_fallback"]["correct"]} | {r["total_parameters"]} |')
    lines += ['','Each primary variant receives2,000 updates and the same component/light/rotation/augmentation draws. The controls separate search-window labels, genuine-match anchors, coarse-encoder adaptation and cross-image context. The native fine encoder, reference-surface iterations and3px pose estimator stay frozen; coarse adaptation changes trainable capacity, disclosed separately. Initialization is eligible; the protocol selects on unfiltered pose count, then2px point count, then earlier update. Repeat seeds belong only to the selected primary configuration.','',
        'Intervals in the JSON describe paired terrain-component resampling after hypothesis, model and checkpoint selection on these same exposed cases. A lower endpoint at zero is inconclusive. Lighting variants are not independent terrain tests. A raw-pose gain does not establish1% accepted-error or nonoverlap risk.','',
        'The SIFT-first comparison uses available estimated-pose status only, never correctness. The separately measured historical CPU speed ratio cannot be assigned to a new model. No final-test, orbital-accuracy, novelty or SOTA claim follows from this study.','',
        f'Audit: {audit["replayed_pose_rows"]} saved checkpoint rows and {audit["recorded_updates"]} sampler/optimizer-update records; exact retrained updates: {audit["exact_retrained_steps"]}. All failed and selected outcomes are retained.','']
    if prefix.with_suffix('.md').exists() or prefix.with_suffix('.png').exists():
        raise FileExistsError('Preserve every report snapshot')
    prefix.with_suffix('.md').write_text('\n'.join(lines))
    figure,axes = plt.subplots(1,3,figsize=(15,4.7))
    primary = [r for r in records if r['seed'] == 20261101]
    for r in primary:
        axes[0].plot([v['step'] for v in r['validation']], [v['selection']['correct_poses'] for v in r['validation']], marker='o',label=r['variant'].replace('_',' '))
    axes[0].axhline(54,color='grey',linestyle='--',label='frozen surface')
    axes[0].set(xlabel='Optimizer updates',ylabel='Correct raw poses /72',title='Primary ablations; initialization eligible');axes[0].legend(fontsize=8)
    labels = [r['variant'].replace('_',' ')+'\n'+str(r['seed']) for r in records];x=np.arange(len(records))
    axes[1].bar(x-.18,[r['raw']['correct'] for r in records],.36,label='correct raw')
    axes[1].bar(x+.18,[r['raw']['wrong'] for r in records],.36,label='wrong raw')
    axes[1].axhline(57,color='grey',linestyle='--',label='stronger SIFT baseline')
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Pairs /72',title='All audited runs; no best-seed headline');axes[1].tick_params(axis='x',labelsize=7);axes[1].legend(fontsize=8)
    gains=np.array([r['raw_gain_pp'] for r in records]);bounds=np.array([r['raw_interval_pp'] for r in records])
    axes[2].errorbar(x,gains,yerr=np.stack([gains-bounds[:,0],bounds[:,1]-gains]),fmt='o',capsize=4)
    axes[2].axhline(0,color='grey',linestyle='--');axes[2].set(xticks=x,xticklabels=labels,ylabel='Raw gain over frozen surface (points)',title='Descriptive four-component intervals');axes[2].tick_params(axis='x',labelsize=7)
    figure.suptitle('Native-window development study — four exposed terrain components'+(' — PARTIAL' if audit['partial'] else ''))
    figure.tight_layout();figure.savefig(prefix.with_suffix('.png'),dpi=180);plt.close(figure)
    return result


def read_rows_sift(path):
    import gzip
    with gzip.open(path,'rt') as stream:
        return {r['pair_id']:r for r in json.load(stream) if r['method'] == 'sift_native' and r['solver'] == 'ap3p_reclassify'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    parser.add_argument('--audit',required=True);parser.add_argument('--prefix',required=True);args=parser.parse_args()
    report(args.run,args.audit,args.prefix)
