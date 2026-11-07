"""Report completed initial/trained/motion-ablation common development outcomes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dataset import sha256
from uncertainty_groups import geographic_groups


def completed(root):
    root = Path(root)
    end = json.loads((root/'COMPLETE.json').read_text())
    for name in ['frozen_inputs', 'results']:
        if sha256(root/f'{name}.json') != end[f'{name}_sha256']:
            raise ValueError('Changed completed outcomes/design')
    rows = json.loads((root/'results.json').read_text())
    if end['rows'] != len(rows):
        raise ValueError('Incorrect completed row count')
    values = {(r['pair_id'], r['method']): r for r in rows}
    if len(values) != len(rows):
        raise ValueError('Duplicate outcomes')
    frozen = json.loads((root/'frozen_inputs.json').read_text())
    expected = {(c['pair_id'], m) for c in frozen['cases'] for m in frozen['methods']}
    if set(values) != expected or frozen['split'] != 'dev':
        raise ValueError('Incomplete or non-development comparison')
    return frozen, values


def report(augmented, nomotion, output):
    output = Path(output)
    af, ar = completed(augmented)
    nf, nr = completed(nomotion)
    if af['cases'] != nf['cases'] or af['manifest'] != nf['manifest']:
        raise ValueError('Training ablations evaluated different development cases')
    a, n = af['training']['metadata'], nf['training']['metadata']
    if not a['augmented'] or n['augmented'] or a['seed'] != n['seed'] or a['steps'] != n['steps']:
        raise ValueError('Unexpected augmentation control')
    if a['fixed_development_bank_sha256'] != n['fixed_development_bank_sha256'] or af['training']['completed']['initial_sha256'] != nf['training']['completed']['initial_sha256']:
        raise ValueError('Initialization or validation bank differs between training ablations')
    predictions = {
        'Initial structural CNN': {p: r for (p, m), r in ar.items() if m == 'contrastive_initial'},
        'Trained, no motion augmentation': {p: r for (p, m), r in nr.items() if m == 'contrastive_selected'},
        'Trained, motion augmentation': {p: r for (p, m), r in ar.items() if m == 'contrastive_selected'}}
    # Every initial decision should be identical across the two independent
    # executions; timing is deliberately excluded from this deterministic check.
    keys = ['input_array_sha256', 'transform', 'reverse_transform', 'support', 'correct', 'reason']
    for (p, method), row in ar.items():
        if method == 'contrastive_initial' and any(row[k] != nr[(p, method)][k] for k in keys):
            raise ValueError('Initial-checkpoint decision differs between ablation executions')
    baselines = [('LoFTR', 'runs/geometry-dev-v1', 'loftr_outdoor'),
                 ('Structural search', 'runs/geometry-dev-v1', 'self_similarity_search'),
                 ('Earlier image-selected hybrid', 'runs/blend-dev-v1', 'blend_global')]
    bindings = {}
    for label, root, method in baselines:
        frozen, values = completed(root)
        if frozen['cases'] != af['cases'] or frozen['manifest'] != af['manifest']:
            raise ValueError('Baseline common case plan differs')
        predictions[label] = {p: r for (p, m), r in values.items() if m == method}
        bindings[label] = json.loads((Path(root)/'COMPLETE.json').read_text())
    for rows in predictions.values():
        if rows.keys() != predictions['Initial structural CNN'].keys():
            raise ValueError('Missing common case')
        for p, row in rows.items():
            if row['input_array_sha256'] != predictions['Initial structural CNN'][p]['input_array_sha256']:
                raise ValueError('Different native images/masks in a compared method')
    summary = []
    families = ['translation', 'rotation', 'scale', 'combined']
    for label, rows in predictions.items():
        positive = [r for r in rows.values() if r['label_source'] == 'exact_geometry']
        negative = [r for r in rows.values() if r['label_source'] == 'known_nonoverlap']
        groups = {family: sum(r['correct'] for r in positive if r['geometry_family'] == family) for family in families}
        summary.append(dict(method=label, correct=sum(r['correct'] for r in positive), positive_cases=len(positive),
            success_percent=100*sum(r['correct'] for r in positive)/len(positive),
            positive_fits=sum(r['estimated'] for r in positive), negative_fits=sum(r['estimated'] for r in negative),
            negative_cases=len(negative), correct_by_geometry=groups,
            median_forward_ms=None if not all('forward_ms' in r for r in positive) else float(np.median([r['forward_ms'] for r in positive]))))
    tiles = [t for t in af['manifest']['tiles'] if t['split']=='dev']
    _, components, overlaps = geographic_groups(tiles)
    result = dict(schema_version=1, summary=summary, training={
        'augmented': dict(selected_step=a['selected_step'], initial_loss=a['history'][0]['development_descriptor_loss'],
            selected_loss=min(h['development_descriptor_loss'] for h in a['history'])),
        'no_motion': dict(selected_step=n['selected_step'], initial_loss=n['history'][0]['development_descriptor_loss'],
            selected_loss=min(h['development_descriptor_loss'] for h in n['history']))},
        common_case_input_bytes_verified=True, identical_initial_predictions_verified=True,
        development_geographic_components=len(components), overlapping_development_pairs=overlaps,
        source_run_completions={str(p): json.loads((Path(p)/'COMPLETE.json').read_text()) for p in [augmented, nomotion]},
        baseline_completions=bindings, report_source_sha256=sha256(__file__),
        warning='Development-only raw fit success. No calibrated rejection or current-SOTA superiority is established. Baseline timing excludes upstream candidate generation and is not directly comparable.')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    colors = ['#88939f', '#d89e39', '#327c9b']
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    x = np.arange(4)
    for index, entry in enumerate(summary[:3]):
        ax.bar(x+(index-1)*.25, [100*entry['correct_by_geometry'][f]/288 for f in families],
            width=.24, color=colors[index], label=entry['method'])
    ax.set_xticks(x, ['Translation', 'Rotation', 'Scale', 'Combined'])
    ax.set_ylim(0, 105)
    ax.set_ylabel('Correct alignments (%)')
    ax.set_title('Dense structural descriptor: development ablation')
    ax.legend(fontsize=8, loc='lower left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(.99, .98, '288 pairs per family · 128×128 · ≤3px median error\nRepeated terrain; no fresh geographic test',
        transform=ax.transAxes, ha='right', va='top', fontsize=8)
    fig.savefig(output.with_suffix('.png'), dpi=180)
    plt.close(fig)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--augmented', default='runs/contrastive-terrain-dev-v1')
    p.add_argument('--nomotion', default='runs/contrastive-terrain-nomotion-dev-v1')
    p.add_argument('--output', default='reports/contrastive_terrain_development.json')
    args = p.parse_args()
    report(args.augmented, args.nomotion, args.output)
