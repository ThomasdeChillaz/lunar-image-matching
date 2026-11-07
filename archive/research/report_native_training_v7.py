"""Report every completed native ablation and seed against strong simple controls."""
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


def report():
    root = Path('runs/localization-v7-native-development')
    summary = json.loads((root/'summary.json').read_text())
    audit_path = Path('reports/native_training_audit_v7.json')
    audit = json.loads(audit_path.read_text())
    if not audit['passed'] or audit['trained_runs'] != 5 or audit['summary_sha256'] != sha256(root/'summary.json'):
        raise ValueError('Independently audited complete native cohort required')
    source = Path('runs/localization-v5-axial-correction/results.json.gz')
    source_audit = json.loads(Path('reports/localization_v5_rescore_audit.json').read_text())
    if not source_audit['passed'] or source_audit['results_sha256'] != sha256(source):
        raise ValueError('Corrected baseline replay required')
    with gzip.open(source, 'rt') as f:
        rows = json.load(f)
    manifest = json.loads(Path('reports/localization_training_data_v5.json').read_text())
    components = {t['tile_id']: t['component'] for t in manifest['tiles']}
    ids = {c['pair_id'] for c in manifest['cases'] if c['split'] == 'model_selection'}
    frozen = {r['pair_id']: r for r in rows if r['method'] == 'quarter_turn_bank' and r['selected']}
    sift = {r['pair_id']: r for r in rows if r['method'] == 'sift_native' and r['solver'] == 'ap3p_reclassify'}
    for c in manifest['cases']:
        if c['pair_id'] in ids and c['pair_id'] not in frozen:
            frozen[c['pair_id']] = dict(c, pose=None, pose_correct=False, strict_pose_correct=False)
    if len(ids) != 72 or set(frozen) != ids or set(sift) != ids:
        raise ValueError('Incomplete common baseline')
    for collection in (frozen, sift):
        for r in collection.values():
            r['component'] = components[r['tile_id']]
    fallback = {i: sift[i] if sift[i]['pose'] is not None else frozen[i] for i in ids}
    count = lambda collection: sum(r['pose_correct'] for r in collection.values())
    table = []
    for entry in summary['results']:
        folder = root/f"{entry['variant']}-{entry['seed']}"
        with gzip.open(folder/'development_results.json.gz', 'rt') as f:
            candidate_rows = json.load(f)
        candidate = {r['pair_id']: r for r in candidate_rows}
        if len(candidate_rows) != 72 or set(candidate) != ids:
            raise ValueError('Incomplete candidate denominator')
        mixed = {i: sift[i] if sift[i]['pose'] is not None else candidate[i] for i in ids}
        wrong = sum(r['estimated_cam2world'] is not None and not r['pose_correct'] for r in candidate.values())
        fallback_wrong = sum((r.get('pose') is not None or r.get('estimated_cam2world') is not None) and not r['pose_correct'] for r in mixed.values())
        table.append(dict(variant=entry['variant'], seed=entry['seed'], selected_step=entry['selected_step'],
            correct=count(candidate), strict=sum(r['strict_pose_correct'] for r in candidate.values()), wrong=wrong,
            sift_fallback_correct=count(mixed), sift_fallback_wrong=fallback_wrong,
            raw_gain_vs_frozen_pp=100*(count(candidate)-count(frozen))/72,
            descriptive_component_interval_pp=paired_interval(candidate, frozen),
            fallback_gain_vs_frozen_fallback_pp=100*(count(mixed)-count(fallback))/72,
            fallback_descriptive_component_interval_pp=paired_interval(mixed, fallback),
            selected_checkpoint_sha256=sha256(folder/'selected.pt'), results_sha256=sha256(folder/'development_results.json.gz')))
    result = dict(scope='Completed exposed-development native study; four dependent selection components. No calibrated acceptance, fresh final or SOTA claim.',
        optimizer_steps=10000, winner=summary['winner'], frozen_correct=count(frozen),
        frozen_fallback_correct=count(fallback), frozen_fallback_wrong=sum(r['pose'] is not None and not r['pose_correct'] for r in fallback.values()),
        runs=table, source_sha256=sha256(__file__), interval_implementation_sha256=sha256('report_coarse_training_v5.py'),
        summary_sha256=sha256(root/'summary.json'), audit_sha256=sha256(audit_path), baseline_results_sha256=sha256(source),
        interval_note='Descriptive paired terrain-component bootstrap after model selection, only4 components; not fresh-test confidence.',
        calibration_or_test_pixels_loaded=0)
    write_new('reports/native_training_comparison_v7.json', result)
    selected = [r for r in table if r['variant'] == summary['winner']]
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    x = np.arange(1+len(selected)); width = .36
    ax.bar(x-width/2, [count(frozen)]+[r['correct'] for r in selected], width, color='#597b99', label='Native orientation matcher')
    ax.bar(x+width/2, [count(fallback)]+[r['sift_fallback_correct'] for r in selected], width, color='#228c7f', label='Same SIFT-first fallback')
    ax.set(xticks=x, xticklabels=['Frozen']+[str(r['seed']) for r in selected], ylim=(0, 72),
        ylabel='Correct raw poses / all72 development pairs', title='Native learned refinement: every selected-configuration seed')
    ax.axhline(.7*72, linestyle=':', color='#955a49', label='70% target before rejection')
    ax.legend(fontsize=8, loc='lower right')
    fig.savefig('reports/native_training_outcomes_v7.png', dpi=170); plt.close(fig)
    lines = '\n'.join(f"| {r['variant']} | {r['seed']} | {r['selected_step']} | {r['correct']} | {r['strict']} | {r['wrong']} | {r['sift_fallback_correct']} | {r['sift_fallback_wrong']} |" for r in table)
    Path('reports/NATIVE_TRAINING_V7.md').write_text(f'''# Native refinement: completed development study

All three2000-step primary ablations and both predefined winner repeats completed:10000 optimizer steps. Each uses corrected axial-depth3D labels, a frozen coarse matcher, native512px image features and the same pose estimator. Selection uses the fixed16-case bank; all72 expanded readouts remain dependent, exposed development evidence from eight geometries/four terrain components.

| Variant | Seed | Selected step | Correct /72 | Strict /72 | Wrong raw poses | Correct with SIFT /72 | Wrong with SIFT |
|---|---:|---:|---:|---:|---:|---:|---:|
{lines}

Frozen native orientation matching gives45/72, strict17, with12 wrong estimates. The stronger simple SIFT-first fallback gives53/72 with six wrong estimates. Every candidate receives exactly that same fallback decision: use SIFT when its estimator returns a pose, otherwise the candidate. This engineering rule was developed from exposed results; neither raw recovery nor fallback is calibrated acceptance. No query truth selects branches or fallback outputs.

![Every selected-configuration seed](native_training_outcomes_v7.png)

The short100-step47/72 pilot is retained separately and is not the primary result. The learned dustbin ablation selects its initial checkpoint; trained versions reject essentially every point. Its failure remains published. Training loss and small correspondence-count gains do not establish a useful pose gain. All seed outcomes and descriptive component intervals are in native_training_comparison_v7.json. Those intervals use only four exposed components after selection, so they are not independent final-test confidence.

The uniform-positive coverage ablation is a separate subsequent experiment. Actual native LoMa-R and RoMa-v2 comparisons must finish all72 assigned cases and paired compact controls before any ranking. Their partial outputs are incomplete. The fresh final-data gate, reliable acceptance, transfer and paper targets remain unmet. Real orbital navigation accuracy is unverified. No human review, paid compute or automatic outreach.

Independent replay verifies360 candidate rows,234 cached supervision cases, source/weight hashes, actual optimizer completion and changes, balanced sampling/RNG, selection and every seed. Reproduce with `verify_native_training_v7.py` and this reporting command. Own code is MIT; source data are MoonAnything LunarGeo, Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0.
''')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    report()
