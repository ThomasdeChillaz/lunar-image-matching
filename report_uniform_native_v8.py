"""Completed positive-sampling ablation: every configuration and seed."""
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
    with gzip.open(path, 'rt') as f:
        rows = json.load(f)
    result = {r['pair_id']: r for r in rows}
    if len(rows) != 72 or len(result) != 72:
        raise ValueError('All72 unique eligible cases required')
    return result


def report():
    root = Path('runs/localization-v8-uniform-native-development')
    summary = json.loads((root/'summary.json').read_text())
    audit = json.loads(Path('reports/native_uniform_training_audit_v8.json').read_text())
    if not audit['passed'] or audit['trained_runs'] != 5 or audit['summary_sha256'] != sha256(root/'summary.json'):
        raise ValueError('Completed independently audited uniform cohort required')
    original = Path('runs/localization-v7-native-development')
    old_audit = json.loads(Path('reports/native_training_audit_v7.json').read_text())
    if not old_audit['passed'] or old_audit['summary_sha256'] != sha256(original/'summary.json'):
        raise ValueError('Completed prior cohort replay required')
    initial = Path('runs/localization-v7-native-initial-control/initial_results.json.gz')
    equivalence = json.loads(Path('reports/native_initial_equivalence_v7.json').read_text())
    if sha256(initial) != equivalence['new_results_sha256']:
        raise ValueError('Frozen initial baseline differs')
    frozen = read_rows(initial); table = []
    for entry in summary['results']:
        folder = root/f"{entry['variant']}-{entry['seed']}"
        candidate = read_rows(folder/'development_results.json.gz')
        previous = read_rows(original/folder.name/'development_results.json.gz')
        if set(candidate) != set(frozen) or set(previous) != set(frozen):
            raise ValueError('Sampling ablation case identities differ')
        n = sum(r['pose_correct'] for r in candidate.values())
        old_n = sum(r['pose_correct'] for r in previous.values())
        table.append(dict(variant=entry['variant'], seed=entry['seed'], selected_step=entry['selected_step'],
            correct=n, strict=sum(r['strict_pose_correct'] for r in candidate.values()),
            wrong=sum(r['estimated_cam2world'] is not None and not r['pose_correct'] for r in candidate.values()),
            previous_same_variant_seed_correct=old_n, sampling_gain_pp=100*(n-old_n)/72,
            sampling_descriptive_component_interval_pp=paired_interval(candidate, previous),
            versus_frozen_descriptive_component_interval_pp=paired_interval(candidate, frozen),
            selected_checkpoint_sha256=sha256(folder/'selected.pt'), results_sha256=sha256(folder/'development_results.json.gz')))
    result = dict(scope='Completed development-only positive coverage/centering ablation. Not a fresh test, reliable acceptance or architectural gain.',
        winner=summary['winner'], optimizer_steps=10000, runs=table,
        frozen_correct=sum(r['pose_correct'] for r in frozen.values()),
        stronger_surface_sift_correct=57, stronger_surface_sift_wrong=2,
        sampling_change='Uniform exact projected positives across12 train components, with query-centered +/-12px jitter; original negatives, architecture, optimizer and native inference unchanged. Coverage and positive centering both change.',
        interval_note='Paired descriptive bootstrap on4 exposed selection components after model selection, not independent final-test confidence.',
        source_sha256=sha256(__file__), summary_sha256=sha256(root/'summary.json'), calibration_or_test_pixels_loaded=0)
    write_new('reports/native_uniform_training_comparison_v8.json', result)
    selected = [r for r in table if r['variant'] == summary['winner']]
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained'); x = np.arange(len(selected)); w = .36
    ax.bar(x-w/2, [r['previous_same_variant_seed_correct'] for r in selected], w, color='#597b99', label='Previous coarse-filtered positives')
    ax.bar(x+w/2, [r['correct'] for r in selected], w, color='#218c7f', label='Uniform projected positives')
    ax.axhline(result['frozen_correct'], color='#a57652', linestyle=':', label='Frozen native baseline')
    ax.set(xticks=x, xticklabels=[str(r['seed']) for r in selected], ylim=(0, 72), ylabel='Correct raw poses / all72 development pairs',
        title='Broader positive sampling: modest gains, every selected-model seed')
    ax.legend(fontsize=8, loc='lower right'); fig.savefig('reports/native_uniform_training_v8.png', dpi=170); plt.close(fig)
    lines = '\n'.join(f"| {r['variant']} | {r['seed']} | {r['selected_step']} | {r['previous_same_variant_seed_correct']} | {r['correct']} | {r['strict']} | {r['wrong']} |" for r in table)
    Path('reports/UNIFORM_NATIVE_V8.md').write_text(f'''# Broader positive sampling helps modestly; the stronger control remains ahead

All five2000-step runs completed:10000 optimizer steps. The bank-selected pair-refinement seeds give **47/45/47** correct raw poses out of72, versus **45/42/42** with the previous positive sampler and **45** for the frozen native baseline. The encoder-only primary gives48/72 but is not substituted for the predefined selected configuration or primary seed. The failed no-match model again selects initialization after later checkpoints reject essentially everything.

| Variant | Seed | Selected step | Previous /72 | Uniform /72 | Strict /72 | Wrong raw poses |
|---|---:|---:|---:|---:|---:|---:|
{lines}

![All selected-model seeds](native_uniform_training_v8.png)

The change broadens positive location coverage across all12 training components and centers positive query patches around exact projections with jitter. Both coverage and centering change; this does not isolate one factor. Architecture, original negative sampler, optimizer/budget, sensor augmentations and native inference are unchanged. Real rendered lighting provides the physical appearance changes. Query geometry is training supervision/scoring only, never an inference input.

Gains over the previous trained configuration do not establish a large gain over the frozen model or the stronger deterministic surface control. That control gives54/72, or57/72 with the same SIFT-first fallback and two wrong outputs. Its inference pipeline differs and must be an explicit strong control for any future learned method. The uniform learned models retain many wrong poses and are not calibrated acceptance. All paired descriptive four-component intervals are in native_uniform_training_comparison_v8.json; repeated lighting pairs are dependent and this study cannot establish fresh generalization.

Independent replay validates all360 candidate results,234 cached supervision cases plus10463 uniform projected training locations, source/checkpoint hashes, initialization, actual weight/optimizer changes, exact sampler/RNG replay, checkpoint selection and all predefined seeds. Actual selected-model quarter-turn/blank controls pass for all five models. No calibration/test pixels, human review, paid compute, automatic outreach, SOTA or verified orbital accuracy claim. Own code/checkpoints MIT; MoonAnything LunarGeo data: Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0.

Replay with `verify_native_training_v8.py --run runs/localization-v8-uniform-native-development --output runs/new-uniform-audit.json`. The publication report preserves checked-in outputs. The next focused hypothesis is geometry-normalized supervision, assessed against the stronger deterministic pipeline, rather than adding another head to unnormalized features.
''')
    print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    report()
