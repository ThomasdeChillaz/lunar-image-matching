"""Preserve the failed broad-initialization control and its global-match diagnosis."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new


def report():
    root = Path('runs/localization-v14-broad-recovery-development')
    diagnostic = Path('runs/localization-v14-missing-branch-diagnosis')
    audit = json.loads(Path('reports/native_recovery_audit_v14.json').read_text())
    branch_audit = json.loads(Path('reports/native_missing_branch_audit_v14.json').read_text())
    if (not audit['passed'] or not branch_audit['passed'] or audit['summary_sha256'] != sha256(root/'summary.json')
            or branch_audit['summary_sha256'] != sha256(diagnostic/'summary.json')):
        raise ValueError('Both completed independent replays required')
    result = json.loads((root/'summary.json').read_text()); details = json.loads((diagnostic/'summary.json').read_text())
    baseline = json.loads(Path('reports/native_surface_comparison_v11.json').read_text())['surface']
    values = [r['max_true_12px_matches'] for r in details['per_case']]
    fractions = [r['max_true_12px_fraction'] for r in details['per_case']]
    output = dict(scope='Failed development engineering control; do not adopt. Not a learned, calibrated-acceptance or fresh-test result.',
        baseline=baseline, candidate=result, diagnostic=details,
        pairs_with_fewer_than_six_true_12px_matches=sum(v < 6 for v in values),
        median_best_true_12px_fraction=float(np.median(fractions)),
        decision='Reject broad12px initialization as a recovery improvement. It adds fifteen wrong poses and no correct poses despite strict3px final fits.',
        next_hypothesis='Improve global correspondence discrimination. Test a small pair-context module with an exact-identity frozen initialization and capacity controls; do not infer a gain from software readiness.',
        source_sha256=sha256(__file__), audit_sha256=sha256('reports/native_recovery_audit_v14.json'),
        branch_audit_sha256=sha256('reports/native_missing_branch_audit_v14.json'))
    write_new('reports/native_recovery_comparison_v14.json', output)
    text = f'''# Broader initialization fails: global correspondence quality is the bottleneck

The frozen surface pipeline has **54/72 correct raw poses and three wrong poses**.
A12pixel initial RANSAC proposal followed by the unchanged3pixel surface
refinement returns **54/72 correct and18 wrong poses**. All15 formerly missing
cases now yield an estimate; **none is correct**. Strict success stays16/72.
This control is rejected as an improvement. It does not change the default
pipeline or any historical result. Returning a pose is not reliable acceptance.

The rule preserves every historical estimated result. On missing cases only,
it fits all four orientation branches with12pixel RANSAC, chooses by geometric
support/residual and runs the same two reference-surface updates. A broad pose
is never returned unless at least one unchanged3pixel update succeeds. Unit
tests verify that distinction; the final estimator independently replays on
every case. Nevertheless, a coherent fit to incorrect image matches is wrong.
The selected broad proposals have rotation errors of about25–177degrees.

The fixed16case pilot also adds three wrong estimates and zero correct ones.
The expanded72case denominator is retained. These are dependent, exposed
development cases across eight geometries/four components. No new pixels from
calibration or final testing are used; no1% risk or real navigation claim.

A separate diagnostic scores all **60 coarse/broad branches on the15 missing
cases** after prediction. Even oracle branch choice finds **0/15 correct broad
poses**. The best genuine12pixel match fraction has median
**{100*np.median(fractions):.2f}%**; {sum(v < 6 for v in values)}/15 cases contain
fewer than six genuine12pixel matches in every branch. This rules out branch
selection alone as a fix for these particular broad proposals. It does not
prove every better proposal algorithm impossible. Oracle counts are diagnosis
only and never choose an inference output. Native ground truth comes from
audited cameras/depth; darkness is not an occlusion label.

The next focused hypothesis is to improve global match discrimination before
local refinement. A small attention/context experiment is being prepared with
an exactly frozen starting score matrix and pointwise/self/cross controls.
Generic attention is not new architectural science; any contribution requires
actual learning gains, seed stability, a strong compact baseline and fresh
validation. The separate v13 fine-refinement cohort remains unchanged and live.

Verification:197 repository tests/four optional skips and a fresh100row
software smoke passed after the recovery kernel was added. Actual pretrained
quarter-turn/blank controls pass. Independent audits replay72 final3pixel fits,
the16case pilot and all60 broad fits, native point/pose errors, availability-only
routing, source/input/checkpoint hashes and summaries. Timing was concurrent
with training and supports no speed claim. No paid compute, human review or
automatic outreach.

Reproduce with `evaluate_pose_recovery_v14.py --full --output runs/new-recovery`,
then `verify_pose_recovery_v14.py --run runs/new-recovery --output runs/new-audit.json`.
For branch diagnosis use `diagnose_missing_poses_v14.py --output runs/new-branches`
and `verify_missing_poses_v14.py --run runs/new-branches --output runs/new-branch-audit.json`.
MoonAnything LunarGeo: Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0.
Own code/checkpoint MIT. This is a negative development finding.
'''
    path = Path('reports/POSE_RECOVERY_V14.md')
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar([0, 1], [54, 54], color='#3979a0', label='Correct')
    axes[0].bar([0, 1], [3, 18], bottom=[54, 54], color='#ba554d', label='Wrong')
    axes[0].bar([0, 1], [15, 0], bottom=[57, 72], color='#cccccc', label='No pose')
    axes[0].set_xticks([0, 1], ['Frozen surface', 'Broad initialization\n+ strict refinement'])
    axes[0].set_ylim(0, 78); axes[0].set_ylabel('All72 eligible development pairs')
    axes[0].set_title('More outputs, no additional correct poses'); axes[0].legend(fontsize=8)
    labels = [r['pair_id'].split('/')[-2]+'/'+r['pair_id'].split('/')[-1].removeprefix('lights-') for r in details['per_case']]
    axes[1].bar(np.arange(len(values)), values, color='#3979a0')
    axes[1].axhline(6, color='#ba554d', linestyle='--', label='Six correspondences')
    axes[1].set_xticks(np.arange(len(values)), labels, rotation=60, ha='right', fontsize=8)
    axes[1].set_ylabel('Genuine native matches within12px'); axes[1].set_title('Best of four branches • oracle diagnosis only')
    axes[1].legend(fontsize=8)
    fig.suptitle('Failed recovery control • exposed development, no new learning', fontsize=12)
    fig.savefig('reports/native_recovery_failure_v14.png', dpi=160); plt.close(fig)
    print(json.dumps({k: v for k, v in output.items() if k != 'diagnostic'}, indent=2)); return output


if __name__ == '__main__':
    report()
