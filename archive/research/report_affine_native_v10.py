"""Compare the audited engineering correction on identical development pairs."""
import gzip
import json
from pathlib import Path

from dataset import sha256
from lunargeo_dataset import write_new
from report_coarse_training_v5 import paired_interval


def report():
    root = Path('runs/localization-v10-affine-development')
    audit = json.loads(Path('reports/native_affine_audit_v10.json').read_text())
    if not audit['passed'] or audit['summary_sha256'] != sha256(root/'summary.json'):
        raise ValueError('Completed affine replay required')
    initial = Path('runs/localization-v7-native-initial-control/initial_results.json.gz')
    eq = json.loads(Path('reports/native_initial_equivalence_v7.json').read_text())
    if not eq['passed'] or eq['new_results_sha256'] != sha256(initial):
        raise ValueError('Audited equivalent frozen native baseline required')
    with gzip.open(initial, 'rt') as f:
        base = {r['pair_id']: r for r in json.load(f)}
    with gzip.open(root/'results.json.gz', 'rt') as f:
        candidate = {r['pair_id']: r for r in json.load(f)}
    source = Path('runs/localization-v5-axial-correction/results.json.gz')
    old_audit = json.loads(Path('reports/localization_v5_rescore_audit.json').read_text())
    if not old_audit['passed'] or old_audit['results_sha256'] != sha256(source):
        raise ValueError('Corrected SIFT replay required')
    with gzip.open(source, 'rt') as f:
        sift = {r['pair_id']: r for r in json.load(f) if r['method'] == 'sift_native' and r['solver'] == 'ap3p_reclassify'}
    if len(base) != 72 or set(base) != set(candidate) or set(sift) != set(base):
        raise ValueError('Incomplete common development cases')
    for key, r in sift.items():
        r['component'] = base[key]['component']; r['estimated_cam2world'] = r['pose']
    fallback_base = {i: sift[i] if sift[i]['pose'] is not None else base[i] for i in base}
    fallback_candidate = {i: sift[i] if sift[i]['pose'] is not None else candidate[i] for i in base}
    count = lambda rows: dict(cases=len(rows), correct=sum(r['pose_correct'] for r in rows.values()),
        strict=sum(r['strict_pose_correct'] for r in rows.values()),
        wrong=sum(r['estimated_cam2world'] is not None and not r['pose_correct'] for r in rows.values()))
    per_group = {group: dict(cases=sum(r['component'] == group for r in base.values()),
        frozen_correct=sum(r['pose_correct'] for r in base.values() if r['component'] == group),
        affine_correct=sum(r['pose_correct'] for r in candidate.values() if r['component'] == group))
        for group in sorted({r['component'] for r in base.values()})}
    result = dict(scope='Exposed-development deterministic preconditioning; not learned novelty, accepted-pose accuracy or independent validation.',
        frozen=count(base), affine=count(candidate), frozen_sift_fallback=count(fallback_base), affine_sift_fallback=count(fallback_candidate),
        gained_case_ids=[i for i in base if candidate[i]['pose_correct'] and not base[i]['pose_correct']],
        lost_case_ids=[i for i in base if base[i]['pose_correct'] and not candidate[i]['pose_correct']],
        raw_gain_pp=100*(count(candidate)['correct']-count(base)['correct'])/72,
        descriptive_component_interval_pp=paired_interval(candidate, base),
        fallback_descriptive_component_interval_pp=paired_interval(fallback_candidate, fallback_base), per_component=per_group,
        interval_note='Descriptive paired bootstrap on4 exposed selection components after hypothesis formation; not fresh-test confidence.',
        source_sha256=sha256(__file__), result_sha256=sha256(root/'results.json.gz'),
        frozen_results_sha256=sha256(initial), calibration_or_test_pixels_loaded=0)
    write_new('reports/native_affine_comparison_v10.json', result)
    Path('reports/AFFINE_NATIVE_V10.md').write_text(f'''# Viewpoint preconditioning is a useful engineering control

The frozen28,272-parameter matcher gains{result['raw_gain_pp']:.2f} percentage points,45→49 correct raw poses out of72, after coarse image matches estimate an affine warp before native descriptor refinement. The fixed16-case bank gave11/16 for both, with strict success5→6. Full strict success remains17/72. No query depth or true query pose is used in inference. This correction adds no learned parameters.

Raw wrong poses change12→11. With the identical SIFT-first engineering fallback, frozen gives{result['frozen_sift_fallback']['correct']}/72 and the affine pipeline gives{result['affine_sift_fallback']['correct']}/72, with{result['affine_sift_fallback']['wrong']} wrong proposals. These are uncalibrated outputs, not reliable accepted poses. The raw descriptive terrain interval is{result['descriptive_component_interval_pp']} percentage points; four dependent, exposed selection components cannot establish a fresh-test improvement.

The warp is estimated by affine RANSAC on coarse image matches, with12native-pixel support,2000 iterations,.99 confidence and at least6 inliers. Singular values outside[0.5,2] or reflection reject the warp and use standard refinement. A valid warp resamples the query into reference coordinates; a33pixel rectified search follows, then coordinates map back to original query pixels. This changes the search's physical extent and interpolation as well as appearance normalization; it does not isolate affine invariance alone. Every returned pose uses the unchanged3native-pixel AP3P estimator. No isolated runtime advantage is claimed during concurrent local experiments.

Independent replay checks the frozen checkpoint, every input/source hash, all native correspondence/pose errors and branch selection, then reconstructs reference3D points independently and reruns the pose estimator. Actual frozen-model512pixel translation and blank-image controls pass. All eligible72 cases remain in the denominator. Source data: MoonAnything LunarGeo, Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0; own code/weights MIT.

The corresponding learned native study gives45/42/42 across its selected model's seeds, versus45 frozen. Thus this engineering correction is a stronger control for future learning claims. Uniform-positive training remains a separate live experiment. Fresh validation, calibrated acceptance, modern comparison and verified orbital navigation are not established. No paid compute, human review or email.

Reproduce: `evaluate_affine_native_v10.py --full --output runs/new-affine`, then `verify_affine_native_v10.py --run runs/new-affine --output runs/new-affine-audit.json`. Every old result is preserved. Full per-component and gained/lost case records are in native_affine_comparison_v10.json.
''')
    print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    report()
