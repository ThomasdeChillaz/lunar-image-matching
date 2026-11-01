"""Report geometry normalization against every preserved simple control."""
import gzip
import json
from pathlib import Path

from dataset import sha256
from lunargeo_dataset import write_new
from report_coarse_training_v5 import paired_interval


def report():
    root = Path('runs/localization-v11-surface-warp-development')
    audit = json.loads(Path('reports/native_surface_audit_v11.json').read_text())
    if not audit['passed'] or audit['summary_sha256'] != sha256(root/'summary.json'):
        raise ValueError('Completed surface-warp replay required')
    initial = Path('runs/localization-v7-native-initial-control/initial_results.json.gz')
    eq = json.loads(Path('reports/native_initial_equivalence_v7.json').read_text())
    if not eq['passed'] or eq['new_results_sha256'] != sha256(initial):
        raise ValueError('Audited equivalent frozen baseline required')
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
        raise ValueError('Incomplete common development comparison')
    for key, row in sift.items():
        row['component'] = base[key]['component']; row['estimated_cam2world'] = row['pose']
    fallback_base = {i: sift[i] if sift[i]['pose'] is not None else base[i] for i in base}
    mixed = {i: sift[i] if sift[i]['pose'] is not None else candidate[i] for i in base}
    count = lambda rows: dict(cases=len(rows), correct=sum(r['pose_correct'] for r in rows.values()),
        strict=sum(r['strict_pose_correct'] for r in rows.values()),
        estimated=sum(r['estimated_cam2world'] is not None for r in rows.values()),
        wrong=sum(r['estimated_cam2world'] is not None and not r['pose_correct'] for r in rows.values()))
    result = dict(scope='Deterministic3D normalization on exposed development terrain; no learned breakthrough, calibrated acceptance or fresh final claim.',
        frozen=count(base), surface=count(candidate), frozen_sift_fallback=count(fallback_base), surface_sift_fallback=count(mixed),
        raw_gain_pp=100*(count(candidate)['correct']-count(base)['correct'])/72,
        fallback_gain_pp=100*(count(mixed)['correct']-count(fallback_base)['correct'])/72,
        descriptive_component_interval_pp=paired_interval(candidate, base), fallback_descriptive_component_interval_pp=paired_interval(mixed, fallback_base),
        per_component={g: dict(cases=sum(r['component'] == g for r in base.values()),
            frozen_correct=sum(r['pose_correct'] for r in base.values() if r['component'] == g),
            surface_correct=sum(r['pose_correct'] for r in candidate.values() if r['component'] == g)) for g in sorted({r['component'] for r in base.values()})},
        gained_case_ids=[i for i in base if candidate[i]['pose_correct'] and not base[i]['pose_correct']],
        lost_case_ids=[i for i in base if base[i]['pose_correct'] and not candidate[i]['pose_correct']],
        frozen_correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in base.values()),
        surface_correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in candidate.values()),
        interval_note='Descriptive paired bootstrap on4 exposed selection components after hypothesis formation. Not fresh-test confidence.',
        source_sha256=sha256(__file__), results_sha256=sha256(root/'results.json.gz'), frozen_results_sha256=sha256(initial),
        calibration_or_test_pixels_loaded=0)
    write_new('reports/native_surface_comparison_v11.json', result)
    b, c, f = result['frozen'], result['surface'], result['surface_sift_fallback']
    Path('reports/SURFACE_WARP_V11.md').write_text(f'''# Explicit surface geometry improves a frozen small matcher

On the same72 exposed development pairs, fixed two-iteration reference-surface rectification changes raw pose recovery **{b['correct']}→{c['correct']}/72**, and wrong poses **{b['wrong']}→{c['wrong']}**. The model remains the frozen28,272-parameter CNN: this is a deterministic geometry/inference correction, not a new learned breakthrough. The stricter pose count changes{b['strict']}→{c['strict']}/72, so the primary gain is not uniform precision dominance. Exact two-pixel correspondence counts change{result['frozen_correct_at_two_px']}→{result['surface_correct_at_two_px']}, with changed support/returned-point denominators retained in the raw results.

The same SIFT-first fallback gives **{f['correct']}/72** with **{f['wrong']} wrong raw poses**, versus53/72 with six wrong for the old fallback. The raw gain is{result['raw_gain_pp']:.2f} points; its descriptive terrain interval is{result['descriptive_component_interval_pp']}. The fallback interval is{result['fallback_descriptive_component_interval_pp']}. These intervals resample only four exposed terrain components after the hypothesis was developed. They are not final-test confidence or a1% risk guarantee. Nonoverlap rejection and pair confidence have not been calibrated for this method.

First the unchanged image-only orientation matcher estimates a query pose using reference geometry. That estimate projects the known reference axial-depth surface into the query image, which is resampled into reference coordinates. Frozen descriptors search a33pixel neighborhood there; those points map back into original query pixels and the unchanged3native-pixel AP3P estimator updates pose. Repeat twice, retaining the last successful result if a subsequent pose fit fails. True query pose and query depth are absent from inference and used only afterward for scoring. Query-depth occlusion is unavailable during inference; valid warp support means finite/front-facing reference projection and source image bounds, never non-darkness. Rectification, interpolation, support and physical search extent change together, so this does not isolate one factor.

All57 initially estimated cases remain estimated; this control cannot recover pairs with no initial pose. A useful next learned experiment would train with geometry-normalized lighting pairs and compare against this stronger deterministic control. The earlier unnormalized learned native model gives45/42/42 across its three seeds; adding complexity alone did not help. The uniform-positive study remains separate and live.

Independent replay verifies every source/input/checkpoint hash, actual frozen-model quarter-turn/blank controls, native correspondence and pose errors, then reconstructs reference3D points independently and reruns the common pose estimator. It does not independently reimplement learned inference or assert true query-depth visibility. The fixed16-case pilot gives12/16 versus11 frozen; all72 expanded cases remain in the denominator. No new final pixels, paid compute, human review, automatic outreach, global SOTA or orbital-navigation claim.

Reproduce with `evaluate_geometry_warp_v11.py --full --output runs/new-surface-warp`, then `verify_geometry_warp_v11.py --run runs/new-surface-warp --output runs/new-surface-audit.json`. Source data: MoonAnything LunarGeo, Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0. Own code/checkpoint: MIT.
''')
    print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    report()
