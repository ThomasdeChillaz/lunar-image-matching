"""Exposed-development, matched-support test of shared descriptor scales."""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import subprocess
from time import perf_counter

import cv2
import numpy as np
import torch

from coarse_pipeline_v5 import infer as initial_infer
from compact_lunar import CompactMatcher
from dataset import sha256
from evaluate_geometry_warp_v11 import SOURCES as BASE_SOURCES
from fine_pyramid_v22 import encode, refine_surface
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import score

PROFILES = ((1,), (1, 2), (1, 2, 4))
SOURCES = list(dict.fromkeys(BASE_SOURCES+['fine_pyramid_v22.py', 'evaluate_fine_pyramid_v22.py', 'localization_refiner_v3.py']))


def control(coarse):
    from synthetic import texture
    image = texture(733, 512); query = np.rot90(image, 1).copy()
    k = np.array([[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]])
    depth = np.full(image.shape, 10.)
    initial, _ = initial_infer(coarse, coarse.model, image, query, depth, k, np.eye(4), k)
    if initial is None:raise ValueError('Frozen native initialization control failed')
    expected = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
    features = encode(coarse.model, image); records = []
    for profile in PROFILES:
        actual = refine_surface(coarse.model, image, query, depth, k, np.eye(4), k, initial, profile, features)
        re = float(np.linalg.norm(actual['pose'][:3, :3]-expected))
        te = float(np.linalg.norm(actual['pose'][:3, 3]))
        records.append(dict(scales=list(profile), rotation_matrix_error=re, translation_m=te,
            iterations=len(actual['geometry_history']), passed=re < .01 and te < .05))
    blank, _ = initial_infer(coarse, coarse.model, np.zeros_like(image), np.zeros_like(image), depth, k, np.eye(4), k)
    return dict(passed=all(r['passed'] for r in records) and blank is None, records=records,
        blank_pose_null=blank is None, scope='Actual shared frozen checkpoint software control, not lunar performance.')


def run(output, source_commit, full=False, max_seconds=1800):
    root = Path(output)
    if root.exists():raise FileExistsError('Preserve every experiment/control attempt')
    if not re.fullmatch('[0-9a-f]{40}', source_commit):raise ValueError('Full published source commit required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', source_commit+':'+name]) != Path(name).read_bytes():
            raise ValueError('Diagnostic source differs from published freeze: '+name)
    root.mkdir(parents=True); (root/'cases').mkdir(); (root/'source').mkdir()
    torch.set_num_threads(2); cv2.setNumThreads(1)
    manifest_path = Path('reports/localization_training_data_v5.json'); manifest = json.loads(manifest_path.read_text())
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}
    bank = set(manifest['selection_bank'])
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection' and (full or c['pair_id'] in bank)]
    if len(cases) != (72 if full else 16):raise ValueError('Fixed development case list differs')
    checkpoint = Path(manifest['coarse_model'])/'selected.pt'
    if sha256(checkpoint) != manifest['coarse_checkpoint_sha256']:raise ValueError('Frozen encoder differs')
    sources = {name: sha256(name) for name in SOURCES}
    for name in sources:(root/'source'/name).write_bytes(Path(name).read_bytes())
    protocol = dict(scope='Exposed model-selection diagnostic, no training, new test, accepted-pose or runtime superiority claim.',
        recorded_utc=datetime.now(timezone.utc).isoformat(), source_commit=source_commit,
        source_sha256=sources, manifest_sha256=sha256(manifest_path), case_ids=[c['pair_id'] for c in cases],
        profiles=[list(p) for p in PROFILES], fine_encoder_checkpoint_sha256=sha256(checkpoint),
        hypothesis='At native resolution the frozen25px descriptor field covers4x less terrain than during128px pretraining. Shared1/2/4 scales may recover context without more parameters.',
        matching='Same initial pose and original point list for every profile; same33nativepx candidates, equal-weight cosine scores, peak-only subpixel interpolation, two surface updates.',
        support='Same conservative56nativepx feature support and72px normalized-warp support in every scale profile. Historical v11 original-border results remain separate.',
        query_geometry='Query depth/true pose used only after prediction for scoring; shadow is not occlusion.',
        selection='No per-case truth-based method or checkpoint choice; report every profile.',
        input_sha256={t['native_cache']: t['native_cache_sha256'] for t in tiles.values()},
        cpu_threads=2, max_comparison_seconds=max_seconds, calibration_or_test_pixels_loaded=0,
        timing='Cross-case feature caching allowed for a bounded diagnostic. Not an isolated speed benchmark.')
    write_new(root/'protocol.json', protocol)
    coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval().requires_grad_(False)
    controls = control(coarse); write_new(root/'controls.json', controls)
    if not controls['passed']:raise ValueError('Actual scale-refinement control failed; no lunar inference')
    data = {}
    for key in sorted({c['tile_id'] for c in cases}):
        tile = tiles[key]
        if sha256(tile['native_cache']) != tile['native_cache_sha256']:raise ValueError('Development input changed')
        with np.load(tile['native_cache'], allow_pickle=False) as z:data[key] = {name: z[name].copy() for name in ('images', 'depth_a', 'depth_b')}
    historical_path = Path('runs/localization-v11-surface-warp-development/results.json.gz')
    audit = json.loads(Path('reports/native_surface_audit_v11.json').read_text())
    if not audit['passed'] or audit['results_sha256'] != sha256(historical_path):raise ValueError('Historical surface evidence changed')
    with gzip.open(historical_path, 'rt') as f:
        historical = {r['pair_id']: r for r in json.load(f)}
    rows = []; initial_features = {}; pyramids = {}; started = perf_counter()
    for index, case in enumerate(cases):
        if perf_counter()-started > max_seconds:
            write_new(root/'TIMEOUT.json', dict(completed_pairs=index, eligible_pairs=len(cases))); return
        key = case['tile_id']; tile = tiles[key]; z = data[key]; a, b = tile['cameras']
        ia, ib = 2*case['light_a'], 2*case['light_b']+1; ref, query = z['images'][ia], z['images'][ib]
        initial, branches = initial_infer(coarse, coarse.model, ref, query, z['depth_a'], a['intrinsics'],
            a['cam2world'], b['intrinsics'], initial_features, ((key, ia), (key, ib)))
        if (key, ia) not in pyramids:pyramids[key, ia] = encode(coarse.model, ref)
        for profile in PROFILES:
            selected = refine_surface(coarse.model, ref, query, z['depth_a'], a['intrinsics'],
                a['cam2world'], b['intrinsics'], initial, profile, pyramids[key, ia])
            branch = selected if selected is not None else branches[0]
            row = dict(case, method='fine_scales_'+ '_'.join(map(str, profile)),
                **score(branch['p'], branch['q'], None if selected is None else branch['pose'], z, tile['cameras']),
                geometry_history=branch.get('geometry_history', []))
            rows.append(row)
            write_new(root/'cases'/f'{index:03d}-{row["method"]}.json', row)
        print('fine pyramid', index+1, '/', len(cases), [r['pose_correct'] for r in rows[-3:]], flush=True)
    summary = {}
    for profile in PROFILES:
        method = 'fine_scales_'+ '_'.join(map(str, profile)); part = [r for r in rows if r['method'] == method]
        summary[method] = dict(cases=len(part), correct=sum(r['pose_correct'] for r in part),
            strict=sum(r['strict_pose_correct'] for r in part), estimated=sum(r['estimated_cam2world'] is not None for r in part),
            correct_at_two_px=sum(r['correct_at_native_px']['2'] for r in part), returned=sum(r['returned'] for r in part))
    original = [historical[c['pair_id']] for c in cases]
    result = dict(scope=protocol['scope'], results=summary,
        historical_original_support=dict(cases=len(original), correct=sum(r['pose_correct'] for r in original),
            strict=sum(r['strict_pose_correct'] for r in original)),
        comparison_seconds=perf_counter()-started, calibration_or_test_pixels_loaded=0,
        protocol_sha256=sha256(root/'protocol.json'), historical_results_sha256=sha256(historical_path),
        case_sha256={p.name: sha256(p) for p in sorted((root/'cases').glob('*.json'))})
    write_new(root/'summary.json', result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    p.add_argument('--source-commit', required=True); p.add_argument('--full', action='store_true'); p.add_argument('--max-seconds', type=int, default=1800)
    a = p.parse_args(); run(a.output, a.source_commit, a.full, a.max_seconds)
