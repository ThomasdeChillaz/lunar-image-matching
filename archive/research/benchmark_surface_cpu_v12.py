"""Full surface-control scoring plus a fixed16-case same-worker timing replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
from time import perf_counter

import cv2
import numpy as np
import torch

from compact_lunar import CompactMatcher
from dataset import sha256
from evaluate_geometry_warp_v11 import control as surface_control
from geometry_warp_native_v11 import infer
from lunargeo_dataset import write_new
from modern_baselines_v2 import ModernMatcher
from native_baseline_pose_v7 import controls, propose_pose, score


def run(method, source, output, max_seconds=2400):
    root = Path(output)
    if root.exists():
        raise FileExistsError('Preserve every attempt')
    root.mkdir(parents=True); (root/'cases').mkdir(); torch.set_num_threads(2); cv2.setNumThreads(1)
    path = Path('reports/localization_training_data_v5.json'); manifest = json.loads(path.read_text())
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection']
    bank = set(manifest['selection_bank'])
    if len(cases) != 72 or len(bank) != 16 or not bank <= {c['pair_id'] for c in cases}:
        raise ValueError('Fixed full/latency case definitions differ')
    sources = ['benchmark_surface_cpu_v12.py', 'evaluate_geometry_warp_v11.py', 'geometry_warp_native_v11.py',
        'native_baseline_pose_v7.py', 'modern_baselines_v2.py', 'coarse_pipeline_v5.py', 'compact_lunar.py',
        'contrastive_terrain.py', 'localization_diagnostics_v3.py', 'lunar_geometry_v5.py', 'lunar_geometry_v2.py',
        'localization_robust_pose_v4.py', 'localization_pose_refit_v3.py', 'rotation_bank_pose_v4.py']
    checkpoint = Path(manifest['coarse_model'])/'selected.pt'
    if sha256(checkpoint) != manifest['coarse_checkpoint_sha256']:
        raise ValueError('Frozen small checkpoint differs')
    protocol = dict(scope='Exposed-development surface-control CPU validation; no calibrated acceptance, final test or global SOTA claim.',
        method=method, profile='common', cpu_threads=2, case_ids=[c['pair_id'] for c in cases],
        paired_timing_case_ids=[c['pair_id'] for c in cases if c['pair_id'] in bank],
        latency_selection='The existing fixed16-case bank, frozen before any native benchmark; no selection by runtime or correctness.',
        latency='Alternate method order by original global index. Fresh features for both methods on paired cases. Initialization/download/ground-truth scoring excluded; both loaded and warmed in the same worker.',
        accuracy='Surface method on all72; modern repeat only on fixed16. Retain previous fixed36+36 full modern result as historical principal reference; never select better duplicates.',
        manifest_sha256=sha256(path), source_sha256={p: sha256(p) for p in sources}, frozen_checkpoint_sha256=sha256(checkpoint),
        input_sha256={t['native_cache']: t['native_cache_sha256'] for t in tiles.values()},
        runtime=dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__, platform=platform.platform()),
        public_execution_id=os.environ.get('GITHUB_RUN_ID'), source_commit=os.environ.get('GITHUB_SHA'),
        max_comparison_seconds=max_seconds, calibration_or_test_pixels_loaded=0)
    write_new(root/'protocol.json', protocol)
    data = {}
    for key, t in tiles.items():
        if sha256(t['native_cache']) != t['native_cache_sha256']:
            raise ValueError('Native source content differs')
        with np.load(t['native_cache'], allow_pickle=False) as z:
            data[key] = {k: z[k].copy() for k in ('images', 'depth_a', 'depth_b')}
    matcher = ModernMatcher(method, source, 'common', 'cpu'); write_new(root/'provenance.json', matcher.provenance)
    modern_control = controls(matcher); write_new(root/'modern_controls.json', modern_control)
    coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval()
    compact_control = surface_control(coarse); write_new(root/'surface_controls.json', compact_control)
    if not modern_control['passed'] or not compact_control['passed']:
        raise ValueError('Actual pretrained/frozen control failed; no lunar inference')
    rows = []; cached = {}; last_tile = None; started = perf_counter()
    for i, c in enumerate(cases):
        if perf_counter()-started > max_seconds:
            write_new(root/'TIMEOUT.json', dict(completed_cases=i, eligible_cases=72)); return
        if c['tile_id'] != last_tile:
            cached.clear(); last_tile = c['tile_id']
        t = tiles[c['tile_id']]; z = data[c['tile_id']]; a, b = t['cameras']
        ia, ib = 2*c['light_a'], 2*c['light_b']+1; ref, query = z['images'][ia], z['images'][ib]
        paired = c['pair_id'] in bank
        order = ([method, 'compact_surface'] if i % 2 == 0 else ['compact_surface', method]) if paired else ['compact_surface']
        for which in order:
            tick = perf_counter()
            if which == method:
                p, q = matcher.points(ref, query)
                pose, info = propose_pose(p, q, z['depth_a'], a['intrinsics'], a['cam2world'], b['intrinsics']); matcher.sync()
            else:
                chosen, branches = infer(coarse, ref, query, z['depth_a'], a['intrinsics'], a['cam2world'], b['intrinsics'],
                    None if paired else cached, ((c['tile_id'], ia), (c['tile_id'], ib)))
                branch = chosen if chosen is not None else branches[0]
                p, q, pose, info = branch['p'], branch['q'], None if chosen is None else branch['pose'], branch['info']
            elapsed = (perf_counter()-tick)*1000
            # Query truth first enters here, after prediction and measured time.
            row = dict(c, method=which, paired_timing=paired, elapsed_ms=elapsed, pose_info=info,
                **score(p, q, pose, z, t['cameras']))
            rows.append(row); name = hashlib.sha256((c['pair_id']+which).encode()).hexdigest()[:20]
            write_new(root/'cases'/(name+'.json'), row)
        print('surface CPU', i+1, '/72', 'paired' if paired else 'surface-only', flush=True)
    results = {}
    for which in (method, 'compact_surface'):
        part = [r for r in rows if r['method'] == which]; timed = [r for r in part if r['paired_timing']]
        results[which] = dict(cases=len(part), correct=sum(r['pose_correct'] for r in part),
            strict=sum(r['strict_pose_correct'] for r in part), estimated=sum(r['estimated_cam2world'] is not None for r in part),
            paired_timing_cases=len(timed), median_paired_inference_ms=float(np.median([r['elapsed_ms'] for r in timed])))
    lookup = {(r['pair_id'], r['method']): r['elapsed_ms'] for r in rows if r['paired_timing']}
    speedup = float(np.median([lookup[c, method]/lookup[c, 'compact_surface'] for c in bank]))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(scope=protocol['scope'], results=results, median_paired_surface_speedup=speedup,
        comparison_seconds=perf_counter()-started, peak_joint_process_rss_bytes=int(rss if platform.system() == 'Darwin' else rss*1024),
        memory_note='Both models loaded; joint-process memory is not isolated per-method memory.', calibration_or_test_pixels_loaded=0)
    write_new(root/'summary.json', result); print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--method', choices=['loma_r', 'romav2'], default='romav2')
    p.add_argument('--source', required=True); p.add_argument('--output', required=True); p.add_argument('--max-seconds', type=int, default=2400)
    a = p.parse_args(); run(a.method, a.source, a.output, a.max_seconds)
