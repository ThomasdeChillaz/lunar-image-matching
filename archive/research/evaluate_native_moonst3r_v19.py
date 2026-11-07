"""Native MOONSt3R correspondences with the common reference-depth pose estimator.

Requires a passed actual control and public source freeze. Author predicted3D
never enters pose estimation. Exact released training overlap remains unknown.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
from time import perf_counter

import cv2
import numpy as np
import torch

from control_native_moonst3r_v19 import SOURCES as CONTROL_SOURCES
from dataset import sha256
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import propose_pose, score
from native_moonst3r_v19 import NativeMOONSt3R

SOURCES = list(dict.fromkeys(CONTROL_SOURCES+['evaluate_native_moonst3r_v19.py']))


def run(output, source_commit, dependency_dir, bank=False, max_seconds=14400):
    if not re.fullmatch('[0-9a-f]{40}', source_commit):raise ValueError('Full public source freeze required')
    for name in SOURCES:
        if subprocess.check_output(['git', 'show', source_commit+':'+name]) != Path(name).read_bytes():
            raise ValueError('Published inference source differs: '+name)
    control_root = Path('runs/localization-v19-moonst3r-native-control')
    gate = json.loads((control_root/'summary.json').read_text())
    if not gate['passed'] or not gate['actual_native_pretrained_forward_attempted']:
        raise ValueError('Actual pretrained-forward/native software gate required')
    gp = json.loads((control_root/'protocol.json').read_text())
    for name, want in gp['source_sha256'].items():
        if sha256(name) != want:raise ValueError('Controlled inference source changed: '+name)
    manifest_path = Path('reports/localization_training_data_v5.json'); manifest = json.loads(manifest_path.read_text())
    if manifest['calibration_or_test_pixels_loaded'] or not manifest['train_selection_disjoint']:
        raise ValueError('Audited development-only inputs required')
    cases = [c for c in manifest['cases'] if c['split'] == 'model_selection' and (not bank or c['pair_id'] in manifest['selection_bank'])]
    if len(cases) != (16 if bank else 72):raise ValueError('Fixed denominator differs')
    tiles = {t['tile_id']: t for t in manifest['tiles'] if t['split'] == 'model_selection'}
    root = Path(output)
    if root.exists():
        if (root/'summary.json').exists():raise FileExistsError('Completed comparisons are immutable')
        protocol = json.loads((root/'protocol.json').read_text())
        if (protocol['source_commit'], protocol['bank_only'], protocol['wall_cap_seconds'], protocol['case_ids']) != (
            source_commit, bank, max_seconds, [c['pair_id'] for c in cases]):raise ValueError('Resume protocol differs')
        for name, want in protocol['source_sha256'].items():
            if sha256(name) != want:raise ValueError('Executed source changed')
    else:
        root.mkdir(parents=True); (root/'cases').mkdir(); (root/'source').mkdir()
        protocol = dict(scope='Exposed-development common-native MOONSt3R pose diagnostic, not fresh transfer, accepted navigation or SOTA.',
            source_commit=source_commit, source_sha256={n: sha256(n) for n in SOURCES}, bank_only=bank,
            case_ids=[c['pair_id'] for c in cases], manifest_sha256=sha256(manifest_path),
            input_sha256={t['native_cache']: t['native_cache_sha256'] for t in tiles.values()},
            actual_gate_summary_sha256=sha256(control_root/'summary.json'), cpu_threads=2,
            profile='common_native512_full_field_of_view', native_input_size=512,
            geometry='Same reference axial geometry/AP3P as compact and modern baselines. Query depth/true pose enter scoring only; predicted3D unused.',
            comparison='Accuracy against saved common-case baselines. No concurrent paired-worker latency comparison is implied.',
            overlap='Checkpoint-declared training split files unpublished; exact lunar training-data overlap unknown.',
            wall_cap_seconds=max_seconds, created_utc=datetime.now(timezone.utc).isoformat(),
            calibration_or_test_pixels_loaded=0)
        write_new(root/'protocol.json', protocol)
        for name in SOURCES:(root/'source'/name).write_bytes(Path(name).read_bytes())
    if sha256(manifest_path) != protocol['manifest_sha256']:raise ValueError('Manifest changed')
    # Conservative shared deadline includes startup and interrupted wall time;
    # restarting can never grant another comparison budget.
    created = datetime.fromisoformat(protocol['created_utc'])
    remaining = max_seconds-(datetime.now(timezone.utc)-created).total_seconds()
    if remaining <= 0:raise TimeoutError('Shared comparison deadline exhausted; preserve incomplete records')
    def timeout(*unused):raise TimeoutError('Shared baseline wall cap reached; preserve complete cases')
    previous = signal.signal(signal.SIGALRM, timeout); signal.alarm(max(1, int(remaining)))
    started = perf_counter(); rows = []
    try:
        torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
        model = NativeMOONSt3R(dependency_dir=dependency_dir, threads=2)
        original_provenance = json.loads((control_root/'provenance.json').read_text())
        if model.provenance != original_provenance:raise ValueError('Model differs from actual controlled baseline')
        if (root/'provenance.json').exists():
            if json.loads((root/'provenance.json').read_text()) != model.provenance:raise ValueError('Resume model changed')
        else:write_new(root/'provenance.json', model.provenance)
        data = {}
        for index, case in enumerate(cases):
            name = hashlib.sha256(case['pair_id'].encode()).hexdigest()[:20]+'.json'; path = root/'cases'/name
            if path.exists():
                row = json.loads(path.read_text())
                if any(row[k] != v for k, v in case.items()) or row['method'] != 'moonst3r_native':raise ValueError('Saved case identity differs')
                rows.append(row); continue
            key = case['tile_id']; tile = tiles[key]
            if key not in data:
                if sha256(tile['native_cache']) != tile['native_cache_sha256']:raise ValueError('Native input changed')
                with np.load(tile['native_cache'], allow_pickle=False) as z:
                    data[key] = {n: z[n].copy() for n in ('images', 'depth_a', 'depth_b')}
            z = data[key]; a, b = tile['cameras']; tick = perf_counter(); failure = None
            try:
                p, q = model.points(z['images'][2*case['light_a']], z['images'][2*case['light_b']+1])
                pose, info = propose_pose(p, q, z['depth_a'], a['intrinsics'], a['cam2world'], b['intrinsics'])
            except (RuntimeError, cv2.error) as exc:
                p = q = np.empty((0, 2)); pose = None; info = dict(inliers=0, reason='inference_error')
                failure = dict(type=type(exc).__name__, message=str(exc))
            elapsed = perf_counter()-tick
            row = dict(case, method='moonst3r_native', elapsed_seconds=elapsed, inference_error=failure, pose_info=info,
                **score(p, q, pose, z, tile['cameras']))
            write_new(path, row); rows.append(row)
            print('MOONSt3R native', index+1, '/', len(cases), 'correct', sum(r['pose_correct'] for r in rows), flush=True)
        summary = dict(scope=protocol['scope'], cases=len(rows), correct=sum(r['pose_correct'] for r in rows),
            strict=sum(r['strict_pose_correct'] for r in rows), wrong=sum(r['estimated_cam2world'] is not None and not r['pose_correct'] for r in rows),
            inference_errors=sum(r['inference_error'] is not None for r in rows),
            median_inference_seconds=float(np.median([r['elapsed_seconds'] for r in rows])),
            elapsed_this_invocation_seconds=perf_counter()-started, protocol_sha256=sha256(root/'protocol.json'),
            case_sha256={p.name: sha256(p) for p in sorted((root/'cases').glob('*.json'))}, calibration_or_test_pixels_loaded=0)
        write_new(root/'summary.json', summary); print(json.dumps(summary, indent=2)); return summary
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True); p.add_argument('--source-commit', required=True)
    p.add_argument('--dependency-dir'); p.add_argument('--bank', action='store_true'); p.add_argument('--max-seconds', type=int, default=14400)
    a = p.parse_args()
    if not 600 <= a.max_seconds <= 14400:raise ValueError('Bounded600–14400second comparison required')
    run(a.output, a.source_commit, a.dependency_dir, a.bank, a.max_seconds)
