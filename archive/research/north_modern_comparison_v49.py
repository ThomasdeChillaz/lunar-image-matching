"""Prepared common northern comparison; no pretrained result until executed.

Standalone SIFT and pinned modern image matchers get reference geometry only.
The paired compact control retains its existing native/surface pipeline. This
does not rename the earlier SIFT-first fallback as standalone SIFT.
"""
import argparse
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import re
import resource
import subprocess
from time import perf_counter

import numpy as np

METHODS = ('sift', 'matchanything', 'loma_r', 'romav2', 'moonst3r')
MANIFEST = 'reports/north_training_data_v38.json'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value: raise ValueError('Saved comparison evidence differs')
        return
    temporary = Path(str(path) + '.temporary')
    if temporary.exists(): raise FileExistsError('Preserve interrupted temporary output')
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def checked_points(p, q):
    p, q = [np.asarray(v, dtype=float).reshape(-1, 2) for v in (p, q)]
    if p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Finite paired native correspondences required')
    if any(((v < 0) | (v > 511)).any() for v in (p, q)):
        raise ValueError('Backend returned coordinates outside native image')
    return p, q


def image_only_prediction(matcher, inputs, estimator):
    if len(inputs) != 6: raise ValueError('Images plus reference-only geometry required')
    # Backend sees images only. The estimator gets reference depth/pose and Ks.
    p, q = checked_points(*matcher.points(inputs[0], inputs[1]))
    pose, info = estimator(p, q, *inputs[2:])
    matcher.sync()
    return p, q, pose, info


def upstream_gate():
    for path in ('runs/localization-v40-recovery-training-state.json',
                 'runs/north-v41-regression-state.json', 'runs/north-v43-followup-state.json'):
        state = read(path)
        if state['state'] != 'complete': raise ValueError('Required earlier pipeline is unfinished')
        for field in ('pid', 'child_pid'):
            if state.get(field):
                active = subprocess.run(['ps', '-p', str(state[field]), '-o', 'command='],
                    text=True, capture_output=True).stdout.strip()
                if active: raise ValueError('Upstream PID is active or reused; inspect before inference')
    checks = ('reports/north_software_checks_v41.json',
              'reports/north_expanded_control_audit_v39.json',
              'reports/north_coarse_reachability_audit_v43.json')
    if not all(read(p)['passed'] for p in checks): raise ValueError('Required independent checks failed')
    return {p: sha(p) for p in checks}


def build_matcher(method, source, dependency_dir):
    if method in ('loma_r', 'romav2'):
        from cpu_budget_v2 import inspect_cpu
        if not inspect_cpu()['native_bf16']:
            raise RuntimeError('Unchanged author CPU profile requires reported native BF16; no silent precision fallback')
        from modern_baselines_v2 import ModernMatcher
        return ModernMatcher(method, source, 'common', 'cpu')
    if method == 'matchanything':
        from native_matchanything_v17 import NativeMatchAnything
        return NativeMatchAnything('.', dependency_dir, 'native512', 2)
    if method == 'moonst3r':
        from native_moonst3r_v19 import NativeMOONSt3R
        return NativeMOONSt3R(source=source, dependency_dir=dependency_dir, threads=2)
    if method != 'sift': raise ValueError('Unsupported fixed comparison method')
    from native_sift_pose_v4 import points
    from loftr_matcher import valid_correspondences
    class SIFT:
        provenance = dict(method='standalone_SIFT', profile='original_native512',
            filtering='Existing SIFT ratio matching; no compact fallback or planar fit', cpu_threads=2)
        def sync(self): pass
        def points(self, a, b, ma=None, mb=None):
            ma, mb = [np.ones(im.shape, bool) if mask is None else np.asarray(mask, bool)
                      for im, mask in ((a, ma), (b, mb))]
            if any(im.shape != (512, 512) or im.dtype != np.uint8 for im in (a, b)):
                raise ValueError('Native512 uint8 images required')
            if ma.shape != a.shape or mb.shape != b.shape: raise ValueError('Mask shape differs')
            if not ma.any() or not mb.any(): return np.empty((0, 2)), np.empty((0, 2))
            p, q = points(a, b)
            return valid_correspondences(p, q, ma, mb)
    return SIFT()


def run(method, source, dependency_dir, output, commit, full, cap):
    start = datetime.now(timezone.utc); tick = perf_counter()
    # Admission happens before importing torch, loading data or a network.
    if method not in METHODS or type(cap) is not int or not 60 <= cap <= 21600:
        raise ValueError('Fixed method and bounded60..21600second allowance required')
    if not re.fullmatch('[0-9a-f]{40}', commit): raise ValueError('Full public source commit required')
    upstream = upstream_gate()
    software_path = 'reports/north_modern_software_checks_v49.json'
    software = read(software_path)
    if not (software['passed'] and software['full_repository_regression'] and software['fresh_smoke']):
        raise ValueError('New comparison software checks are incomplete')
    required_software = {'north_modern_comparison_v49.py', 'verify_north_modern_v49.py',
                         'test_north_modern_comparison_v49.py'}
    if not required_software.issubset(software['source_sha256']):
        raise ValueError('Software checks do not bind the new comparison sources')
    for name, digest in software['source_sha256'].items():
        if sha(name) != digest: raise ValueError('Comparison software changed after checks')
    from benchmark_north_control_v38 import inference_inputs, score_case, case_list, totals
    from north_training_bundle_v38 import load_bundle
    from evaluate_geometry_warp_v11 import SOURCES as SURFACE_SOURCES, control
    from native_baseline_pose_v7 import propose_pose, controls
    from compact_lunar import CompactMatcher
    from geometry_warp_native_v11 import infer
    import cv2
    import torch
    sources = list(dict.fromkeys(SURFACE_SOURCES + ['north_modern_comparison_v49.py',
        'benchmark_north_control_v38.py', 'north_training_bundle_v38.py', 'north_development_plan_v35.py',
        'native_baseline_pose_v7.py', 'native_sift_pose_v4.py', 'native_matchanything_v17.py',
        'native_moonst3r_v19.py', 'modern_baselines_v2.py', 'cpu_budget_v2.py',
        'matchanything_model_lock.json', 'moonst3r_model_lock_v19.json',
        'matchanything_matcher.py', 'eloftr_matcher.py', 'loftr_matcher.py',
        'reports/compact_baseline_lock_v2.json',
        'verify_north_modern_v49.py', 'test_north_modern_comparison_v49.py']))
    for p in sources:
        if subprocess.check_output(['git', 'show', commit + ':' + p]) != Path(p).read_bytes():
            raise ValueError('Comparison source differs from public freeze: ' + p)
    root = Path(output)
    if root.exists(): raise FileExistsError('Keep all partial/complete comparison attempts')
    root.mkdir(parents=True); (root / 'cases').mkdir()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261101); cv2.setNumThreads(1)
    bundle = load_bundle(MANIFEST, selection_only=True); manifest = bundle[0]
    cases = case_list(manifest, full)
    protocol = dict(method=method, profile='common_full_native_content', full=full,
        publication_commit=commit, source_sha256={p: sha(p) for p in sources},
        manifest_sha256=sha(MANIFEST), upstream_check_sha256=upstream,
        software_check_sha256=sha(software_path),
        case_ids=[c['pair_id'] for c in cases], started_utc=start.isoformat(),
        absolute_deadline_utc=(start + timedelta(seconds=cap)).isoformat(), max_calendar_seconds=cap,
        cpu_threads=2, input_size=512, calibrated_acceptance_evaluated=False,
        scope='Exposed northern development raw-pose comparison, not final/acceptance/SOTA evidence.',
        geometry='Common reference axial depth/pose and query intrinsics; query pose/depth first enter scoring.',
        timing='Both models warmed and alternated per case; fresh compact feature cache each call. All matching/pose work timed. Controls/loading count toward allowance.',
        limitation='Compact includes its unchanged four-rotation/native/surface refinement; modern raw correspondences use the common final pose estimator. Common profiles do not replace missing author-profile comparisons.',
        runtime=dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__, platform=platform.platform()))
    write_once(root / 'protocol.json', protocol)
    def clock():
        wall = (datetime.now(timezone.utc) - start).total_seconds()
        if wall < 0 or wall >= cap or perf_counter() - tick >= cap:
            raise TimeoutError('Original fixed comparison allowance exhausted; preserve partial cases')
    rows = []
    try:
        matcher = build_matcher(method, source, dependency_dir); clock()
        write_once(root / 'provenance.json', matcher.provenance)
        coarse = CompactMatcher(manifest['coarse_model']); coarse.model.eval().requires_grad_(False)
        checks = dict(pretrained_or_sift=controls(matcher), compact=control(coarse)); clock()
        write_once(root / 'controls.json', checks)
        if not all(c['passed'] for c in checks.values()): raise ValueError('Actual native controls failed')
        for i, case in enumerate(cases):
            inputs = inference_inputs(bundle, case)
            order = [method, 'frozen_surface'] if i % 2 == 0 else ['frozen_surface', method]
            for which in order:
                clock(); before = perf_counter(); failure = None; history = []
                try:
                    if which == method:
                        p, q, pose, info = image_only_prediction(matcher, inputs, propose_pose)
                    else:
                        selected, branches = infer(coarse, *inputs)
                        branch = selected if selected is not None else branches[0]
                        p, q = checked_points(branch['p'], branch['q'])
                        pose = None if selected is None else branch['pose']; info = branch['info']
                        history = branch.get('geometry_history', [])
                except (RuntimeError, ValueError, cv2.error) as exc:
                    p = q = np.empty((0, 2)); pose = None; info = dict(inliers=0, reason='inference_error')
                    failure = dict(type=type(exc).__name__, message=str(exc))
                elapsed_ms = (perf_counter() - before) * 1000
                # First query-pose/depth consumption is explicitly after timing.
                row = dict(case, method=which, elapsed_ms=elapsed_ms, inference_error=failure,
                    pose_info=info, geometry_history=history, **score_case(bundle, case, p, q, pose))
                path = root / 'cases' / (hashlib.sha256((case['pair_id'] + '/' + which).encode()).hexdigest() + '.json')
                write_once(path, row); rows.append(row); clock()
            print('NORTH MODERN', method, i + 1, '/', len(cases), flush=True)
        temporary = root / 'per_case.json.gz.temporary'
        with gzip.open(temporary, 'wt') as stream: json.dump(rows, stream, allow_nan=False)
        temporary.replace(root / 'per_case.json.gz')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary = dict(complete=True, full=full, eligible_pairs_per_method=len(cases),
            results={name: totals([r for r in rows if r['method'] == name]) for name in (method, 'frozen_surface')},
            protocol_sha256=sha(root / 'protocol.json'), results_sha256=sha(root / 'per_case.json.gz'),
            peak_joint_process_rss_bytes=int(rss if platform.system() == 'Darwin' else rss * 1024),
            memory_scope='Joint process including both models/controls, not isolated model memory.',
            calibrated_acceptance_evaluated=False, calibration_or_test_pixels_loaded=0)
        write_once(root / 'summary.json', summary)
        write_once(root / 'COMPLETE.json', dict(sha256={str(p.relative_to(root)): sha(p)
            for p in sorted(root.rglob('*')) if p.is_file()}))
        return summary
    except BaseException as exc:
        write_once(root / 'FAILED.json', dict(error_type=type(exc).__name__, error=str(exc),
            completed_method_rows=len(rows), completed_case_ids=sorted({r['pair_id'] for r in rows}),
            calendar_seconds=(datetime.now(timezone.utc) - start).total_seconds()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, required=True); p.add_argument('--source', default='upstream')
    p.add_argument('--dependency-dir'); p.add_argument('--output', required=True)
    p.add_argument('--publication-commit', required=True); p.add_argument('--full', action='store_true')
    p.add_argument('--max-seconds', type=int, default=3600)
    a = p.parse_args(); run(a.method, a.source, a.dependency_dir, a.output, a.publication_commit, a.full, a.max_seconds)
