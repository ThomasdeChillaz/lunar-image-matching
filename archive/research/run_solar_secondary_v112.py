"""Publish and supervise the fixed v112 acquisition and blind-coverage sentry."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys

from control_solar_png_dither_v105 import renderer_path
from render_solar_clean_radiance_v106 import DEPENDENCIES, digest, read, relative, write
from render_solar_raw_acquisition_v109 import publication
from solar_compute_budget_v67 import reserve
from solar_secondary_acquire_v112 import SOURCE_PATHS as ACQUISITION_CLOSURE

SOURCE = 'run_solar_secondary_v112.py'
ROSTER = 'reports/solar-secondary-ranking-roster-v112.json'
ROSTER_SHA = '59ea302a5bd9f924dd03366ebe66b92b5458ea29a4abb47371ae205ae6e3826f'
LEDGER = 'reports/solar_compute_admission_v67.json'
JOURNAL = 'runs/solar-compute-reservations-v67.json'
ROOT = Path('runs/solar-secondary-acquisition-v112')
WORKERS = {'acquire': 'solar_secondary_acquire_v112.py', 'sentry': 'solar_secondary_sentry_v112.py'}
CAPS = {'acquire': 500, 'sentry': 1200}
LEASE_SECONDS = 3600
PROFILE = 'reports/solar-preselection-compact-controls-cost-pilot-v86.json'
PROFILE_NAME = 'frozen_step0_A1B0C1_quantization_aware'
ACQUIRE_SOURCES = (*DEPENDENCIES, 'render_solar_clean_radiance_v106.py',
                   'render_solar_raw_acquisition_v109.py', 'solar_eval_data_v69.py',
                   'verify_solar_development_v68.py', 'verify_north_camera_qualified_v64.py',
                   'north_camera_visibility_v59.py', 'native_baseline_pose_v7.py',
                   'lunar_geometry_v2.py', 'lunar_geometry_v5.py', 'verify_north_development_v35.py',
                   'localization_robust_pose_v4.py', 'localization_diagnostics_v3.py')
SENTRY_SOURCES = ('solar_blind_beams_v112.py', 'solar_lookup_worker_v70.py', 'solar_map_v63.py',
                  'solar_sun_search_v68.py', 'solar_inference_cache_v84.py', 'solar_frontend_v63.py',
                  'solar_assignment_v63.py', 'localize_terrain.py', 'solar_active_v63.py',
                  'replay_raw_acquisition_v109.py',
                  'reports/solar-secondary-mesh-v112/solar_secondary_mesh_v112.py',
                  'reports/solar-secondary-atlas-v111/solar_secondary_atlas_v111.py',
                  'reports/solar-low-frequency-ranking-v111/solar_low_frequency_ranking_v111.py')


def utc(value):
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError('Timezone-aware timestamp required')
    return stamp.astimezone(timezone.utc)


def dispatch_path(phase):
    return f'reports/solar-secondary-{phase}-dispatch-v112.json'


def checked(path, expected):
    path = relative(str(path))
    if digest(path) != expected:
        raise ValueError(f'Frozen file changed: {path}')
    return path


def roster():
    spec = read(checked(ROSTER, ROSTER_SHA))
    if (spec['calibration_or_final_access'] is not False or not spec['prepared_only']
            or any(row['split'] != 'model_selection' for row in spec['endpoints'])):
        raise ValueError('Only the fixed model-selection design is permitted')
    return spec


def profile():
    spec = roster()
    protocol = read(checked(PROFILE, spec['predecessor_sha256'][PROFILE]))
    row, = [r for r in protocol['profiles'] if r['name'] == PROFILE_NAME]
    checked(row['checkpoint'], row['configuration']['checkpoint_sha256'])
    return row


def source_pins(phase, spec):
    names = (SOURCE, 'control_solar_png_dither_v105.py', 'solar_compute_budget_v67.py',
             WORKERS[phase], *ACQUIRE_SOURCES, *ACQUISITION_CLOSURE)
    if phase == 'sentry':
        names = (*names, *SENTRY_SOURCES, *read('reports/solar_learning_contract_v79.json')['source_sha256'])
    expected = {name: digest(name) for name in sorted(set(names))}
    previous = spec['reused_source_sha256']
    for name, actual in expected.items():
        if name in previous and previous[name] != actual:
            raise ValueError(f'Previously frozen source changed: {name}')
    return expected


def previous_acquisition():
    receipt = ROOT / 'watchdog-acquire.json'
    gate = ROOT / 'gate.json'
    render = ROOT / 'sentry' / 'COMPLETE.json'
    if (read(receipt)['complete'] is not True or read(gate)['passed'] is not True
            or read(render)['complete'] is not True):
        raise ValueError('Require passing acquisition and exact pose gate before blind sentry')
    return {str(path): digest(path) for path in (receipt, gate, render, ROOT/'geometry'/'COMPLETE.json')}


def build(phase, created):
    spec = roster()
    ledger_hash = spec['predecessor_sha256'][LEDGER]
    checked(LEDGER, ledger_hash)
    metadata = {ROSTER: ROSTER_SHA, LEDGER: ledger_hash}
    predecessor = {}
    if phase == 'sentry':
        predecessor = previous_acquisition()
        metadata[PROFILE] = spec['predecessor_sha256'][PROFILE]
        profile()
    return dict(schema=112, phase=phase, cap_seconds=CAPS[phase], root=str(ROOT), split='model_selection',
                cpu_threads=2, calibration_or_final_access=False, renderer='original v109 Blender build',
                source_sha256=source_pins(phase, spec), metadata_sha256=metadata, predecessor_sha256=predecessor,
                created_utc=created.isoformat(), latest_start_utc=(created + timedelta(seconds=LEASE_SECONDS)).isoformat())


def freeze(phase):
    write(dispatch_path(phase), build(phase, datetime.now(timezone.utc)))


def validate(phase, commit):
    path = dispatch_path(phase)
    plan = read(path)
    if plan != build(phase, utc(plan['created_utc'])):
        raise ValueError('Frozen phase or its upstream artifacts changed')
    publication(path, plan, commit)
    now = datetime.now(timezone.utc)
    if not utc(plan['created_utc']) <= now <= utc(plan['latest_start_utc']):
        raise TimeoutError('Original publication-to-launch lease expired')
    return plan


def run_step(command, logfile, deadline):
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               VECLIB_MAXIMUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    with logfile.open('xb') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        try:
            limit = (deadline - datetime.now(timezone.utc)).total_seconds()
            if child.wait(timeout=max(0, limit)) != 0:
                raise RuntimeError(f'Worker step failed. See {logfile}')
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)


def launch(phase, commit):
    plan = validate(phase, commit)
    blender = [str(renderer_path()), '--background', '--factory-startup', '--disable-autoexec', '--threads', '2',
               '--python-exit-code', '2', '--python', WORKERS[phase], '--']
    host = [sys.executable, WORKERS[phase]]
    steps = (('geometry', blender), ('gate', host), ('sentry', blender)) if phase == 'acquire' else (
        ('prepare', host), ('reference', blender), ('blind', host), ('project', blender), ('seal', host))
    if phase == 'acquire':
        ROOT.mkdir()
    watchdog = ROOT / f'watchdog-{phase}.json'
    if watchdog.exists():
        raise FileExistsError('Preserve prior phase and its original deadline')
    path = dispatch_path(phase)
    checksum = digest(path)
    charge = reserve(LEDGER, plan['metadata_sha256'][LEDGER], JOURNAL, checksum + '/secondary-v112',
                     'baseline', CAPS[phase])
    deadline = utc(charge['reservation']['reserved_utc']) + timedelta(seconds=CAPS[phase])
    args = ['--root', str(ROOT), '--publication-commit', commit, '--deadline-utc', deadline.isoformat()]
    if phase == 'acquire':
        args += ['--roster', ROSTER]
    else:
        args += ['--dispatch', path]
    done = []
    error = None
    try:
        for step, prefix in steps:
            step_args = ['--gate-sha256', digest(ROOT/'gate.json')] if phase == 'acquire' and step == 'sentry' else []
            run_step([*prefix, step, *args, *step_args], ROOT / f'{phase}-{step}.log', deadline)
            done.append(step)
            if phase == 'acquire' and step == 'gate' and read(ROOT/'gate.json')['passed'] is not True:
                raise ValueError('Exact pose geometry gate failed')
        target = ROOT / ('sentry/COMPLETE.json' if phase == 'acquire' else 'blind-sentry/COMPLETE.json')
        if read(target)['complete'] is not True:
            raise ValueError('Worker did not seal its full declared output')
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc)}
        raise
    finally:
        write(watchdog, dict(complete=error is None, phase=phase, protocol_sha256=checksum,
                             publication_commit=commit, finished_steps=done, error=error,
                             absolute_deadline_utc=deadline.isoformat(), **charge))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('freeze').add_argument('phase', choices=CAPS)
    runner = sub.add_parser('launch')
    runner.add_argument('phase', choices=CAPS)
    runner.add_argument('--publication-commit', required=True)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.phase)
    else:
        with Path('runs/solar-secondary-v112.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            launch(args.phase, args.publication_commit)


if __name__ == '__main__':
    main()
