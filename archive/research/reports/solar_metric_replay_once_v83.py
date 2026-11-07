"""One-shot local operation; uses the existing metric verifier, no new metric."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def utc(value): return datetime.fromisoformat(value)
def now(): return datetime.now(timezone.utc)
def write_new(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def main(job_path, commit):
    # This outside-repository operational recipe is source-bound by the job.
    sys.path.insert(0, str(Path.cwd()))
    from solar_compute_budget_v67 import reserve
    from solar_learning_pipeline_v73 import CANONICAL_JOURNAL, load_contract, remaining
    job = read(job_path); p = read(job['inference_protocol']); cap = job['max_calendar_seconds']
    if job['controller_source_sha256'] != sha(__file__): raise ValueError('Operational recipe changed')
    if job['prepared_only'] is not False or job['category'] != 'baseline' or job['cpu_threads'] != 2: raise ValueError('Frozen two-thread baseline replay required')
    if type(cap) is not int or cap != (300 if p['purpose'] == 'cost_pilot' else 900): raise ValueError('Original fixed replay cap changed')
    if job['inference_protocol_sha256'] != sha(job['inference_protocol']) or job['inference_run'] != p['output']: raise ValueError('Inference binding differs')
    if utc(job['absolute_deadline_utc']) > utc(p['absolute_deadline_utc']) or utc(job['latest_start_utc'])+timedelta(seconds=cap) > utc(job['absolute_deadline_utc']): raise ValueError('Replay extended the original deadline')
    for name in (job_path, job['inference_protocol'], *job['source_sha256']):
        if subprocess.check_output(['git', 'show', commit+':'+name]) != Path(name).read_bytes(): raise ValueError('Unpublished or changed replay input: '+name)
    for name, h in job['source_sha256'].items():
        if sha(name) != h: raise ValueError('Frozen replay source changed: '+name)
    if job['reservation_journal'] != CANONICAL_JOURNAL or job['reservation_journal'] != p['reservation_journal'] or job['compute_ledger'] != p['compute_ledger'] or job['compute_ledger_sha256'] != p['compute_ledger_sha256']: raise ValueError('Canonical accounting differs')
    if job['learning_contract'] != p['learning_contract'] or job['learning_contract_sha256'] != p['learning_contract_sha256'] or sha(job['learning_contract']) != job['learning_contract_sha256']: raise ValueError('Frozen learning/accounting contract changed')
    contract = load_contract(job['learning_contract'])
    if remaining(contract)['baseline'] < cap: raise ValueError('Original baseline ceiling cannot admit replay')
    root = Path(job['inference_run']); summary = read(root/'summary.json'); seal = read(root/'COMPLETE.json')
    if (root/'FAILED.json').exists() or not summary['complete'] or summary['protocol_sha256'] != sha(job['inference_protocol']) or not seal['external_watchdog_passed'] or seal['summary_sha256'] != sha(root/'summary.json'): raise ValueError('Inference must complete under its original watchdog first')
    ids = p['pilot_case_ids'] if p['purpose'] == 'cost_pilot' else p['case_ids']
    if ids != job['case_ids'] or [x['name'] for x in p['profiles']] != job['profile_names'] or p['tracks'] != job['tracks'] or len(summary['rows']) != job['expected_rows']: raise ValueError('Replay grid differs from the frozen inference grid')
    for key in ('candidate_output', 'output', 'watchdog_receipt', 'reservation_receipt', 'log'):
        if Path(job[key]).exists(): raise FileExistsError('Preserve prior replay outputs; never restart: '+key)
    started = now(); tick = time.monotonic(); absolute = utc(job['absolute_deadline_utc'])
    if started > utc(job['latest_start_utc']) or started+timedelta(seconds=5) >= absolute: raise TimeoutError('Original replay lease expired')
    deadline = min(started+timedelta(seconds=cap), absolute); work_deadline = deadline-timedelta(seconds=5)
    charged = reserve(job['compute_ledger'], job['compute_ledger_sha256'], CANONICAL_JOURNAL, sha(job_path)+'/metric_replay', 'baseline', cap)
    write_new(job['reservation_receipt'], charged)
    child = None; failure = None; complete = False
    def interrupted(signum, frame): raise InterruptedError('Replay interrupted; preserve charge and outputs')
    signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
    command = [sys.executable, 'verify_solar_evaluation_v70.py', job['inference_run'], '--protocol', job['inference_protocol'], '--output', job['candidate_output']]
    try:
        with Path(job['log']).open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2'))
            code = child.wait(timeout=max(0., min((work_deadline-now()).total_seconds(), cap-5-(time.monotonic()-tick))))
        if code or now() >= work_deadline or time.monotonic()-tick >= cap-5: raise RuntimeError('Replay failed or exited after its original work allowance')
        result = read(job['candidate_output'])
        if result.get('passed') is not True or result['replayed_rows'] != job['expected_rows'] or result['summary_sha256'] != sha(root/'summary.json') or result['replay_source_sha256'] != job['source_sha256']['verify_solar_evaluation_v70.py']: raise ValueError('Actual replay evidence differs')
        complete = True
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, message=str(exc))
    finally:
        if child is not None and child.poll() is None:
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: child.wait(timeout=max(0., min(3., (deadline-now()).total_seconds()-1.)))
            except subprocess.TimeoutExpired:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                try: child.wait(timeout=max(0., min(1., (deadline-now()).total_seconds())))
                except subprocess.TimeoutExpired: failure = dict(type='CleanupFailure', message='No unbounded cleanup wait permitted'); complete = False
        if now() >= deadline or time.monotonic()-tick >= cap: complete = False; failure = dict(type='TimeoutError', message='Original replay hard allowance exhausted')
        receipt = dict(complete=complete, job_sha256=sha(job_path), inference_protocol_sha256=sha(job['inference_protocol']), controller_source_sha256=sha(__file__),
            publication_commit=commit, started_utc=started.isoformat(), ended_utc=now().isoformat(), absolute_deadline_utc=deadline.isoformat(), work_deadline_utc=work_deadline.isoformat(),
            original_absolute_deadline_utc=job['absolute_deadline_utc'], cap_seconds=cap, calendar_seconds=time.monotonic()-tick, reservation_sha256=sha(job['reservation_receipt']),
            child_pid=None if child is None else child.pid, child_returncode=None if child is None else child.returncode, failure=failure,
            candidate_sha256=sha(job['candidate_output']) if Path(job['candidate_output']).exists() else None,
            scope='Independent saved-metric arithmetic only; no fresh matching, pose estimation, training or rendering; cleanup is inside the charged fixed cap.')
        write_new(job['watchdog_receipt'], receipt)
    if not complete: raise SystemExit(json.dumps(failure))
    # Exclusive promotion occurs only after timely external success. Failed or
    # late candidate bytes remain preserved and cannot qualify the next stage.
    os.link(job['candidate_output'], job['output'])
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main(sys.argv[1], sys.argv[2])
