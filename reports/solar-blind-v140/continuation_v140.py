"""Prepare a complete unchanged-method continuation; launch needs new approval.

There is no approval issuer or default allowance in this program. The prior
failed attempt, fixed reservations, source bytes and data roster stay unchanged.
"""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HOME = Path('reports/solar-blind-v140')
WORK = Path('runs/solar-blind-v140')
PRIVATE = Path('runs/solar-blind-v140-allowance')
LEDGER = PRIVATE/'ledger.json'
JOURNAL = PRIVATE/'reservations.json'
SPEC = spec_from_file_location('frozen_controller_v139_for_v140', ROOT/'reports/solar-blind-v139/study_v139.py')
old = module_from_spec(SPEC)
SPEC.loader.exec_module(old)
read, sha, check, write = old.read, old.sha, old.check, old.write
CAP = 900


def prior_state():
    old.original_guard()
    previous = read(old.HOME/'protocol.json')
    old.pins(previous)
    parent = read(old.JOURNAL)
    check(parent['ledger_sha256'] == sha(old.LEDGER)
          and len(parent['reservations']) == 3
          and sum(r['seconds'] for r in parent['reservations']) == 2400
          and all(r['category'] == 'baseline' and r['refund_on_failure'] is False
                  for r in parent['reservations']), 'Preserve all original supplementary charges')
    result = read(old.HOME/'result.json')
    check(result['status'] == 'stopped_at_full_cost_gate'
          and result['lunar_inference_run'] is False
          and not (old.WORK/'predictions.json').exists(), 'Preserve the pre-inference failed attempt')
    return previous, parent


def build():
    previous, _ = prior_state()
    sources = dict(previous['sources'])
    for name in ('continuation_v140.py', 'test_continuation_v140.py', 'NEXT_EXPERIMENT.md'):
        p = HOME/name
        sources[str(p)] = sha(p)
    inputs = dict(previous['inputs'])
    for name in ('protocol.json', 'timing.json', 'timing-runtime.json', 'result.json', 'verification.json'):
        p = old.HOME/name
        inputs[str(p)] = sha(p)
    return dict(schema='prepared_unchanged_blind_continuation_v140', prepared_only=True,
        preparation_is_not_execution_authority=True, requested_cap_seconds=CAP,
        parent_journal_sha256=sha(old.JOURNAL), prior_fixed_supplement_charges_seconds=2400,
        sources=sources, inputs=inputs, inference_manifest=previous['inference_manifest'],
        scoring_manifest=previous['scoring_manifest'], case_ids=previous['case_ids'],
        methods=previous['methods'], comparison_cells=48, extra_patch_budget=32,
        native_pixels=512, cpu_threads=2, opencv_threads=1, paid_compute=False,
        training=False, calibration_or_final_access=False, query_truth_inputs=False,
        known_query_sun=False, accepted_pose_study=False,
        matching_scoring_sources_unchanged_from='aaa3d02ed73667c352eaa92f4d142e4adb2c7b6c',
        timing='Repeat exact full-budget two-control v139 pilot; 1.5*8*worst pair+30 must fit original remaining deadline',
        full_roster_or_no_lunar_inference=True, original_requirements_unchanged=True)


def validate_approval(ledger, protocol):
    check(ledger.get('schema') == 'explicit_user_authorized_blind_continuation_v140'
          and ledger.get('user_approved') is True and ledger.get('accounting_complete') is True,
          'New explicit resource approval required; preparation does not authorize execution')
    check(ledger.get('remaining_baseline_wall_seconds') == CAP
          and ledger.get('remaining_training_wall_seconds') == 0
          and ledger.get('paid_compute') is False
          and ledger.get('calibration_or_final_access') is False
          and ledger.get('threads_per_scientific_job') == 2,
          'Approval cannot change the fixed cap or scientific scope')
    check(ledger.get('parent_journal_sha256') == protocol['parent_journal_sha256']
          and ledger.get('protocol_sha256') == sha(HOME/'protocol.json'),
          'Approval must bind the frozen protocol and all prior charges')
    check(isinstance(ledger.get('user_approval_source'), str) and ledger['user_approval_source'].strip()
          and isinstance(ledger.get('approval_text'), str) and ledger['approval_text'].strip(),
          'Actual user resource decision and provenance required')


def guard(protocol):
    prior_state()
    old.pins(protocol)
    check(sha(old.JOURNAL) == protocol['parent_journal_sha256'], 'Parent accounting changed')
    check(all(p.resolve() == (ROOT/p).absolute() for p in (PRIVATE, LEDGER, JOURNAL)),
          'No aliases of private approval or reservation paths')


def published(commit, protocol):
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'refs/remotes/origin/main'], check=True)
    for name in (*protocol['sources'], str(HOME/'protocol.json')):
        check(subprocess.check_output(['git', 'show', commit+':'+name]) == Path(name).read_bytes(),
              'Source differs from public freeze: '+name)


def deadline(reservation):
    check(reservation['seconds'] == CAP and reservation['category'] == 'baseline'
          and reservation['refund_on_failure'] is False, 'Single fixed900-second reservation required')
    start = datetime.fromisoformat(reservation['reserved_utc'])
    check(start.tzinfo is not None, 'Timezone-aware original deadline required')
    return start+timedelta(seconds=CAP)


def sealed(protocol):
    raw, seal = read(WORK/'predictions.json'), read(WORK/'PREDICTIONS_SEALED.json')
    check(seal['sha256'] == sha(WORK/'predictions.json')
          and seal['rows'] == len(raw['rows']) == len(protocol['case_ids']) == 8
          and [r['id'] for r in raw['rows']] == protocol['case_ids']
          and raw['query_truth_used'] is False, 'All eight truth-free predictions must be sealed before scoring')


def worker(operation, ticket_path):
    import resource
    ticket, protocol = read(ticket_path), read(HOME/'protocol.json')
    guard(protocol)
    check(ticket['parent_pid'] == os.getppid() and ticket['protocol_sha256'] == sha(HOME/'protocol.json'),
          'Original admitted controller and protocol required')
    check(sha(LEDGER) == ticket['approval_sha256'], 'Approval changed')
    validate_approval(read(LEDGER), protocol)
    journal = read(JOURNAL)
    check(journal['ledger_sha256'] == ticket['approval_sha256']
          and journal['reservations'] == [ticket['reservation']], 'Original single reservation required')
    stop = deadline(ticket['reservation'])
    check(ticket['deadline'] == stop.isoformat() and datetime.now(timezone.utc) < stop, 'Original deadline expired')
    lease, actual = os.fstat(ticket['lease_fd']), (old.PRIVATE/'scientific.lock').stat()
    check((lease.st_dev, lease.st_ino) == (actual.st_dev, actual.st_ino), 'Inherited serial scientific lease required')
    signal.setitimer(signal.ITIMER_REAL, max(.001, (stop-datetime.now(timezone.utc)).total_seconds()))
    published(ticket['commit'], protocol)
    sys.path.insert(0, str(ROOT/old.HOME))
    import engine_v139 as engine
    engine.cv2.setNumThreads(1)
    def checked(name):
        check(name in protocol['inputs'] and sha(name) == protocol['inputs'][name], 'Changed/unapproved input: '+name)
        return Path(name)
    put = lambda name, value: write(WORK/name, value)
    get = lambda name: read(WORK/name)
    tick = time.perf_counter()
    if operation == 'timing': engine.timing(put)
    elif operation == 'predict': engine.predict(protocol, put, checked)
    elif operation == 'score': sealed(protocol); engine.score(protocol, put, checked, get)
    elif operation == 'replay': sealed(protocol); engine.replay(protocol, put, checked, get)
    else: raise ValueError('Unknown operation')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    put(operation+'-runtime.json', dict(elapsed_seconds=time.perf_counter()-tick,
        whole_worker_peak_rss_bytes=int(peak if sys.platform == 'darwin' else peak*1024),
        memory_scope='Whole worker including shared caches/imports; not per-model memory ranking'))


def execute(commit, approval_sha256):
    from solar_compute_budget_v67 import reserve
    protocol = read(HOME/'protocol.json')
    guard(protocol)
    published(commit, protocol)
    check(sha(LEDGER) == approval_sha256, 'New private user approval required')
    validate_approval(read(LEDGER), protocol)
    check(not WORK.exists() and not WORK.is_symlink(), 'Preserve prior attempts; no overwrites')
    with (old.PRIVATE/'scientific.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check(not JOURNAL.exists(), 'A previous continuation reservation cannot be retried or refunded')
        reservation = reserve(LEDGER, approval_sha256, JOURNAL,
                              sha(HOME/'protocol.json')+'/blind-continuation-v140', 'baseline', CAP)['reservation']
        stop = deadline(reservation)
        WORK.mkdir()
        ticket = dict(commit=commit, parent_pid=os.getpid(), lease_fd=lease.fileno(),
            protocol_sha256=sha(HOME/'protocol.json'), approval_sha256=approval_sha256,
            reservation=reservation, deadline=stop.isoformat())
        write(WORK/'execution.json', ticket)
        tick = time.perf_counter()
        try:
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                VECLIB_MAXIMUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONPATH=str(ROOT))
            for operation in ('timing', 'predict', 'score', 'replay'):
                remaining = (stop-datetime.now(timezone.utc)).total_seconds()
                check(remaining > 0, 'Original deadline expired')
                if operation == 'predict':
                    check(old.forecast_fits(read(WORK/'timing.json'), remaining),
                          'Full comparison cannot fit; do not substitute a smaller lunar subset')
                with (WORK/(operation+'.log')).open('x') as log:
                    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', operation,
                        '--ticket', str(WORK/'execution.json')], env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                    try: status = child.wait(timeout=remaining)
                    except BaseException:
                        os.killpg(child.pid, signal.SIGKILL); child.wait(); raise
                check(status == 0 and datetime.now(timezone.utc) < stop, 'Worker failed or original deadline elapsed')
                if operation == 'predict':
                    write(WORK/'PREDICTIONS_SEALED.json', dict(rows=8, sha256=sha(WORK/'predictions.json')))
            guard(protocol)
            write(WORK/'COMPLETE.json', dict(ticket, complete=True, elapsed_seconds=time.perf_counter()-tick))
        except BaseException as error:
            write(WORK/'FAILED.json', dict(ticket, complete=False, error=type(error).__name__, message=str(error)))
            raise


if __name__ == '__main__':
    check(Path.cwd() == ROOT, 'Run from the original repository root')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('freeze')
    run = sub.add_parser('run'); run.add_argument('--commit', required=True); run.add_argument('--approval-sha256', required=True)
    job = sub.add_parser('worker'); job.add_argument('operation', choices=('timing', 'predict', 'score', 'replay'))
    job.add_argument('--ticket', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'freeze': write(HOME/'protocol.json', build())
    elif args.action == 'run': execute(args.commit, args.approval_sha256)
    else: worker(args.operation, args.ticket)
