"""Source-gated, one-seed controller for the qualified coarse128 study."""

import argparse
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from solar_distributional_pilot_v89 import CANONICAL, metadata_paths, now, read, sha, utc, write


RUNNER = 'solar_distributional_study_runner_v90.py'
LIBRARY = 'solar_distributional_study_v90.py'
THREADS = dict(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               VECLIB_MAXIMUM_THREADS='2')


def job_id(path):
    return sha(path) + '/distributional2000_study'


def binding(settings):
    from solar_coarse_bank_v88 import validate_completed_bank
    from solar_distributional_study_v90 import CHECKPOINT_STEPS, STUDY_FORMAT, StudySettings, require_replayed_pilot

    pilot_path = settings['pilot_protocol']
    replay_path = settings['replay_protocol']
    output = Path(settings['output'])
    runs = Path.cwd() / 'runs'
    if output.is_absolute() or '..' in output.parts or not output.resolve().is_relative_to(runs):
        raise ValueError('Study output must remain under the repository runs directory')
    proof = require_replayed_pilot(pilot_path, replay_path)
    pilot, replay = proof['pilot'], proof['replay']
    bank = validate_completed_bank(pilot['bank_manifest'])
    if (bank['manifest_sha256'] != pilot['bank_manifest_sha256']
            or bank['complete_sha256'] != pilot['bank_complete_sha256']
            or bank['manifest']['view_prior_sha256'] != pilot['bank_prior_sha256']):
        raise ValueError('The exact completed full training bank and prior are required')

    metadata = metadata_paths(replay) | {pilot_path, replay_path}
    metadata.add(str(Path(pilot['output']) / 'reservation.json'))
    replay_output = Path(replay['output'])
    metadata.update(str(replay_output / name) for name in (
        'COMPLETE.json', 'watchdog.json', 'worker_summary.json', 'reservation.json',
        'replay.json', 'replay_trace.jsonl',
    ))
    return dict(
        pilot_protocol=pilot_path, replay_protocol=replay_path, output=str(output),
        purpose=STUDY_FORMAT, category='training', cpu_threads=2,
        calibration_or_final_access=False,
        fixed_settings=asdict(StudySettings()), checkpoint_steps=list(CHECKPOINT_STEPS),
        learning_contract=pilot['learning_contract'], reservation_journal=CANONICAL,
        compute_ledger=pilot['compute_ledger'], compute_ledger_sha256=pilot['compute_ledger_sha256'],
        max_calendar_seconds=proof['measured_study_cap_seconds'],
        pilot_replay_evidence=proof['receipts'],
        source_sha256=pilot['source_sha256'] | {RUNNER: sha(RUNNER), LIBRARY: sha(LIBRARY)},
        metadata_sha256={name: sha(name) for name in sorted(metadata)},
    ), proof


def prepare(settings_path, output):
    from solar_learning_pipeline_v73 import load_contract, remaining

    fields, _ = binding(read(settings_path))
    if fields['max_calendar_seconds'] > remaining(load_contract(fields['learning_contract']))['training']:
        raise ValueError('Measured study cap exceeds original cumulative training allowance')
    write(output, dict(fields, prepared_only=True, publication_dispatch_lease_seconds=1800,
                       created_utc=None, latest_start_utc=None, absolute_deadline_utc=None))


def verify_protocol(path, protocol, commit, deadline):
    """Study commit publishes sources and the protocol's exact input digests."""
    fixed, proof = binding(protocol)
    if any(protocol.get(name) != value for name, value in fixed.items()):
        raise ValueError('Frozen study protocol or a bound input changed')
    if len(commit) != 40 or any(character not in '0123456789abcdef' for character in commit):
        raise ValueError('Exact study publication commit required')

    def command(*args):
        remaining = (deadline - now()).total_seconds()
        if remaining <= 0:
            raise TimeoutError('Study publication verification exceeded its original deadline')
        return subprocess.check_output(['git', *args], timeout=min(10, remaining))

    command('merge-base', '--is-ancestor', commit, 'refs/remotes/origin/main')
    for name, digest in protocol['source_sha256'].items():
        if sha(name) != digest or command('show', f'{commit}:{name}') != Path(name).read_bytes():
            raise ValueError(f'Study source differs from publication: {name}')
    for name in (str(path), protocol['pilot_protocol'], protocol['replay_protocol']):
        if command('show', f'{commit}:{name}') != Path(name).read_bytes():
            raise ValueError(f'Study or historical protocol differs from publication: {name}')
    if now() >= deadline:
        raise TimeoutError('Study verification exceeded its original deadline')
    return proof


def verify_output(protocol, root):
    from solar_distributional_study_v90 import CHECKPOINT_STEPS, STUDY_FORMAT

    science = root / 'science'
    summary = read(science / 'summary.json')
    expected = dict(complete=True, format=STUDY_FORMAT, updates=2000, seed=20261101,
                    checkpoint_steps=list(CHECKPOINT_STEPS), exact_pilot_prefix_rows=100,
                    exact_pilot_prefix_endpoint_states=[0, 100], model_selection_performed=False,
                    teacher_domain='full-coarse128-field-only', calibration_or_final_access=False,
                    model_promoted=False, pilot_replay_evidence=protocol['pilot_replay_evidence'])
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError('Study completion or exact pilot prefix evidence differs')
    pilot = read(protocol['pilot_protocol'])
    for key in ('bank_manifest_sha256', 'bank_prior_sha256'):
        if summary[key] != pilot[key]:
            raise ValueError('Study output changed qualified data identity')
    required = {'execution.json', 'trace.jsonl'} | {f'step_{step}.pt' for step in CHECKPOINT_STEPS}
    if set(summary['artifacts']) != required:
        raise ValueError('Study needs all five checkpoints and original traces')
    for name, digest in summary['artifacts'].items():
        if sha(science / name) != digest:
            raise ValueError(f'Study artifact differs: {name}')
    trace = [json.loads(row) for row in (science / 'trace.jsonl').read_text().splitlines()]
    prefix = read(Path(pilot['output']) / 'science/history_100.json')
    if [row['step'] for row in trace] != list(range(1, 2001)) or trace[:100] != prefix:
        raise ValueError('Study trace lacks all 2,000 rows and exact original 100-row prefix')
    return sha(science / 'summary.json')


def worker(path, commit, controller_pid):
    from solar_learning_pipeline_v73 import journal_state, load_contract

    protocol = read(path)
    root = Path(protocol['output'])
    execution = read(root / 'execution.json')
    if (os.getppid() != controller_pid or execution['controller_pid'] != controller_pid
            or execution['protocol_sha256'] != sha(path) or execution['publication_commit'] != commit):
        raise ValueError('Original admitted external study controller required')
    _, journal, _ = journal_state(protocol['compute_ledger'], protocol['compute_ledger_sha256'], CANONICAL)
    charges = [row for row in journal['reservations'] if row['job_id'] == job_id(path)]
    if (charges != [read(root / 'reservation.json')['reservation']]
            or charges[0]['seconds'] != protocol['max_calendar_seconds'] or charges[0]['category'] != 'training'):
        raise ValueError('Original one-shot study charge required')
    deadline = utc(execution['work_deadline_utc'])

    def clock():
        if now() >= deadline:
            raise TimeoutError('Original study work allowance exhausted')

    with (root / 'WORKER_STARTED').open('x') as marker:
        marker.write(f'{os.getpid()}\n')
    clock()
    proof = verify_protocol(path, protocol, commit, deadline)
    pilot = proof['pilot']

    import cv2
    import numpy as np
    import torch
    from compact_lunar import load_checkpoint
    from solar_distributional_study_v90 import StudySettings, train_study
    from solar_distributional_training_v88 import CoarseTrainingBank, initialize

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    environment = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                       opencv=cv2.__version__, platform=platform.platform(), processor=platform.processor(),
                       torch_threads=torch.get_num_threads())
    if environment != proof['environment']:
        raise ValueError('Exact study prefix requires the actual pilot/replay environment')
    contract = load_contract(protocol['learning_contract'])
    bank = CoarseTrainingBank(
        Path.cwd(), pilot['bank_manifest'], pilot['bank_manifest_sha256'],
        allowed_training_groups=contract['training_groups'],
        eligible_unit_indices=contract['eligible_training_unit_indices'],
        expected_teacher_sha256=pilot['expected_frozen_descriptor_sha256'], cache_units=4, clock=clock,
    )
    for name, digest in pilot['teacher_sha256'].items():
        clock()
        if sha(Path(pilot['teacher_directory']) / name) != digest:
            raise ValueError(f'Qualified teacher changed: {name}')
    frozen, _ = load_checkpoint(pilot['teacher_directory'])
    settings = StudySettings()
    model, optimizer, rng, digest = initialize(frozen, bank.prior, settings)
    train_study(
        model, optimizer, rng, bank, root / 'science', expected_coarse_digest=digest,
        pilot_protocol=protocol['pilot_protocol'], replay_protocol=protocol['replay_protocol'],
        environment=environment, settings=settings, clock=clock,
        provenance=dict(split='training', protocol_path=str(path), protocol_sha256=sha(path),
                        publication_commit=commit, source_sha256=protocol['source_sha256']),
    )
    clock()
    digest = verify_output(protocol, root)
    clock()
    write(root / 'worker_summary.json', dict(complete=True, protocol_sha256=sha(path),
                                             science_summary_sha256=digest, environment=environment))


def run(path, commit, python):
    started, tick = now(), time.monotonic()
    os.environ.update(THREADS)
    protocol = read(path)
    cap = protocol['max_calendar_seconds']
    if protocol.get('prepared_only') is not False or type(cap) is not int or cap <= 5:
        raise ValueError('Publish a finalized study protocol before launch')
    created, latest, absolute = map(utc, (protocol['created_utc'], protocol['latest_start_utc'],
                                         protocol['absolute_deadline_utc']))
    lease = protocol['publication_dispatch_lease_seconds']
    if (type(lease) is not int or not 0 < lease <= 1800 or latest != created + timedelta(seconds=lease)
            or absolute != latest + timedelta(seconds=cap) or not created <= started <= latest):
        raise ValueError('Original finite study dispatch lease expired or changed')
    deadline = min(started + timedelta(seconds=cap), absolute)
    work_deadline = deadline - timedelta(seconds=5)
    verify_protocol(path, protocol, commit, work_deadline)

    from solar_compute_budget_v67 import reserve
    from solar_learning_pipeline_v73 import load_contract, remaining

    if cap > remaining(load_contract(protocol['learning_contract']))['training']:
        raise ValueError('Measured study cap exceeds original remaining training allowance')

    root = Path(protocol['output'])
    root.mkdir(parents=True, exist_ok=False)
    child, charge, failure, science_digest = None, None, None, None
    pending = []
    old_signals = {sig: signal.signal(sig, lambda number, frame: pending.append(number))
                   for sig in (signal.SIGINT, signal.SIGTERM)}

    def check_work():
        if pending or now() >= work_deadline or time.monotonic() - tick >= cap - 5:
            raise TimeoutError('Original study work allowance exhausted or interrupted')

    try:
        check_work()
        charge = reserve(protocol['compute_ledger'], protocol['compute_ledger_sha256'], CANONICAL,
                         job_id(path), 'training', cap)
        write(root / 'reservation.json', charge)
        write(root / 'execution.json', dict(protocol_sha256=sha(path), publication_commit=commit,
                                            controller_pid=os.getpid(), started_utc=started.isoformat(),
                                            work_deadline_utc=work_deadline.isoformat(),
                                            hard_deadline_utc=deadline.isoformat(), cap_seconds=cap))
        check_work()
        with (root / 'worker.log').open('x') as log:
            child = subprocess.Popen(
                [python, RUNNER, 'worker', '--protocol', str(path), '--publication-commit', commit,
                 '--controller-pid', str(os.getpid())], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=os.environ | THREADS,
            )
            while child.poll() is None:
                check_work()
                time.sleep(0.1)
        check_work()
        if child.returncode != 0:
            raise RuntimeError(f'Study worker exited with code {child.returncode}')
        result = read(root / 'worker_summary.json')
        science_digest = verify_output(protocol, root)
        if (result.get('complete') is not True or result['protocol_sha256'] != sha(path)
                or result['science_summary_sha256'] != science_digest):
            raise ValueError('Study worker completion or science summary changed')
        check_work()
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
    finally:
        if child is not None and child.poll() is None:
            for sig, wait_seconds in ((signal.SIGTERM, 3), (signal.SIGKILL, 1)):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=max(0, min(wait_seconds, (deadline - now()).total_seconds() - 1)))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if child.poll() is None:
                failure = dict(type='CleanupFailure', message='Original study cleanup bound exhausted')
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)

    ended, elapsed = now(), time.monotonic() - tick
    if ended >= deadline or elapsed >= cap:
        failure = dict(type='TimeoutError', message='Original study hard deadline exhausted')
    write(root / 'watchdog.json', dict(
        complete=failure is None, protocol_sha256=sha(path), publication_commit=commit,
        started_utc=started.isoformat(), ended_utc=ended.isoformat(), calendar_seconds=elapsed,
        cap_seconds=cap, hard_deadline_utc=deadline.isoformat(), work_deadline_utc=work_deadline.isoformat(),
        compute_reservation=charge, child_returncode=None if child is None else child.returncode,
        failure=failure,
    ))
    if failure is not None:
        raise RuntimeError(str(failure))
    write(root / 'COMPLETE.json', dict(external_watchdog_passed=True, protocol_sha256=sha(path),
                                       watchdog_sha256=sha(root / 'watchdog.json'),
                                       worker_summary_sha256=sha(root / 'worker_summary.json'),
                                       science_summary_sha256=science_digest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('prepare')
    command.add_argument('--settings', required=True)
    command.add_argument('--output', required=True)
    for name in ('run', 'worker'):
        command = commands.add_parser(name)
        command.add_argument('--protocol', required=True)
        command.add_argument('--publication-commit', required=True)
        if name == 'run':
            command.add_argument('--python', default=sys.executable)
        else:
            command.add_argument('--controller-pid', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.settings, args.output)
    elif args.command == 'run':
        run(args.protocol, args.publication_commit, args.python)
    else:
        worker(args.protocol, args.publication_commit, args.controller_pid)
