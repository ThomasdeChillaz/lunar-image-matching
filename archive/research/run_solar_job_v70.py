"""External deadline for one published teacher-extraction or training invocation."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def run(protocol_path, commit, python, program, replay_root=None, replay_output=None):
    p = json.loads(Path(protocol_path).read_text())
    sha = lambda n: hashlib.sha256(Path(n).read_bytes()).hexdigest()
    mode = 'replay' if replay_root else 'run'
    if program not in ('train_solar_response_v63.py', 'build_solar_training_bank_v68.py') or p['cpu_threads'] != 2:
        raise ValueError('One admitted two-thread scientific program required')
    if replay_root and (program != 'train_solar_response_v63.py' or not replay_output or replay_output != p['exact_replay_output']):
        raise ValueError('Published pilot replay output required')
    for name in ('run_solar_job_v70.py', program):
        if sha(name) != p['source_sha256'][name] or subprocess.check_output(['git', 'show', commit + ':' + name]) != Path(name).read_bytes():
            raise ValueError('Watchdog or scientific source differs from publication')
    if subprocess.check_output(['git', 'show', commit + ':' + protocol_path]) != Path(protocol_path).read_bytes():
        raise ValueError('Invocation protocol is not published')
    receipt = Path(p['watchdog_receipts'][mode])
    log_path = Path(p['watchdog_log_prefix'] + mode + '.log')
    if receipt.exists() or log_path.exists():
        raise FileExistsError('Preserve prior receipts/logs; no automatic job restart')
    cap = p['replay_cap_seconds'] if replay_root else p['max_calendar_seconds']
    if type(cap) is not int or cap <= 0:
        raise ValueError('Explicit positive fixed allowance required')
    started = datetime.now(timezone.utc)
    latest_key = 'replay_latest_start_utc' if replay_root else 'latest_start_utc'
    if latest_key in p and started > datetime.fromisoformat(p[latest_key]):
        raise TimeoutError('Original published dispatch lease expired; never reset the job deadline')
    deadline = min(started + timedelta(seconds=cap), datetime.fromisoformat(p['replay_absolute_deadline_utc'] if replay_root else p['absolute_deadline_utc']))
    if started >= deadline:
        raise TimeoutError('Original published job deadline expired')
    command = [python, program, '--protocol', protocol_path, '--publication-commit', commit]
    if replay_root:
        command += ['--replay-pilot', replay_root, '--replay-output', replay_output]
    pending = []
    def interrupted(signum, frame):
        pending.append(signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    tick = time.monotonic()
    child = None
    awake = None
    complete = False
    failure = None
    try:
        with log_path.open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2'))
            awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(child.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while child.poll() is None:
                if pending:
                    raise InterruptedError('Stop requested; preserve all scientific outputs')
                if datetime.now(timezone.utc) >= deadline or time.monotonic() - tick >= cap:
                    raise TimeoutError('Original scientific job allowance exhausted')
                time.sleep(.25)
            if pending or datetime.now(timezone.utc) >= deadline or time.monotonic() - tick >= cap:
                raise TimeoutError('Job interrupted or completed after its fixed allowance')
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)
            complete = True
    except BaseException as exc:
        failure = dict(error_type=type(exc).__name__, error=str(exc))
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise
    finally:
        if awake is not None and awake.poll() is None:
            awake.terminate()
        record = dict(complete=complete, mode=mode, program=program,
            protocol_sha256=sha(protocol_path), publication_commit=commit,
            source_sha256=sha(__file__), started_utc=started.isoformat(),
            absolute_deadline_utc=deadline.isoformat(), cap_seconds=cap,
            calendar_seconds=time.monotonic()-tick,
            child_pid=None if child is None else child.pid,
            child_returncode=None if child is None else child.returncode,
            failure=failure, source_log_sha256=sha(log_path) if log_path.exists() else None,
            scope='Deadline/termination receipt, not an accuracy or replay claim.')
        with receipt.open('x') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True)
    p.add_argument('--publication-commit', required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--program', required=True)
    p.add_argument('--replay-pilot')
    p.add_argument('--replay-output')
    a = p.parse_args()
    run(a.protocol, a.publication_commit, a.python, a.program, a.replay_pilot, a.replay_output)
