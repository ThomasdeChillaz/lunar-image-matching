"""Fixed 2,000-update continuation for the distributional coarse128 study.

This library has no launch or reservation authority. A source-published,
bounded external controller must check replay evidence before constructing
the model and pass a deadline check to the training loop. The fitted teacher
domain is the full coarse128 field only.
"""

from dataclasses import asdict, dataclass
from datetime import timedelta
import json
import math
from pathlib import Path

from solar_distributional_training_v88 import (
    PilotSettings,
    fitting_step,
    identical,
    prior_digest,
    sha,
    snapshot,
)


STUDY_FORMAT = 'distributional-coarse128-study-v90'
CHECKPOINT_STEPS = (0, 500, 1000, 1500, 2000)


@dataclass(frozen=True)
class StudySettings(PilotSettings):
    updates: int = 2000

    def validate(self):
        expected = asdict(PilotSettings())
        expected['updates'] = CHECKPOINT_STEPS[-1]
        if asdict(self) != expected:
            raise ValueError('Only the fixed seed20261101 / 2,000-update study is qualified')
        return self


def require_replayed_pilot(pilot_protocol, replay_protocol):
    """Validate the two actual v89 runs using saved artifacts, without Torch."""
    from solar_distributional_pilot_v89 import (
        CANONICAL,
        completed_pilot,
        job_id,
        read,
        utc,
        verify_worker_evidence,
    )
    from solar_learning_pipeline_v73 import journal_state

    pilot = read(pilot_protocol)
    pilot_run = Path(pilot['output'])
    pilot, pilot_watchdog, pilot_summary = completed_pilot(pilot_protocol, pilot_run)
    replay = read(replay_protocol)
    if (replay.get('mode') != 'replay' or replay.get('prepared_only') is not False
            or Path(replay['original_pilot_protocol']) != Path(pilot_protocol)
            or Path(replay['original_pilot_run']) != pilot_run
            or Path(replay['output']) == pilot_run):
        raise ValueError('Separate completed replay of the exact original pilot required')
    for key in ('purpose', 'learning_contract', 'source_sha256', 'fixed_settings',
                'cpu_threads', 'category', 'calibration_or_final_access',
                'compute_ledger', 'compute_ledger_sha256', 'reservation_journal',
                'bank_manifest', 'bank_manifest_sha256', 'bank_prior', 'bank_prior_sha256',
                'bank_complete_sha256', 'bank_run', 'bank_protocol', 'teacher_directory',
                'teacher_sha256', 'expected_frozen_descriptor_sha256'):
        if replay[key] != pilot[key]:
            raise ValueError(f'Exact replay changed the pilot {key}')

    replay_run = Path(replay['output'])
    _, journal, _ = journal_state(pilot['compute_ledger'], pilot['compute_ledger_sha256'], CANONICAL)
    receipts = {}
    for mode, path, protocol, output in (
        ('pilot', pilot_protocol, pilot, pilot_run),
        ('replay', replay_protocol, replay, replay_run),
    ):
        marker_path = output / 'COMPLETE.json'
        watchdog_path = output / 'watchdog.json'
        worker_path = output / 'worker_summary.json'
        marker, watchdog, worker = map(read, (marker_path, watchdog_path, worker_path))
        digest = sha(path)
        if (marker.get('external_watchdog_passed') is not True
                or watchdog.get('complete') is not True or worker.get('complete') is not True
                or worker.get('model_selection_performed') is not False
                or any(item['protocol_sha256'] != digest for item in (marker, watchdog, worker))
                or marker['watchdog_sha256'] != sha(watchdog_path)
                or marker['worker_summary_sha256'] != sha(worker_path)):
            raise ValueError(f'Incomplete or changed v89 {mode} completion evidence')
        started, ended, deadline = map(utc, (watchdog['started_utc'], watchdog['ended_utc'],
                                            watchdog['hard_deadline_utc']))
        cap = protocol['max_calendar_seconds']
        elapsed = watchdog['calendar_seconds']
        if (not 0 < elapsed <= cap or not math.isfinite(elapsed)
                or watchdog['cap_seconds'] != cap or watchdog['child_returncode'] != 0
                or watchdog['failure'] is not None
                or not utc(protocol['created_utc']) <= started <= utc(protocol['latest_start_utc'])
                or deadline != min(started + timedelta(seconds=cap), utc(protocol['absolute_deadline_utc']))
                or not started < ended < deadline):
            raise ValueError(f'v89 {mode} exceeded its original deadline or failed')
        reservation = read(output / 'reservation.json')
        charges = [entry for entry in journal['reservations'] if entry['job_id'] == job_id(path, protocol)]
        if (reservation != watchdog['compute_reservation'] or charges != [reservation['reservation']]
                or charges[0]['category'] != 'training' or charges[0]['seconds'] != cap
                or not started <= utc(charges[0]['reserved_utc']) <= ended):
            raise ValueError(f'v89 {mode} lacks its separate original training charge')
        receipts[mode] = dict(protocol_sha256=digest, complete_sha256=sha(marker_path),
                              watchdog_sha256=sha(watchdog_path), worker_summary_sha256=sha(worker_path),
                              reservation_sha256=sha(output / 'reservation.json'))

    replay_summary = read(replay_run / 'worker_summary.json')
    verify_worker_evidence(replay, replay_run, replay_summary)
    if replay_summary['environment'] != pilot_summary['environment']:
        raise ValueError('v89 exact replay must use the original environment')
    receipts['replay']['exact_replay_sha256'] = replay_summary['replay_sha256']
    receipts['replay']['replay_trace_sha256'] = replay_summary['replay_trace_sha256']
    return dict(pilot=pilot, replay=replay, receipts=receipts,
                environment=pilot_summary['environment'],
                measured_study_cap_seconds=math.ceil(40 * pilot_watchdog['calendar_seconds']))


def _write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def train_study(model, optimizer, rng, bank, output, *, expected_coarse_digest,
                pilot_protocol, replay_protocol, environment, provenance, clock,
                settings=StudySettings()):
    """Run a single admitted seed. Match every pilot row and both endpoint states."""
    settings.validate()
    clock()
    proof = require_replayed_pilot(pilot_protocol, replay_protocol)
    pilot = proof['pilot']
    if (bank.manifest_sha256 != pilot['bank_manifest_sha256']
            or bank.prior_sha256 != pilot['bank_prior_sha256']
            or expected_coarse_digest != pilot['expected_frozen_descriptor_sha256']
            or environment != proof['environment']
            or provenance.get('split') != 'training'):
        raise ValueError('Study must preserve the qualified pilot data, teacher and environment')

    import torch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original = Path(pilot['output']) / 'science'
    completed_updates = 0
    try:
        clock()
        bank.validate_model(model, expected_coarse_digest)
        prior_seal = prior_digest(model)
        usable_counts = bank.preflight()
        with (original / 'history_100.json').open() as handle:
            prefix = json.load(handle)

        def require_pilot_state(step):
            original_state = torch.load(original / f'step_{step}.pt', map_location='cpu', weights_only=True)
            if not identical(snapshot(model, optimizer, rng, step, PilotSettings()), original_state):
                raise ValueError(f'Study model, optimizer, prior or RNG differs at pilot step {step}')

        def save_checkpoint(step):
            clock()
            state = snapshot(model, optimizer, rng, step, settings)
            state['format'] = STUDY_FORMAT
            with (output / f'step_{step}.pt').open('xb') as handle:
                torch.save(state, handle)

        _write_new(output / 'execution.json', dict(
            format=STUDY_FORMAT, settings=asdict(settings), provenance=provenance,
            checkpoint_steps=CHECKPOINT_STEPS, environment=environment,
            pilot_replay_evidence=proof['receipts'], bank_manifest_sha256=bank.manifest_sha256,
            bank_prior_sha256=bank.prior_sha256, coarse_descriptor_sha256=expected_coarse_digest,
            target_usable_counts=usable_counts, parameter_report=model.parameter_report,
            teacher_domain='full-coarse128-field-only', calibration_or_final_access=False,
        ))
        require_pilot_state(0)
        save_checkpoint(0)
        with (output / 'trace.jsonl').open('x') as trace:
            for step in range(1, settings.updates + 1):
                clock()
                record = dict(step=step, **fitting_step(
                    model, optimizer, bank, rng, expected_coarse_digest=expected_coarse_digest,
                    expected_prior_digest=prior_seal, settings=settings, clock=clock,
                ))
                completed_updates = step
                trace.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
                trace.flush()
                if step <= len(prefix) and not identical(record, prefix[step - 1]):
                    raise ValueError(f'Study sampling, loss or gradient differs at pilot step {step}')
                if step == len(prefix):
                    require_pilot_state(step)
                if step in CHECKPOINT_STEPS:
                    save_checkpoint(step)

        files = [f'step_{step}.pt' for step in CHECKPOINT_STEPS] + ['execution.json', 'trace.jsonl']
        summary = dict(
            complete=True, format=STUDY_FORMAT, updates=settings.updates, seed=settings.seed,
            checkpoint_steps=CHECKPOINT_STEPS, exact_pilot_prefix_rows=100,
            exact_pilot_prefix_endpoint_states=[0, 100], pilot_replay_evidence=proof['receipts'],
            bank_manifest_sha256=bank.manifest_sha256, bank_prior_sha256=bank.prior_sha256,
            teacher_domain='full-coarse128-field-only', calibration_or_final_access=False,
            model_selection_performed=False, model_promoted=False,
            artifacts={name: sha(output / name) for name in files},
        )
        clock()
        _write_new(output / 'summary.json', summary)
        return summary
    except BaseException as error:
        _write_new(output / 'FAILED.json', dict(type=type(error).__name__, message=str(error),
                   completed_updates=completed_updates, all_partial_artifacts_preserved=True))
        raise
