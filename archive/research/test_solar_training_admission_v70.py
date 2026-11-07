"""Admission evidence tests use JSON fixtures; no model or image inference."""
import json
from pathlib import Path
import pytest
from solar_training_admission_v70 import sha, validate_pilot_evidence


def fixture(tmp_path):
    def write(name, value):
        path = tmp_path/name
        path.write_text(json.dumps(value))
        return str(path)
    common = dict(bank_sha256='bank', source_sha256={'run_solar_job_v70.py':'hash'}, initializer_sha256={'selected.pt':'frozen'}, pooled=False, cpu_threads=2)
    pilot = write('pilot.json', dict(common, updates=100, seed=20261101))
    checkpoint = tmp_path/'step_100.pt'
    checkpoint.write_bytes(b'procedural fixture only')
    initial=tmp_path/'step_0.pt';initial.write_bytes(b'initial fixture only')
    summary = write('summary.json', dict(complete=True, updates=100, protocol_sha256=sha(pilot), calendar_seconds=2., checkpoint_sha256={'step_0.pt':sha(initial),'step_100.pt':sha(checkpoint)}))
    replay = write('replay.json', dict(passed=True, updates=100, exact_model_optimizer_rng_loss_gradient_sampler=True, protocol_sha256=sha(pilot), pilot_summary_sha256=sha(summary)))
    receipt=dict(complete=True,protocol_sha256=sha(pilot),program='train_solar_response_v63.py',child_returncode=0,source_sha256='hash',calendar_seconds=2.,cap_seconds=50,started_utc='2026-11-03T00:00:00+00:00',absolute_deadline_utc='2026-11-03T00:00:50+00:00')
    run = write('watchdog_run.json', dict(receipt,mode='run'))
    rerun = write('watchdog_replay.json', dict(receipt,mode='replay'))
    p = dict(common, max_calendar_seconds=100, measured_training_projection_seconds=80.)
    for key,path in (('pilot_protocol',pilot),('pilot_summary',summary),('pilot_replay_receipt',replay),('pilot_watchdog_receipt',run),('replay_watchdog_receipt',rerun)):
        p[key]=path
        p[key+'_sha256']=sha(path)
    return p


def test_measured_pilot_and_exact_replay_are_required_instead_of_booleans(tmp_path):
    p=fixture(tmp_path)
    validate_pilot_evidence(p)
    # Neither a successful protocol flag nor a changed checksum excuses a
    # failed external cap, a different model configuration or a missing replay.
    p['pilot_exact_replay_passed']=True
    p['measured_cost_fits']=True
    receipt=Path(p['pilot_watchdog_receipt'])
    record=json.loads(receipt.read_text());record['complete']=False
    receipt.write_text(json.dumps(record));p['pilot_watchdog_receipt_sha256']=sha(receipt)
    with pytest.raises(ValueError,match='watchdogs'):
        validate_pilot_evidence(p)


def test_pilot_configuration_checkpoint_and_measured_cost_cannot_be_substituted(tmp_path):
    p=fixture(tmp_path)
    with pytest.raises(ValueError,match='configuration'):
        validate_pilot_evidence(dict(p,pooled=True))
    with pytest.raises(ValueError,match='cost'):
        validate_pilot_evidence(dict(p,max_calendar_seconds=79))
    with pytest.raises(ValueError,match='Finite'):
        validate_pilot_evidence(dict(p,measured_training_projection_seconds=float('nan')))
    (tmp_path/'step_100.pt').write_bytes(b'changed fixture checkpoint')
    with pytest.raises(ValueError,match='checkpoint'):
        validate_pilot_evidence(p)
