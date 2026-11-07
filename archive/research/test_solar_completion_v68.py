"""Admission/physical-schedule regression controls; no render or neural job."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import pytest
from solar_completion_v68 import camera_replay,admit,verify_seal,sha
from solar_data_v65 import schedule


def unit():
    s=dict(component='software-test',split='training')
    return dict(site=s,core_max_height=400.,camera_pairs=schedule(s,400.,2)[:1],output='old/software-test/training_motion-0',calibration_or_final_access=False)


def test_tiny_rotation_replay_does_not_change_frozen_matrix_or_physical_draws():
    u=unit();saved=deepcopy(u);u['camera_pairs'][0]['reference']['cam2world'][0][0]+=2e-16
    report=camera_replay(u);assert report['passed']and report['rotation_basis_tolerance']==1e-12
    assert u['camera_pairs'][0]['reference']['cam2world'][0][0]!=saved['camera_pairs'][0]['reference']['cam2world'][0][0]
    # Validator must not silently regenerate/replace the stored render matrix.
    mutated=deepcopy(u);mutated['camera_pairs'][0]['reference']['position'][0]+=1e-12
    with pytest.raises(ValueError):camera_replay(mutated)
    mutated=deepcopy(u);mutated['camera_pairs'][0]['reference']['cam2world'][0][0]+=1e-10
    with pytest.raises(ValueError):camera_replay(mutated)


def contract(tmp_path):
    p=tmp_path/'master.json';p.write_text(json.dumps(dict(units=[unit()for _ in range(96)])))
    now=datetime(2026,11,4,tzinfo=timezone.utc)
    return dict(phase='missing_only_completion',completion_cap_seconds=50400,per_unit_cap_seconds=900,original_deadline_met=False,cpu_threads=2,calibration_or_final_access=False,matcher_forwards_allowed=False,optimizer_updates_allowed=0,completion_absolute_deadline_utc=(now+timedelta(seconds=50400)).isoformat(),completion_contract_created_utc=now.isoformat(),factor_two_max_projection_seconds=47692.6,copy_and_admission_margin_seconds=2700,missing_unit_indices=list(range(10,96)),original_master=str(p),original_master_sha256=sha(p),original_root='old',new_output_root='new'),now


def test_completion_cannot_render_copied_units_or_reset_its_deadline(tmp_path):
    c,now=contract(tmp_path);u,path,deadline=admit(c,10,now+timedelta(hours=13))
    assert str(path)=='new/software-test/training_motion-0'and deadline==now+timedelta(seconds=50400)
    with pytest.raises(ValueError):admit(c,0,now)
    with pytest.raises(TimeoutError):admit(c,10,deadline)
    changed=deepcopy(c);changed['completion_absolute_deadline_utc']=(deadline+timedelta(seconds=1)).isoformat()
    with pytest.raises(TimeoutError):admit(changed,10,now)


def test_completion_is_distinct_data_allowance_and_does_not_refund_old_attempt(tmp_path):
    c,now=contract(tmp_path);c['original_deadline_met']=True
    with pytest.raises(ValueError):admit(c,10,now)
    c,now=contract(tmp_path);c['factor_two_max_projection_seconds']=48000
    with pytest.raises(ValueError):admit(c,10,now)


def test_immutable_copy_seal_rejects_missing_and_extra_payloads(tmp_path):
    folder=tmp_path/'unit';folder.mkdir();p=folder/'summary.json';p.write_text(json.dumps(dict(complete=True,rendered_images=48,images=[{}]*48)))
    (folder/'COMPLETE.json').write_text(json.dumps(dict(sha256={'summary.json':sha(p)})))
    assert verify_seal(folder)['complete']
    (folder/'extra.png').write_bytes(b'not-part-of-seal')
    with pytest.raises(ValueError):verify_seal(folder)


def test_zero_exit_without_complete_receipt_is_not_success(tmp_path):
    from solar_completion_v68 import require_renderer_summary
    with pytest.raises(RuntimeError):require_renderer_summary(tmp_path,'contract')
    (tmp_path/'summary.json').write_text(json.dumps(dict(complete=False,rendered_images=0,images=[],completion_contract_sha256='contract')))
    with pytest.raises(ValueError):require_renderer_summary(tmp_path,'contract')
