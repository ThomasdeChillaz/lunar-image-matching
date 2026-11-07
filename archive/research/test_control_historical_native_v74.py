"""Stub and arithmetic tests only: no images, neural models or downloads."""
from datetime import datetime,timezone,timedelta
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
REPO=HERE if(HERE/'historical_compact_adapters_v73.py').exists()else HERE.parents[1]/'outputs'/'lunar-image-matching'
sys.path.insert(0,str(HERE));sys.path.append(str(REPO))
import control_historical_native_v74 as control
import historical_compact_adapters_v73 as historical


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return str(path)


def base_profile(method='historical_original_native512'):
    configuration=dict(model_directory='model',checkpoint='selected',input_profile='raw_native512')
    return dict(name=method,backend='historical_compact',method=method,configuration=configuration,
        configuration_sha256=historical.value_digest(configuration),artifact_sha256={},source_sha256={},cpu_threads=2,native_size=512,calibration_or_final_access=False)


def test_stub_suite_retains_raw_arrays_but_cannot_create_actual_qualification():
    monitor=SimpleNamespace(counts=dict(descriptor=0),origin='stub_test_only');order=[]
    p=np.array([[80.,80.],[100.,100.],[150.,100.],[150.,200.],[250.,250.],[350.,300.]])
    pose=np.eye(4);pose[:3,3]=[-.2,.1,0.]
    class StubAdapter:
        def infer(self,inp):
            order.append(inp)
            if inp=='invalid_support':raise ValueError('Common valid native reference geometry required')
            if inp=='blank':return dict(reference_xy=[],query_xy=[],pose=None)
            monitor.counts['descriptor']+=1
            return dict(reference_xy=p,query_xy=p+[8.,-4.],pose=pose)
    output=control.execute_suite(StubAdapter(),monitor,{n:n for n in('cold_known_transform','known_transform','blank','invalid_support')},base_profile(),lambda a,b:(pose,dict(inliers=6)))
    assert order==['cold_known_transform','known_transform','blank','invalid_support']
    assert output['known_transform']['reference_to_query_affine']==[[1.,0.,8.],[0.,1.,-4.]]
    assert output['known_transform']['query_xy']==(p+[8.,-4.]).tolist()
    assert output['actual_pretrained_forward']is False and output['replay']['passed']is False
    assert output['blank']['reference_xy']==[]and output['invalid_support']['pose_returned']is False
    assert output['invalid_support']['rejection']['type']=='ValueError'
    assert output['neural_forward_counts']['invalid_support']['descriptor']==0
    # These are unsealed stub return values, never native_control evidence.
    assert output['measurement_origin']=='stub_test_only'


def test_analytic_pose_truth_is_fixed_and_detects_wrong_or_nonrigid_pose():
    truth=np.eye(4);truth[:3,3]=[-.2,.1,0.]
    actual=control.pose_arithmetic(truth)
    assert actual['rotation_error_deg']==0. and actual['translation_error_m']==0.
    wrong=truth.copy();wrong[0,3]+=.1
    assert control.pose_arithmetic(wrong)['translation_error_m']>.05
    assert control.pose_arithmetic(None)['pose_returned']is False
    with pytest.raises(ValueError,match='rigid transform'):control.pose_arithmetic(np.ones((4,4)))


def test_historical_validator_exact_evidence_schema_is_preserved():
    method_names=set(historical.METHODS)
    assert method_names=={'historical_original_native512','historical_frozen_surface','historical_north_context'}
    # Control bootstrap never calls the circular qualified-profile constructor.
    source=(HERE/'control_historical_native_v74.py').read_text()
    assert 'HistoricalCompactAdapter.__new__'in source and 'h._HistoricalRuntime(profile)'in source
    assert 'validate_historical_profile(profile);write_new' in source


def test_actual_inputs_are_not_used_during_preparation_or_stub_suite(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    settings=dict(prepared_only=False,max_calendar_seconds=10,publication_dispatch_lease_seconds=1800,
        original_master='master.json',dataset_summary='dataset.json',geometry_audit='audit.json',compute_ledger='ledger.json')
    for path in('master.json','dataset.json','audit.json','ledger.json'):write(path,{})
    path=write('settings.json',settings);seen=[]
    def blocked_gate(p):seen.append('full96_gate');raise ValueError('actual full96 gate is absent')
    monkeypatch.setattr(control,'full_gate',blocked_gate)
    monkeypatch.setattr(control,'actual_inputs',lambda *args:pytest.fail('Preparation generated an image'))
    monkeypatch.setattr(control,'bootstrap_profile',lambda *args:pytest.fail('Model lineage reached before full-data gate'))
    with pytest.raises(ValueError,match='full96 gate'):control.prepare(path,'protocol.json')
    assert seen==['full96_gate']and not Path('protocol.json').exists()


def budget_protocol(tmp_path):
    ledger=write('ledger.json',dict(accounting_complete=True,remaining_baseline_wall_seconds=67200,remaining_training_wall_seconds=20220))
    initial=[dict(job_id='existing/candidate',category='baseline',seconds=300,refund_on_failure=False)]
    write(control.CANONICAL_JOURNAL,dict(ledger_sha256=control.sha(ledger),reservations=initial))
    now=datetime.now(timezone.utc)
    return dict(method='historical_original_native512',control_kind='historical_compact',output='run',max_calendar_seconds=1,latest_start_utc=(now+timedelta(seconds=10)).isoformat(),
        absolute_deadline_utc=(now+timedelta(seconds=20)).isoformat(),compute_ledger=ledger,compute_ledger_sha256=control.sha(ledger),reservation_journal=control.CANONICAL_JOURNAL,initial_reservations=initial)


def test_canonical_baseline_journal_and_existing_charges_cannot_reset(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path);p=budget_protocol(tmp_path)
    assert control.balance(p)[0]==66900
    p['reservation_journal']='runs/other.json'
    with pytest.raises(ValueError,match='canonical baseline journal'):control.balance(p)
    p['reservation_journal']=control.CANONICAL_JOURNAL
    j=control.read(control.CANONICAL_JOURNAL);j['reservations']=[];write(control.CANONICAL_JOURNAL,j)
    with pytest.raises(ValueError,match='removed or rewritten'):control.balance(p)


@pytest.mark.parametrize('late_exit',[False,True])
def test_external_watchdog_rejects_running_and_terminal_overrun(monkeypatch,tmp_path,late_exit):
    monkeypatch.chdir(tmp_path);p=budget_protocol(tmp_path);protocol=write('protocol.json',p)
    # The controller hashes its published source in the receipt; source-only copy.
    (tmp_path/control.SELF).write_bytes((HERE/control.SELF).read_bytes())
    monkeypatch.setattr(control,'admission',lambda *args:p)
    killed=[]
    class Child:
        pid=99999999;returncode=None;args=['procedural-worker']
        def __init__(self,*args,**kwargs):self.returncode=0 if late_exit else None
        def poll(self):return self.returncode
        def wait(self,timeout=None):self.returncode=-15
    ticks=iter([0.,2.,2.,2.]if late_exit else[0.,0.,2.,2.,2.,2.])
    monkeypatch.setattr(control,'time',SimpleNamespace(monotonic=lambda:next(ticks,2.),sleep=lambda seconds:None))
    monkeypatch.setattr(control.subprocess,'Popen',Child)
    monkeypatch.setattr(control.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    with pytest.raises(TimeoutError):control.run(protocol,'f'*40,sys.executable)
    receipt=control.read('run/receipt.json');assert receipt['complete']is False and receipt['passed']is False
    charges=control.read(control.CANONICAL_JOURNAL)['reservations'];assert len(charges)==2 and charges[-1]['category']=='baseline'and charges[-1]['seconds']==1
    assert(bool(killed))is(not late_exit)


def test_expired_latest_start_refuses_before_charge_or_process(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path);p=budget_protocol(tmp_path);p['latest_start_utc']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat();protocol=write('protocol.json',p)
    monkeypatch.setattr(control,'admission',lambda *args:p)
    monkeypatch.setattr(control.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('Expired lease started a control'))
    with pytest.raises(TimeoutError,match='dispatch lease expired'):control.run(protocol,'f'*40,sys.executable)
    assert len(control.read(control.CANONICAL_JOURNAL)['reservations'])==1


def test_existing_matchanything_api_supports_actual_two_thread_reuse():
    import control_native_matchanything_v17 as legacy
    assert tuple(inspect.signature(legacy.run).parameters)==('output','dependency_dir','profile','threads','max_seconds')
    source=Path(legacy.__file__).read_text()
    assert "choices=[1,2],default=2"in source and "choices=['native512','training_size832']"in source
    # No model/control function is invoked by this inspection.


def test_failed_outer_receipt_cannot_be_replaced_by_successful_looking_inner_metadata(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path);p=budget_protocol(tmp_path)
    (tmp_path/control.SELF).write_bytes((HERE/control.SELF).read_bytes());p['source_sha256']={control.SELF:control.sha(control.SELF)}
    protocol=write('protocol.json',p)
    receipt=dict(complete=False,passed=False,failure={'message':'timeout'},method=p['method'],protocol_sha256=control.sha(protocol),source_sha256=p['source_sha256'][control.SELF],cpu_threads=2,child_returncode=0)
    path=write('failed-receipt.json',receipt)
    with pytest.raises(ValueError,match='outer control receipt'):control.validate_receipt(protocol,path)


def test_even_complete_flags_cannot_hide_outer_deadline_overrun(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path);p=budget_protocol(tmp_path)
    (tmp_path/control.SELF).write_bytes((HERE/control.SELF).read_bytes());p['source_sha256']={control.SELF:control.sha(control.SELF)}
    protocol=write('protocol.json',p);start=datetime.now(timezone.utc)
    forged_unit_test=dict(complete=True,passed=True,failure=None,method=p['method'],protocol_sha256=control.sha(protocol),source_sha256=p['source_sha256'][control.SELF],cpu_threads=2,child_returncode=0,
        cap_seconds=1,calendar_seconds=2.,started_utc=start.isoformat(),absolute_deadline_utc=(start+timedelta(seconds=1)).isoformat())
    path=write('FORGED_UNIT_TEST_ONLY.json',forged_unit_test)
    with pytest.raises(ValueError,match='deadline cap'):control.validate_receipt(protocol,path)


def test_completed_qualification_is_read_only_after_dispatch_lease_and_budget_exhaustion(monkeypatch,tmp_path):
    """Synthetic metadata only; stop before qualification, never a gate receipt."""
    monkeypatch.chdir(tmp_path);p=budget_protocol(tmp_path)
    (tmp_path/control.SELF).write_bytes((HERE/control.SELF).read_bytes())
    latest=datetime.now(timezone.utc)-timedelta(hours=1);created=latest-timedelta(seconds=1800);start=latest-timedelta(seconds=.5)
    p.update(schema_version=74,cpu_threads=2,calibration_or_final_access=False,query_truth_used=False,
        publication_dispatch_lease_seconds=1800,created_utc=created.isoformat(),latest_start_utc=latest.isoformat(),absolute_deadline_utc=(latest+timedelta(seconds=1)).isoformat(),
        source_sha256={control.SELF:control.sha(control.SELF)},artifact_sha256={},base_profile=base_profile(),configuration=base_profile()['configuration'],
        configuration_sha256=base_profile()['configuration_sha256'],native_shape=[512,512],image_seed=20261101,reference_to_query_affine=[[1.,0.,8.],[0.,1.,-4.]],reference_depth_m=10.,
        intrinsics=[[400.,0.,255.5],[0.,400.,255.5],[0.,0.,1.]],expected_query_cam2world=[[1.,0.,0.,-.2],[0.,1.,0.,.1],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    protocol=write('protocol.json',p);write('run/outputs.json',{'TEST_ONLY':'not measurements'})
    journal=control.read(control.CANONICAL_JOURNAL);journal['reservations'] += [dict(job_id=control.sha(protocol)+'/native_control',category='baseline',seconds=1,refund_on_failure=False),dict(job_id='another-real-job-placeholder',category='baseline',seconds=66899,refund_on_failure=False)]
    write(control.CANONICAL_JOURNAL,journal);before=Path(control.CANONICAL_JOURNAL).read_bytes()
    write('run/receipt.json',dict(complete=True,passed=True,failure=None,method=p['method'],protocol_sha256=control.sha(protocol),outputs_sha256=control.sha('run/outputs.json'),source_sha256=p['source_sha256'][control.SELF],cpu_threads=2,child_returncode=0,
        cap_seconds=1,calendar_seconds=.1,started_utc=start.isoformat(),absolute_deadline_utc=(start+timedelta(seconds=1)).isoformat()))
    monkeypatch.setattr(control.subprocess,'check_output',lambda args:Path(args[-1].split(':',1)[1]).read_bytes())
    monkeypatch.setattr(control,'full_gate',lambda p:None);monkeypatch.setattr(control,'bootstrap_profile',lambda p:None)
    import solar_compute_budget_v67
    monkeypatch.setattr(solar_compute_budget_v67,'reserve',lambda *args:pytest.fail('Completed qualification tried to charge or dispatch'))
    def stop_before_qualification(*args):raise RuntimeError('TEST_ONLY: reached independent replay after completed admission')
    monkeypatch.setattr(control,'replay_outputs',stop_before_qualification)
    assert control.balance(p)[0]==0
    with pytest.raises(RuntimeError,match='reached independent replay'):control.qualify(protocol,'f'*40,'MUST_NOT_EXIST.json')
    assert Path(control.CANONICAL_JOURNAL).read_bytes()==before and not Path('MUST_NOT_EXIST.json').exists()


def test_optional_external_control_cannot_bypass_failed_outer_receipt(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    protocol=write('outer.json',dict(control_kind='legacy_matchanything',method='matchanything',child_output='native',source_sha256={'source.py':'a'*64}))
    receipt=write('outer-receipt.json',dict(TEST_ONLY='failure metadata'))
    profile=dict(method='matchanything',author_profile='native512',cpu_threads=2,external_control=dict(protocol=protocol,protocol_sha256=control.sha(protocol),receipt=receipt,receipt_sha256=control.sha(receipt),publication_commit='f'*40))
    monkeypatch.setattr(control,'admission',lambda *args:control.read(protocol))
    def failed_receipt(*args):raise ValueError('late outer receipt')
    monkeypatch.setattr(control,'validate_receipt',failed_receipt)
    with pytest.raises(ValueError,match='late outer receipt'):control.validate_external_pretrained(profile)


def test_optional_external_control_requires_identical_inner_files_and_exposes_comparison_pins(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path);inner={}
    for name in('protocol','summary','controls','provenance'):
        path=write('native/'+name+'.json',dict(TEST_ONLY=name));inner.update({name:path,name+'_sha256':control.sha(path)})
    protocol=write('outer.json',dict(control_kind='legacy_matchanything',method='matchanything',child_output='native',source_sha256={'source.py':'a'*64}))
    receipt=write('outer-receipt.json',dict(TEST_ONLY='not measurements'))
    evidence=dict(protocol=protocol,protocol_sha256=control.sha(protocol),receipt=receipt,receipt_sha256=control.sha(receipt),publication_commit='f'*40)
    profile=dict(method='matchanything',author_profile='native512',cpu_threads=2,external_control=evidence,pretrained_evidence=inner)
    monkeypatch.setattr(control,'admission',lambda *args:control.read(protocol));monkeypatch.setattr(control,'validate_receipt',lambda *args:dict(TEST_ONLY=True))
    gate=control.validate_external_pretrained(profile)
    assert gate['input_sha256']=={protocol:control.sha(protocol),receipt:control.sha(receipt)} and gate['source_sha256']=={'source.py':'a'*64}
    wrong=write('other-summary.json',dict(TEST_ONLY='summary'));inner['summary']=wrong;inner['summary_sha256']=control.sha(wrong)
    with pytest.raises(ValueError,match='identical four inner'):control.validate_external_pretrained(profile)
    assert control.validate_external_pretrained(dict(method='moonst3r'))is None
