"""Driver arithmetic/metadata/fake-process controls, no actual Torch or jobs."""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

import qualify_solar_pose_free_v85 as driver


def test_step0_common_feature_scores_and_ties():
    frozen=np.eye(32,dtype=np.float32)[:2]
    query=np.vstack([frozen[0],frozen[0],frozen[1]])
    result,arrays=driver.constant_score_control(frozen,frozen.copy(),query)
    assert result['max_score_difference']==0. and result['tolerance']==2e-6
    assert result['near_top_ties_at_tolerance']==1 and result['argmax_differences']==0
    np.testing.assert_array_equal(arrays['frozen_reference'],arrays['predicted_reference'])


@pytest.mark.parametrize('failure',['different_basis','nan','zero'])
def test_step0_control_cannot_pass_different_basis_or_nonfinite(failure):
    frozen=np.eye(32,dtype=np.float32)[:2];decoded=frozen.copy()
    if failure=='different_basis':decoded=np.roll(decoded,2,axis=1)
    elif failure=='nan':decoded[0,0]=np.nan
    else:decoded[0]=0.
    with pytest.raises(ValueError):driver.constant_score_control(frozen,decoded,frozen)


def test_recall_preserves_all_anchors_and_separates_geometric_coverage():
    reference=np.array([[50.,50.],[400.,400.]])
    target=np.array([[50.,50.],[500.,500.]])
    bank=np.array([[50.,50.],[50.,58.],[58.,50.],[58.,58.],[66.,50.],[50.,66.],[66.,66.],[74.,74.]])
    candidates=np.repeat(bank[None],2,axis=0)
    result=driver.proposal_recall(reference,candidates,target,np.ones(2,bool),bank,np.ones(8,bool))
    assert result['all_retained_reference_anchors']==2 and result['co_visible']==2
    assert result['targets_inside_query_image']==2 and result['geometric_bank_coverage_at16']==1
    assert result['recalled']['16']==1 and result['in_grid_denominator']==1


@pytest.mark.parametrize('correct,total,passed',[(80,100,True),(79,100,False),(100,100,True),(0,0,False)])
def test_known_transform_gate_uses_all_retained_anchors_not_fov_subset(correct,total,passed):
    recall=dict(all_retained_reference_anchors=total,recalled={'16':correct},in_grid_denominator=max(1,correct))
    result=driver.known_transform_gate(recall)
    assert result['passed']is passed and result['denominator']==total
    assert result['minimum_fraction']==.8 and result['threshold_native_px']==16


def test_json_arrays_are_finite_plain_data_and_nonfinite_fails(tmp_path):
    path=tmp_path/'arrays.json';driver.write(path,dict(x=np.array([1.,2.]),count=np.int64(2)))
    assert json.loads(path.read_text())==dict(x=[1.,2.],count=2)
    with pytest.raises(ValueError):driver.write(tmp_path/'nan.json',dict(x=np.nan))


def base_job(tmp_path):
    return dict(purpose='pose_free_pretrained_diagnostic_qualification',diagnostic_only=True,
        calibration_or_final_access=False,cpu_threads=2,max_calendar_seconds=600,cleanup_inside_cap_seconds=5,
        case_ids=[driver.POSITIVE,driver.NEGATIVE],checkpoints=driver.CHECKPOINTS,
        math_sun_budget=292,per_extraction_cap_seconds=180,output=str(tmp_path/'output'))


@pytest.mark.parametrize('key,value',[('max_calendar_seconds',601),('math_sun_budget',293),('cpu_threads',4),('per_extraction_cap_seconds',float('nan')),('diagnostic_only',False)])
def test_scope_failures_reject_before_accounting_or_model_import(tmp_path,key,value):
    job=base_job(tmp_path);job[key]=value;path=tmp_path/'job.json';path.write_text(json.dumps(job))
    with pytest.raises(ValueError,match='fixed diagnostic'):driver.validate_job(str(path),'0'*40)


def fake_controller(tmp_path,monkeypatch,timeout=False):
    protocol=tmp_path/'job.json';protocol.write_text('{}');root=tmp_path/'output'
    current=datetime.now(timezone.utc)
    job=dict(output=str(root),latest_start_utc=(current+timedelta(seconds=10)).isoformat(),absolute_deadline_utc=(current+timedelta(seconds=610)).isoformat())
    monkeypatch.setattr(driver,'validate_job',lambda *args:(job,{}))
    monkeypatch.setattr(driver.signal,'signal',lambda *args:None)
    kills=[];monkeypatch.setattr(driver.os,'killpg',lambda *args:kills.append(args))
    class Child:
        pid=12345
        def __init__(self,argv,**kwargs):
            self.returncode=None if timeout else 0;self.waits=0
            assert argv[-1]=='--worker' and kwargs['start_new_session']
            assert kwargs['env']['OMP_NUM_THREADS']=='2' and kwargs['env']['OPENBLAS_NUM_THREADS']=='1'
            if not timeout:driver.write(root/'summary.json',dict(complete=True,protocol_sha256=driver.sha(protocol)))
        def poll(self):return self.returncode
        def wait(self,timeout=None):
            self.waits+=1
            assert timeout is not None and 0<=timeout<=600
            if self.waits==1 and self.returncode is None:raise subprocess.TimeoutExpired('fake-worker',timeout)
            if self.returncode is None:self.returncode=-15
            return self.returncode
    monkeypatch.setattr(driver.subprocess,'Popen',Child)
    return protocol,root,kills


def test_external_controller_seals_only_complete_timely_child(tmp_path,monkeypatch):
    protocol,root,kills=fake_controller(tmp_path,monkeypatch)
    receipt=driver.controller(str(protocol),'0'*40)
    assert receipt['complete'] and receipt['cap_seconds']==600 and not kills
    assert (root/'COMPLETE.json').exists() and not(root/'FAILED.json').exists()


def test_external_timeout_preserves_failure_and_terminates_process_group(tmp_path,monkeypatch):
    protocol,root,kills=fake_controller(tmp_path,monkeypatch,timeout=True)
    with pytest.raises(SystemExit):driver.controller(str(protocol),'0'*40)
    assert kills and (root/'FAILED.json').exists() and not(root/'COMPLETE.json').exists()
    assert driver.read(root/'FAILED.json')['complete']is False


def test_source_validation_consumes_original_total_600_seconds(tmp_path,monkeypatch):
    protocol,root,kills=fake_controller(tmp_path,monkeypatch)
    job,_=driver.validate_job(str(protocol),'0'*40)
    origin=datetime.now(timezone.utc);clock=dict(seconds=0.)
    monkeypatch.setattr(driver,'now',lambda:origin+timedelta(seconds=clock['seconds']))
    monkeypatch.setattr(driver.time,'monotonic',lambda:100.+clock['seconds'])
    def validation(*args):clock['seconds']+=5.;return job,{}
    monkeypatch.setattr(driver,'validate_job',validation)
    original=driver.subprocess.Popen;waits=[]
    def launch(*args,**kwargs):
        child=original(*args,**kwargs);wait=child.wait
        def timed_wait(timeout=None):waits.append(timeout);return wait(timeout=timeout)
        child.wait=timed_wait;return child
    monkeypatch.setattr(driver.subprocess,'Popen',launch)
    result=driver.controller(str(protocol),'0'*40)
    assert result['complete'] and result['calendar_seconds']==5.
    assert result['started_utc']==origin.isoformat()
    assert driver.utc(result['absolute_deadline_utc'])==origin+timedelta(seconds=600)
    assert driver.utc(result['work_deadline_utc'])==origin+timedelta(seconds=595)
    assert waits==[590.]


def test_preflight_cannot_launch_worker_after_original_work_allowance(tmp_path,monkeypatch):
    protocol,root,kills=fake_controller(tmp_path,monkeypatch)
    job,_=driver.validate_job(str(protocol),'0'*40);origin=datetime.now(timezone.utc);clock=dict(seconds=0.)
    job.update(latest_start_utc=(origin+timedelta(seconds=1000)).isoformat(),absolute_deadline_utc=(origin+timedelta(seconds=1600)).isoformat())
    monkeypatch.setattr(driver,'now',lambda:origin+timedelta(seconds=clock['seconds']))
    monkeypatch.setattr(driver.time,'monotonic',lambda:100.+clock['seconds'])
    def validation(*args):clock['seconds']+=596.;return job,{}
    monkeypatch.setattr(driver,'validate_job',validation)
    with pytest.raises(TimeoutError,match='qualification deadline exhausted'):driver.controller(str(protocol),'0'*40)
    assert not root.exists()
