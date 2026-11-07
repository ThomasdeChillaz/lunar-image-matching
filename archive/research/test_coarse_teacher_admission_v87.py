"""Toy admission/watchdog tests only: NO subprocess, reserve, Torch or data."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

_spec=importlib.util.spec_from_file_location('coarse_teacher_qualification_test_module_v87',Path(__file__).parent/'reports'/'qualify_solar_coarse_teacher_v87.py')
q=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(q)


def settings():
    created=datetime(2026,11,4,12,tzinfo=timezone.utc)
    return dict(prepared_only=False,purpose='coarse_teacher_domain_qualification',category='training',cpu_threads=2,max_calendar_seconds=600,
        driver='reports/qualify_solar_coarse_teacher_v87.py',reservation_journal=q.CANONICAL,unit_index=0,lighting_indices=list(range(0,24,2)),
        generated_control=dict(seed=20261101,native_translation=[8,-8],topk=8,threshold_native_px=16.,minimum_all_anchor_fraction=.8),
        created_utc=created.isoformat(),publication_dispatch_lease_seconds=1800,latest_start_utc=(created+timedelta(seconds=1800)).isoformat(),absolute_deadline_utc=(created+timedelta(seconds=2400)).isoformat(),
        output='runs/toy-qualification',calibration_or_final_access=False,compute_ledger='toy-ledger.json',compute_ledger_sha256='a'*64)


def test_fixed_startup_inclusive_lease_and_original_deadline():
    p=settings();start=q.utc(p['created_utc'])+timedelta(seconds=100)
    assert q.validate_settings(p,start,p['driver'])==start+timedelta(seconds=600)
    with pytest.raises(TimeoutError):q.validate_settings(p,q.utc(p['latest_start_utc'])+timedelta(microseconds=1),p['driver'])
    with pytest.raises(TimeoutError):q.validate_settings(p,q.utc(p['created_utc'])-timedelta(microseconds=1),p['driver'])


@pytest.mark.parametrize('mutation',[
    {'prepared_only':True},{'category':'baseline'},{'cpu_threads':4},{'max_calendar_seconds':601},
    {'reservation_journal':'runs/free-journal.json'},{'unit_index':True},{'lighting_indices':[0,1]},
    {'calibration_or_final_access':True},{'output':'../other-run'},{'output':'/tmp/other-run'},
    {'publication_dispatch_lease_seconds':1801}])
def test_prepared_odd_foreign_budget_and_cap_changes_refused_before_science(mutation):
    p=settings();p.update(mutation)
    with pytest.raises(ValueError):q.validate_settings(p,q.utc(settings()['created_utc']),p['driver'])


def test_no_post_measurement_tolerance_or_deadline_extension():
    p=settings();p['generated_control']['minimum_all_anchor_fraction']=.5
    with pytest.raises(ValueError,match='tolerances'):q.validate_settings(p,q.utc(p['created_utc']),p['driver'])
    p=settings();p['absolute_deadline_utc']=(q.utc(p['absolute_deadline_utc'])+timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError,match='algebra'):q.validate_settings(p,q.utc(p['created_utc']),p['driver'])


def publication_fixture(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in('base.py','driver.py','solar_coarse_training_adapter_v86.py','solar_distributional_response_v86.py'):(tmp_path/name).write_text('# toy source only\n')
    monkeypatch.setitem(sys.modules,'solar_learning_pipeline_v73',SimpleNamespace(local_source_closure=lambda:{'base.py':q.sha('base.py')}))
    p=dict(driver='driver.py',learning_contract='contract.json',original_master='master.json',dataset_summary='dataset.json',geometry_audit='audit.json',bank_manifest='bank.json',lighting_schedule='lighting.json')
    for key in('learning_contract','original_master','dataset_summary','geometry_audit','bank_manifest','lighting_schedule'):(tmp_path/p[key]).write_text(json.dumps(dict(source_sha256={'base.py':q.sha('base.py')})))
    p['source_sha256']={n:q.sha(n)for n in('base.py','driver.py','solar_coarse_training_adapter_v86.py','solar_distributional_response_v86.py')}
    p['metadata_sha256']={p[k]:q.sha(p[k])for k in('learning_contract','original_master','dataset_summary','geometry_audit','bank_manifest','lighting_schedule')}
    path=tmp_path/'job.json';path.write_text(json.dumps(p));calls=[]
    def public(args,**kwargs):
        assert 0<kwargs['timeout']<=10
        assert args[:2]==['git','show'];calls.append(args);return Path(args[2].split(':',1)[1]).read_bytes()
    monkeypatch.setattr(q.subprocess,'check_output',public)
    return p,'job.json',calls


def test_published_transitive_sources_and_metadata_must_match_exactly(tmp_path,monkeypatch):
    p,path,calls=publication_fixture(tmp_path,monkeypatch)
    q.verify_publication(path,p,'a'*40)
    assert len(calls)==len(p['source_sha256'])+len(p['metadata_sha256'])+1
    Path('solar_coarse_training_adapter_v86.py').write_text('# changed\n')
    with pytest.raises(ValueError,match='closure|published'):q.verify_publication(path,p,'a'*40)


def test_unpublished_protocol_or_missing_closure_member_refused(tmp_path,monkeypatch):
    p,path,calls=publication_fixture(tmp_path,monkeypatch)
    del p['source_sha256']['base.py']
    with pytest.raises(ValueError,match='closure'):q.verify_publication(path,p,'a'*40)
    p,path,calls=publication_fixture(tmp_path,monkeypatch)
    original=q.subprocess.check_output
    monkeypatch.setattr(q.subprocess,'check_output',lambda args,**kw:b'not the public protocol'if args[2].endswith(':job.json')else original(args,**kw))
    with pytest.raises(ValueError,match='protocol'):q.verify_publication(path,p,'a'*40)


def mocked_run(tmp_path,monkeypatch,*,startup_late=False,child_error=False,child_hang=False):
    monkeypatch.chdir(tmp_path);p=settings();p['output']='attempt';path=tmp_path/'job.json';path.write_text(json.dumps(p))
    start=datetime.now(timezone.utc);deadline=start+timedelta(seconds=600)
    monkeypatch.setattr(q,'admission',lambda *a:(p,deadline));charges=[];launches=[]
    def reserve(*args):
        charges.append(args)
        return dict(reservation=dict(job_id=args[3],category=args[4],seconds=args[5],refund_on_failure=False))
    monkeypatch.setitem(sys.modules,'solar_compute_budget_v67',SimpleNamespace(reserve=reserve))
    times=iter([0.,596.]if startup_late else [0.,1.,2.,596.,597.,598.]if child_hang else [0.,1.,2.,3.,4.])
    monkeypatch.setattr(q.time,'monotonic',lambda:next(times,599. if startup_late else 4.))
    class FakeChild:
        pid=123;returncode=0
        def __init__(self,command,**kw):
            launches.append((command,kw));assert '--worker'in command and kw['start_new_session']
            assert kw['env']['OMP_NUM_THREADS']=='2'and kw['env']['MKL_NUM_THREADS']=='2'
            if child_error:raise OSError('toy launch failed')
            if child_hang:self.returncode=None
            out=Path('attempt');(out/'arrays.npz').write_bytes(b'toy array artifact, never decoded')
            (out/'candidate.json').write_text(json.dumps(dict(complete=True,passed=True,protocol_sha256=q.sha(path),arrays_sha256=q.sha(out/'arrays.npz'))))
        def poll(self):return self.returncode
        def wait(self,timeout):
            assert timeout is not None and 0<=timeout<=3
            if not getattr(self,'terminated_once',False):
                self.terminated_once=True
                raise q.subprocess.TimeoutExpired('toy',timeout)
            self.returncode=-9;return -9
    monkeypatch.setattr(q.subprocess,'Popen',FakeChild)
    return path,p,charges,launches


def test_controller_cli_exact_one_canonical_charge_no_retry_no_real_worker(tmp_path,monkeypatch):
    path,p,charges,launches=mocked_run(tmp_path,monkeypatch)
    q.run(path,'a'*40,'toy-python-never-executed')
    assert len(charges)==len(launches)==1
    assert charges[0][2:]==(q.CANONICAL,q.sha(path)+q.SUFFIX,'training',600)
    assert launches[0][0][:2]==['toy-python-never-executed',p['driver']]
    assert Path('attempt/COMPLETE.json').exists()
    with pytest.raises(FileExistsError):q.run(path,'a'*40,'unused')
    assert len(charges)==1


def test_expired_preflight_prevents_reservation_and_launch_without_resetting_cap(tmp_path,monkeypatch):
    path,p,charges,launches=mocked_run(tmp_path,monkeypatch,startup_late=True)
    with pytest.raises(TimeoutError,match='preflight'):q.run(path,'a'*40,'never-executed')
    assert not charges and not launches and not Path('attempt').exists()


def test_failure_keeps_charge_outputs_and_cannot_promote(tmp_path,monkeypatch):
    path,p,charges,launches=mocked_run(tmp_path,monkeypatch,child_error=True)
    with pytest.raises(RuntimeError):q.run(path,'a'*40,'never-executed')
    assert len(charges)==1 and not Path('attempt/COMPLETE.json').exists()
    receipt=json.loads(Path('attempt/watchdog.json').read_text())
    assert not receipt['complete']and receipt['compute_reservation']['reservation']['seconds']==600


def test_timed_out_child_process_group_and_cleanup_stay_inside_original_cap(tmp_path,monkeypatch):
    path,p,charges,launches=mocked_run(tmp_path,monkeypatch,child_hang=True);kills=[]
    monkeypatch.setattr(q.os,'killpg',lambda pid,s:kills.append((pid,s)))
    with pytest.raises(RuntimeError):q.run(path,'a'*40,'never-executed')
    assert kills==[(123,q.signal.SIGTERM),(123,q.signal.SIGKILL)]
    receipt=json.loads(Path('attempt/watchdog.json').read_text())
    assert not receipt['complete']and receipt['calendar_seconds']<600
    assert receipt['child_returncode']==-9 and len(charges)==1 and not Path('attempt/COMPLETE.json').exists()


def test_controller_import_and_settings_tests_have_no_torch_numpy_or_data_import():
    import ast
    tree=ast.parse(Path(q.__file__).read_text())
    names=[n.module for n in tree.body if isinstance(n,ast.ImportFrom)]+[a.name for n in tree.body if isinstance(n,ast.Import)for a in n.names]
    assert 'torch'not in names and 'numpy'not in names and 'PIL'not in names
