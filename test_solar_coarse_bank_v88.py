"""Pure metadata controls; no array decode, Torch, reserve or child worker."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import solar_coarse_bank_v88 as b


def contract_fixture():
    return dict(eligible_training_unit_indices=list(range(64)),training_groups=['g%02d'%i for i in range(16)],training_unit_components={str(i):'g%02d'%(i//4)for i in range(64)})


def test_actual_qualification_elapsed_projects_4unit72s_without_guessed_floor():
    ids,cap=b.derive_plan(dict(phase='cost_pilot',pilot_unit_count=4),contract_fixture(),8.985954438976478)
    assert ids==[0,21,42,63]and cap==72
    assert b.derive_plan(dict(phase='cost_pilot',pilot_unit_count=1),contract_fixture(),8.985954438976478)==([0],18)


def test_full_cap_only_actual_fresh_pilot_all64eligible_no_relabeling():
    pilot=(dict(phase='cost_pilot'),dict(calendar_seconds=16.2),dict(records=[{}, {}, {}, {}],reused_records=0))
    ids,cap=b.derive_plan(dict(phase='full_bank'),contract_fixture(),99.,pilot)
    assert ids==list(range(64))and cap==519
    # Independently excluded camera remains reported/generated; no new64eligible
    # demand is introduced if original full geometry audit excludes a camera.
    c=contract_fixture();c['eligible_training_unit_indices'].remove(2)
    assert b.derive_plan(dict(phase='full_bank'),c,99.,pilot)[0]==c['eligible_training_unit_indices']


@pytest.mark.parametrize('bad',['missing_pilot','reused_pilot','not_cost_pilot','bad_count','missing_terrain','too_small'])
def test_no_unmeasured_full_cap_or_camera_drop_gate(bad):
    c=contract_fixture();s=dict(phase='full_bank');pilot=(dict(phase='cost_pilot'),dict(calendar_seconds=10.),dict(records=[{}]*4,reused_records=0));elapsed=10.
    if bad=='missing_pilot':pilot=None
    elif bad=='reused_pilot':pilot[2]['reused_records']=1
    elif bad=='not_cost_pilot':pilot[0]['phase']='full_bank'
    elif bad=='bad_count':pilot[2]['records']=[{}]*2
    elif bad=='missing_terrain':c['eligible_training_unit_indices']=list(range(60))
    else:s=dict(phase='cost_pilot',pilot_unit_count=1);elapsed=1.
    with pytest.raises(ValueError):b.derive_plan(s,c,elapsed,pilot)


def make_qualification(tmp_path):
    p=dict(purpose='coarse_teacher_domain_qualification',category='training',max_calendar_seconds=600,expected_frozen_descriptor_sha256='d'*64)
    protocol=tmp_path/'qualification.json';b.write(protocol,p);root=tmp_path/'qualification';root.mkdir();(root/'arrays.npz').write_bytes(b'TOY HASH PAYLOAD ONLY; never decoded')
    started=datetime(2026,11,4,1,tzinfo=timezone.utc)
    w=dict(complete=True,protocol_sha256=b.sha(protocol),calendar_seconds=9.,cap_seconds=600,child_returncode=0,failure=None,ended_utc=(started+timedelta(seconds=9)).isoformat(),hard_deadline_utc=(started+timedelta(seconds=600)).isoformat())
    a=dict(complete=True,passed=True,protocol_sha256=b.sha(protocol),arrays_sha256=b.sha(root/'arrays.npz'),teacher_descriptor_sha256='d'*64,odd_refused_before_read=True,all_original_rows_retained=True,generated_control=dict(passed=True),rows=[dict(exact_teacher_identity=True,exact_conversion=True,exact_native_to_coarse_centres=True,compared_rows=53)]*24)
    b.write(root/'watchdog.json',w);b.write(root/'candidate.json',a)
    b.write(root/'COMPLETE.json',dict(external_watchdog_passed=True,protocol_sha256=b.sha(protocol),candidate_sha256=b.sha(root/'candidate.json'),watchdog_sha256=b.sha(root/'watchdog.json')))
    return protocol,root


def test_completed_qualification_replay_valid_after_old_deadline_no_new_lease(tmp_path):
    p,r=make_qualification(tmp_path)
    # This fixed past receipt remains valid after original launch lease expiry.
    assert b.qualified_teacher(p,r)[1]['calendar_seconds']==9.
    (r/'arrays.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):b.qualified_teacher(p,r)


@pytest.mark.parametrize('target',['watchdog','candidate','complete'])
def test_failed_or_partial_qualification_never_admits_bank(tmp_path,target):
    p,r=make_qualification(tmp_path);path=r/('COMPLETE.json'if target=='complete'else target+'.json')
    v=b.read(path)
    if target=='complete':v['external_watchdog_passed']=False
    elif target=='candidate':v['passed']=False
    else:v['complete']=False
    path.write_text(json.dumps(v))
    with pytest.raises(ValueError):b.qualified_teacher(p,r)


def test_unit_cache_key_shared_by_pilot_full_but_sensitive_to_teacher_source_geometry_and_host():
    p=dict(expected_frozen_descriptor_sha256='d'*64,teacher_sha256={'selected.pt':'e'*64},source_sha256={'adapter':'f'*64},qualification_protocol_sha256='a'*64,qualification_complete_sha256='b'*64,phase='cost_pilot',output='pilot')
    r=dict(unit_index=0,metadata_sha256='c'*64);m=dict(input=dict(geometry_audit_sha256='1'*64,summary_sha256='2'*64,seal_sha256='3'*64));environment=dict(torch='toy',threads=2)
    key,inputs=b.unit_input_key(p,r,m,environment)
    q=deepcopy(p);q.update(phase='full_bank',output='full');assert b.unit_input_key(q,r,m,environment)[0]==key
    for field in('expected_frozen_descriptor_sha256','qualification_complete_sha256'):
        q=deepcopy(p);q[field]='9'*64;assert b.unit_input_key(q,r,m,environment)[0]!=key
    changed=deepcopy(m);changed['input']['summary_sha256']='9'*64;assert b.unit_input_key(p,r,changed,environment)[0]!=key
    assert b.unit_input_key(p,r,m,dict(torch='different',threads=2))[0]!=key


def make_completed_bank(tmp_path):
    root=tmp_path/'bank';root.mkdir();protocol=tmp_path/'bank-protocol.json';start=datetime(2026,11,4,1,tzinfo=timezone.utc)
    p=dict(phase='full_bank',unit_indices=[0,1],training_groups=['g0','g1'],max_calendar_seconds=72,created_utc=start.isoformat(),latest_start_utc=(start+timedelta(seconds=1800)).isoformat(),absolute_deadline_utc=(start+timedelta(seconds=1872)).isoformat())
    b.write(protocol,p);b.write(root/'view_prior.json',dict(toy_training_moments=True))
    manifest=dict(complete=True,output=str(root),protocol_path=str(protocol),phase='full_bank',format=b.FORMAT,array_schema='distributional-coarse128-teacher-v86',split='training',calibration_or_final_access=False,heldout_image_reads=0,all_historical_rows_retained=True,records=[dict(unit_index=0),dict(unit_index=1)],eligible_unit_indices=[0,1],training_groups=['g0','g1'],view_prior=str(root/'view_prior.json'),view_prior_sha256=b.sha(root/'view_prior.json'))
    b.write(root/'manifest.json',manifest);b.write(root/'summary.json',dict(complete=True,protocol_sha256=b.sha(protocol),manifest_sha256=b.sha(root/'manifest.json')))
    b.write(root/'watchdog.json',dict(complete=True,protocol_sha256=b.sha(protocol),calendar_seconds=10.,cap_seconds=72,child_returncode=0,failure=None,started_utc=start.isoformat(),ended_utc=(start+timedelta(seconds=10)).isoformat(),hard_deadline_utc=(start+timedelta(seconds=72)).isoformat()))
    b.write(root/'COMPLETE.json',dict(external_watchdog_passed=True,protocol_sha256=b.sha(protocol),summary_sha256=b.sha(root/'summary.json'),manifest_sha256=b.sha(root/'manifest.json'),watchdog_sha256=b.sha(root/'watchdog.json')))
    return root/'manifest.json'


def test_trainer_shared_completed_bank_gate_checks_original_hashes_and_prior(tmp_path):
    m=make_completed_bank(tmp_path);proof=b.validate_completed_bank(m)
    assert proof['manifest']['phase']=='full_bank'and proof['manifest_sha256']==b.sha(m)
    Path(proof['manifest']['view_prior']).write_text('{}')
    with pytest.raises(ValueError,match='prior'):b.validate_completed_bank(m)


@pytest.mark.parametrize('change',['manifest','watchdog','protocol'])
def test_completed_bank_tampering_cannot_be_promoted(tmp_path,change):
    m=make_completed_bank(tmp_path);root=m.parent
    path=m if change=='manifest'else root/'watchdog.json'if change=='watchdog'else tmp_path/'bank-protocol.json'
    v=b.read(path);v['tampered']=True;path.write_text(json.dumps(v))
    with pytest.raises(ValueError):b.validate_completed_bank(m)


def test_module_and_prepare_have_no_top_level_torch_or_array_decode():
    import ast
    tree=ast.parse(Path(b.__file__).read_text());names=[a.name for n in tree.body if isinstance(n,ast.Import)for a in n.names]
    assert not {'torch','numpy','cv2'}&set(names)
    prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='prepare')
    assert not any(isinstance(n,ast.Attribute)and n.attr in('load','load_checkpoint','Popen','reserve')for n in ast.walk(prepare))


def publication_fixture(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in('original.py',*b.EXTRA_SOURCES):
        p=Path(name);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# toy source\n')
    monkeypatch.setitem(sys.modules,'solar_learning_pipeline_v73',SimpleNamespace(local_source_closure=lambda:{'original.py':b.sha('original.py')}))
    p=dict(phase='cost_pilot',learning_contract='contract.json',original_master='master.json',dataset_summary='dataset.json',geometry_audit='audit.json',lighting_schedule='lighting.json',bank_manifest='bank.json',qualification_protocol='qprotocol.json',qualification_run='qrun')
    names=[p[k]for k in('learning_contract','original_master','dataset_summary','geometry_audit','lighting_schedule','bank_manifest','qualification_protocol')]+['qrun/'+n for n in('COMPLETE.json','watchdog.json','candidate.json')]
    for n in names:path=Path(n);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    p['metadata_sha256']={n:b.sha(n)for n in names};p['source_sha256']={n:b.sha(n)for n in('original.py',*b.EXTRA_SOURCES)}
    Path('protocol.json').write_text(json.dumps(p));return p


def test_expired_source_preflight_never_runs_even_first_git_call(tmp_path,monkeypatch):
    p=publication_fixture(tmp_path,monkeypatch);start=datetime(2026,11,4,tzinfo=timezone.utc);calls=[]
    monkeypatch.setattr(b,'now',lambda:start+timedelta(seconds=3))
    monkeypatch.setattr(b.subprocess,'check_output',lambda *a,**kw:calls.append(a))
    with pytest.raises(TimeoutError,match='original deadline'):b.verify_sources('protocol.json',p,'a'*40,start+timedelta(seconds=2))
    assert not calls


def test_each_publication_timeout_uses_original_remaining_deadline(tmp_path,monkeypatch):
    p=publication_fixture(tmp_path,monkeypatch);clock=[datetime(2026,11,4,tzinfo=timezone.utc)];deadline=clock[0]+timedelta(seconds=2);timeouts=[]
    monkeypatch.setattr(b,'now',lambda:clock[0])
    def public(args,timeout):
        timeouts.append(timeout);assert 0<timeout<=min(10.,(deadline-clock[0]).total_seconds())
        clock[0]+=timedelta(seconds=.6);return Path(args[2].split(':',1)[1]).read_bytes()
    monkeypatch.setattr(b.subprocess,'check_output',public)
    with pytest.raises(TimeoutError,match='original deadline'):b.verify_sources('protocol.json',p,'a'*40,deadline)
    assert len(timeouts)==4
    assert all(a>b0 for a,b0 in zip(timeouts,timeouts[1:]))


@pytest.mark.parametrize('field',['source_sha256','qualification_protocol','qualification_run','bank_manifest','learning_contract','teacher_directory','teacher_sha256','expected_frozen_descriptor_sha256','cache_runs'])
def test_launch_reuses_prepare_pilot_source_teacher_and_exact_cache_guard(tmp_path,field):
    root=tmp_path/'pilot';root.mkdir();(root/'COMPLETE.json').write_text('{}')
    p={k:'frozen-'+k for k in('source_sha256','qualification_protocol','qualification_run','bank_manifest','learning_contract','teacher_directory','teacher_sha256','expected_frozen_descriptor_sha256')}
    p.update(pilot_run=str(root),pilot_protocol='pilot-protocol.json',cache_runs=[dict(protocol='pilot-protocol.json',run=str(root),complete_sha256=b.sha(root/'COMPLETE.json'))])
    pp=deepcopy(p);pilot=(pp,{},{});b.require_matching_pilot(p,pilot)
    if field=='cache_runs':p['cache_runs']=[]
    else:pp[field]='changed'
    with pytest.raises(ValueError):b.require_matching_pilot(p,pilot)
