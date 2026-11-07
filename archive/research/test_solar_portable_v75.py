"""Procedural bytes/metadata only. No lunar arrays/images, NN, rays or child jobs."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import base64,json,os,zipfile
import pytest
import solar_portable_evidence_v75 as e
import solar_capsule_v75 as c


def put(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,allow_nan=False)if isinstance(value,(dict,list))else value);return str(p)


def proof(root,paths):
    # Public Git tree/commit objects generated from procedural files, no Git command.
    def tree(folder):
        raw=b'';objects={}
        for item in sorted(folder.iterdir(),key=lambda p:p.name+('/'if p.is_dir()else'')):
            if item.is_dir():oid,desc=tree(item);objects.update(desc);mode=b'40000'
            else:oid=e.git_oid('blob',item.read_bytes());mode=b'100644'
            raw+=mode+b' '+item.name.encode()+b'\0'+bytes.fromhex(oid)
        oid=e.git_oid('tree',raw);objects[oid]=dict(kind='tree',data_base64=base64.b64encode(raw).decode());return oid,objects
    oid,objects=tree(root);raw=f'tree {oid}\nauthor Procedural <nobody@example.invalid> 0 +0000\ncommitter Procedural <nobody@example.invalid> 0 +0000\n\nfixture\n'.encode();commit=e.git_oid('commit',raw)
    objects[commit]=dict(kind='commit',data_base64=base64.b64encode(raw).decode())
    return dict(objects=objects,publications=[dict(commit=commit,paths={p:e.sha(root/p)for p in paths})])


@pytest.mark.parametrize('bad',['/tmp/absolute','../escape','a/../b','a//b','a\\b','~/.ssh/id_rsa','C:\\secret','.', ''])
def test_normalized_relative_paths_only(bad):
    with pytest.raises(ValueError):e.safe(bad)


def test_publication_tree_proof_without_git_runtime(tmp_path):
    root=tmp_path/'repo';put(root/'a.py','public procedural source');put(root/'reports/p.json',dict(example=True))
    p=proof(root,['a.py','reports/p.json']);assert e.verify_publication_proof(root,p)
    put(root/'a.py','changed')
    with pytest.raises(ValueError,match='hash differs'):e.verify_publication_proof(root,p)


def test_corrupted_public_object_cannot_claim_source(tmp_path):
    root=tmp_path/'repo';put(root/'a.py','public procedural source');p=proof(root,['a.py']);key=next(iter(p['objects']))
    p['objects'][key]['data_base64']=base64.b64encode(b'forged').decode()
    with pytest.raises(ValueError,match='object changed'):e.verify_publication_proof(root,p)


def test_historical_absolute_identity_allowed_only_in_opaque_completed_contract(tmp_path):
    put(tmp_path/'reports/old.json',dict(blender_executable='/missing/Mac/Blender',python_executable='/missing/Mac/python',canonical_journal_absolute='/missing/Mac/journal'))
    put(tmp_path/'reports/new.json',dict(blender_executable='/missing/Mac/Blender'))
    findings=c.portability_findings(tmp_path,['reports/old.json','reports/new.json'],historical_contracts={'reports/old.json'})
    assert len(findings['opaque_completed_provenance'])==3 and len(findings['failures'])==1
    assert findings['failures'][0]['file']=='reports/new.json'


@pytest.mark.parametrize('path',['.env','.git/config','reports/credentials.json','runs/final/data.npz','data/reserve/pair.png','reports/calibration/results.json'])
def test_private_and_reserve_artifacts_not_in_capsule(path):
    with pytest.raises(ValueError):c.allowed(path,'input')


def test_workflow_source_is_explicit_but_git_internals_are_not():
    assert c.allowed('.github/workflows/solar.yml','source')
    with pytest.raises(ValueError):c.allowed('.git/config','source')


def test_symlink_escape_is_rejected(tmp_path):
    put(tmp_path/'outside','secret');(tmp_path/'root').mkdir();(tmp_path/'root/link').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError,match='Symlink|escaped'):e.file(tmp_path/'root','link')


@pytest.fixture
def capsule(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);put('a.py','# procedural source only\n');put('data/selected.bin','selected procedural bytes')
    put('reports/portable.json',dict(original_master='reports/master.json',dataset_summary='runs/data.json',geometry_audit='reports/audit.json'))
    for name in c.OWN_SOURCES:put(name,Path(c.__file__).with_name(name).read_text())
    p=dict(schema_version=70,builder_schema_version=74,prepared_only=False,split='model_selection',calibration_or_final_access=False,cpu_threads=2,tracks=['estimated_sun'],blender_executable=None,profiles=[dict(name='SIFT',backend='sift')],case_ids=[],source_sha256={name:e.sha(name)for name in({'a.py'}|c.OWN_SOURCES)},input_sha256={'data/selected.bin':e.sha('data/selected.bin'),'reports/portable.json':e.sha('reports/portable.json')})
    put('reports/p.json',p)
    auth=dict(schema_version=75,protocol_sha256=e.sha('reports/p.json'),source_publication_commit='a'*40,nonce='one-use-procedural',canonical_journal_bytes_base64=base64.b64encode(b'unchanged exact post-charge journal UTC bytes').decode())
    put('reports/auth.json',auth);authorization=dict(path='reports/auth.json',sha256=e.sha('reports/auth.json'))
    import solar_learning_pipeline_v73
    monkeypatch.setattr(solar_learning_pipeline_v73,'full_geometry_gate',lambda *a:None)
    monkeypatch.setattr(c,'validate_frozen_assets',lambda *a:(p,{},dict(opaque_completed_provenance=[])))
    monkeypatch.setattr(c,'required_profile_inputs',lambda *a:{})
    import subprocess
    monkeypatch.setattr(subprocess,'check_output',lambda args,**kwargs:Path(args[-1].split(':',1)[1]).read_bytes())
    return p,authorization


def test_full96_gate_precedes_packing_outputs(capsule,monkeypatch):
    import solar_learning_pipeline_v73
    monkeypatch.setattr(solar_learning_pipeline_v73,'full_geometry_gate',lambda *a:(_ for _ in()).throw(ValueError('full96 pending')))
    with pytest.raises(ValueError,match='full96'):c.pack_capsule('reports/p.json','capsule','a'*40,capsule[1],'reports/portable.json')
    assert not Path('capsule').exists()


def test_byte_preserving_pack_install_zip_roundtrip_without_reserving(capsule,monkeypatch):
    import solar_compute_budget_v67
    monkeypatch.setattr(solar_compute_budget_v67,'reserve',lambda *a:(_ for _ in()).throw(AssertionError('No remote or capsule reserve authority')))
    before_p=Path('reports/p.json').read_bytes();before_auth=Path('reports/auth.json').read_bytes()
    m=c.pack_capsule('reports/p.json','capsule','a'*40,capsule[1],'reports/portable.json')
    assert m['protocol_sha256']==e.sha('reports/p.json')and set(m['files'])==set(capsule[0]['input_sha256'])|set(capsule[0]['source_sha256'])|{'reports/p.json','reports/auth.json'}
    result=c.install_capsule('capsule','installed');assert result['complete']and not result['scientific_execution']
    assert Path('installed/reports/p.json').read_bytes()==before_p and Path('installed/reports/auth.json').read_bytes()==before_auth
    assert c.install_capsule('capsule','installed')['installed_files']==0
    c.make_zip('capsule','capsule.zip');c.unpack_zip('capsule.zip','unzipped')
    assert e.sha('unzipped/manifest.json')==e.sha('capsule/manifest.json')


def test_oversize_records_shortfall_before_payload_copy(capsule):
    result=c.pack_capsule('reports/p.json','oversize','a'*40,capsule[1],'reports/portable.json',max_total_bytes=1)
    assert result['status']=='SHORTFALL'and result['files_copied']==0 and not Path('oversize/payload').exists()
    assert Path('oversize/SHORTFALL.json').exists()


def test_no_overwrite_or_unlisted_payloads(capsule):
    c.pack_capsule('reports/p.json','capsule','a'*40,capsule[1],'reports/portable.json');put('installed/data/selected.bin','existing different bytes')
    with pytest.raises(FileExistsError):c.install_capsule('capsule','installed')
    assert not Path('installed/a.py').exists()
    put('capsule/payload/reports/unlisted.json',{})
    with pytest.raises(ValueError,match='extra payloads'):c.validate_capsule('capsule')


def test_archive_traversal_and_duplicate_members_rejected(tmp_path):
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w')as z:z.writestr('../escape',b'x')
    with pytest.raises(ValueError):c.unpack_zip(bad,tmp_path/'out')
    assert not(tmp_path/'out').exists()


def test_live_mac_executable_is_nonportable_before_any_model_or_physics(tmp_path):
    put(tmp_path/'p.json',dict(prepared_only=False,split='model_selection',calibration_or_final_access=False,cpu_threads=2,tracks=['estimated_sun'],blender_executable='/Mac/Blender'))
    with pytest.raises(c.NonportableInput):c.validate_frozen_assets(tmp_path,'p.json','missing-evidence.json')


def test_portable_consumer_cannot_use_empty_or_prepared_evidence(tmp_path):
    put(tmp_path/'e.json',dict(schema_version=75,kind='completed_reference_physics',complete=False,read_only=True,query_truth_used=False,reference_precompute='full',pinned_inputs={}))
    with pytest.raises(ValueError,match='Completed full'):e.portable_completed_reference_preparation(tmp_path,'e.json')


def portable_fixture(root):
    """Synthetic COMPLETED records; geometry/publication helpers are separately stubbed/tested."""
    source={'verify_solar_development_v68.py':'# procedural audit source','solar_reference_watchdog_v74.py':'# procedural watchdog'}
    for path,value in source.items():put(root/path,value)
    sources={path:e.sha(root/path)for path in source}
    put(root/'reports/audit.json',{});put(root/'data/reference.bin','procedural raw reference');put(root/'data/scene.bin','procedural full scene')
    raw=e.sha(root/'data/reference.bin');mesh=e.sha(root/'data/scene.bin');audit_sha=e.sha(root/'reports/audit.json')
    eligible=[64,68,72,76];audit=dict(units=[dict(unit_index=i,component=f'g{(i-64)//4}',split='model_selection',eligible=i in eligible,qualification_reasons=[]if i in eligible else['independent geometry exclusion'],oracle_outcomes=[dict(correct=True)]*2 if i in eligible else[])for i in range(64,96)])
    inventory=e.selection_inventory(audit)
    ledger=dict(accounting_complete=True,remaining_training_wall_seconds=20220,remaining_baseline_wall_seconds=66900)
    put(root/'portable/ledger.json',ledger);ledger_sha=e.sha(root/'portable/ledger.json')
    initial=[dict(job_id='historical-failed',category='baseline',seconds=300,refund_on_failure=False,reserved_utc='2019-12-31T00:00:00+00:00')]
    contract=dict(reference_precompute='full',query_truth_in_primary_inference=False,calibration_or_final_access=False,python_executable='/missing/original/Mac/python',python_executable_sha256='a'*64,blender_executable='/missing/original/Mac/Blender',blender_executable_sha256='b'*64,canonical_journal_absolute='/missing/original/Mac/runs/solar-compute-reservations-v67.json',reservation_journal=e.CANONICAL_JOURNAL,compute_ledger_sha256=ledger_sha,initial_reservations=initial,selection_units=eligible,protocol_directory='reports/physical',source_sha256=sources,preparation_absolute_deadline_utc='2020-01-02T00:00:00+00:00',per_reference_cap_seconds=900,reference_units={str(i):dict(latitude_deg=85.,approved=dict(reference_geometry_sha256=raw,scene_mesh_sha256=mesh,scene_maximum_height_m=120.))for i in eligible},**inventory)
    put(root/'reports/c.json',contract);contract_sha=e.sha(root/'reports/c.json')
    for i in eligible:
        pre=dict(precompute='full',query_truth_used=False,output=f'data/u{i}.bin',reference_geometry='data/reference.bin',reference_geometry_sha256=raw,mesh='data/scene.bin',mesh_sha256=mesh,scene_mesh='data/scene.bin',scene_mesh_sha256=mesh,geometry_audit='reports/audit.json',geometry_audit_sha256=audit_sha,source_sha256=sources)
        put(root/f'reports/physical/references/u{i:03d}-precompute.json',pre);put(root/f'reports/physical/references/u{i:03d}-lookup.json',dict(query_truth_used=False))
    phases={};caps=dict(reference_one=100,reference_four=200,reference_full=50);times={};charges=list(initial)
    for j,stage in enumerate(e.PHASES):
        start=datetime(2020,1,1,j,tzinfo=timezone.utc);pp=f'reports/physical/{stage}.json';out=f'runs/{stage}'
        p=dict(stage=stage,contract_sha256=contract_sha,source_sha256=sources,output=out,max_calendar_seconds=caps[stage],created_utc=start.isoformat(),latest_start_utc=(start+timedelta(seconds=30)).isoformat(),absolute_deadline_utc=(start+timedelta(seconds=30+caps[stage])).isoformat())
        put(root/pp,p);phases[stage]=(pp,p,start);charges.append(dict(job_id=e.sha(root/pp)+'/reference_physics',category='baseline',seconds=caps[stage],refund_on_failure=False,reserved_utc=start.isoformat()));times[stage]=1.
    rows=[]
    for i in eligible:
        stage='reference_one'if i==64 else'reference_four';pp,p,start=phases[stage];prepath=f'reports/physical/references/u{i:03d}-precompute.json';archive=f'data/u{i}.bin';rp=f'data/u{i}.receipt.json';wp=f'data/u{i}.watchdog.json'
        put(root/archive,'procedural derived physics bytes')
        receipt=dict(complete=True,precompute='full',query_truth_used=False,neural_forwards=0,output_sha256=e.sha(root/archive),protocol_sha256=e.sha(root/prepath),source_sha256=sources,calendar_seconds=.25,original_scene_mesh_sha256=mesh,original_scene_ceiling_m=120.,map_mesh_sha256=mesh)
        put(root/rp,receipt)
        watch=dict(complete=True,child_returncode=0,failure=None,program='prepare_solar_reference_v63.py',child_protocol_sha256=e.sha(root/prepath),watchdog_source_sha256=sources['solar_reference_watchdog_v74.py'],phase_protocol=pp,phase_protocol_sha256=e.sha(root/pp),publication_commit='d'*40,work_cap_seconds=900,calendar_seconds=.5,started_utc=(start+timedelta(seconds=.1)).isoformat(),ended_utc=(start+timedelta(seconds=.6)).isoformat(),work_deadline_utc=(start+timedelta(seconds=5)).isoformat())
        put(root/wp,watch)
        rows.append(dict(unit_index=i,archive=archive,archive_sha256=e.sha(root/archive),program_receipt=rp,program_receipt_sha256=e.sha(root/rp),watchdog_receipt=wp,watchdog_receipt_sha256=e.sha(root/wp),outer_calendar_seconds=.5,internal_calendar_seconds=.25))
    for stage,(pp,p,start)in phases.items():
        selected=rows[:1]if stage=='reference_one'else rows
        summary=dict(complete=True,protocol_sha256=e.sha(root/pp),source_sha256=sources,started_utc=start.isoformat(),ended_utc=(start+timedelta(seconds=1)).isoformat(),calendar_seconds=1.,publication_commit='d'*40,reference_rows=selected)
        put(root/(p['output']+'/summary.json'),summary);put(root/(p['output']+'/COMPLETE.json'),dict(complete=True,summary_sha256=e.sha(root/(p['output']+'/summary.json')),protocol_sha256=e.sha(root/pp)))
    put(root/'portable/journal.json',dict(ledger_sha256=ledger_sha,reservations=charges))
    inputs={str(p.relative_to(root)):e.sha(p)for p in root.rglob('*')if p.is_file()}
    result=dict(pinned_inputs={},reference_archives=sorted(r['archive']for r in rows),reference_unit_indices=eligible,baseline_reserved_seconds=sum(caps.values()),phase_calendar_seconds=times,outer_calendar_seconds_by_archive={r['archive']:.5 for r in rows},internal_calendar_seconds_by_archive={r['archive']:.25 for r in rows},scope='procedural completed fixture only',**inventory)
    payload=dict(schema_version=75,kind='completed_reference_physics',complete=True,read_only=True,query_truth_used=False,reference_precompute='full',protocol=phases['reference_full'][0],summary='runs/reference_full/summary.json',contract='reports/c.json',original_master='reports/master.json',dataset_summary='reports/data.json',geometry_audit='reports/audit.json',ledger_snapshot='portable/ledger.json',journal_snapshot='portable/journal.json',original_canonical_journal=e.CANONICAL_JOURNAL,original_canonical_absolute_identity=contract['canonical_journal_absolute'],historical_executable_identities={n:dict(path=contract[n],sha256=contract[n+'_sha256'])for n in('python_executable','blender_executable')},pinned_inputs=inputs,publication_proof=dict(objects={},publications=[dict(commit='d'*40,paths=inputs)]),result=result)
    put(root/'portable/evidence.json',payload)
    return payload,audit


def test_completed_replay_without_mac_files_live_journal_git_or_fresh_deadline(tmp_path,monkeypatch):
    payload,audit=portable_fixture(tmp_path)
    monkeypatch.setattr(e,'verify_publication_proof',lambda *a:True)
    monkeypatch.setattr(e,'verify_full_gate_metadata',lambda *a:({}, {}, audit))
    calls=[];monkeypatch.setattr(e,'full_physics',lambda root,path,lat:calls.append(path))
    result=e.portable_completed_reference_preparation(tmp_path,'portable/evidence.json')
    assert len(result['reference_archives'])==4 and len(calls)==4 and result['baseline_reserved_seconds']==350
    assert not(tmp_path/e.CANONICAL_JOURNAL).exists()
    assert not Path('/missing/original/Mac/Blender').exists()
    # Recorded reservation is required even if a modified snapshot is hash-pinned.
    journal=read_json= e.read(tmp_path/'portable/journal.json');journal['reservations'].pop();put(tmp_path/'portable/journal.json',journal)
    payload['pinned_inputs']['portable/journal.json']=e.sha(tmp_path/'portable/journal.json');put(tmp_path/'portable/evidence.json',payload)
    with pytest.raises(ValueError,match='canonical charge'):e.portable_completed_reference_preparation(tmp_path,'portable/evidence.json')


def test_original_publication_claim_and_full_physics_are_not_optional(tmp_path,monkeypatch):
    payload,audit=portable_fixture(tmp_path);monkeypatch.setattr(e,'verify_publication_proof',lambda *a:True);monkeypatch.setattr(e,'verify_full_gate_metadata',lambda *a:({}, {}, audit));monkeypatch.setattr(e,'full_physics',lambda *a:True)
    payload['publication_proof']['publications']=[];put(tmp_path/'portable/evidence.json',payload)
    with pytest.raises(ValueError,match='publication proof is incomplete'):e.portable_completed_reference_preparation(tmp_path,'portable/evidence.json')
