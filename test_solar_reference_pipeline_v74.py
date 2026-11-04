"""Procedural metadata and fake-process checks only: no images, rays, NN or child launches."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import json,signal,subprocess
import pytest
import solar_reference_pipeline_v74 as m
import run_solar_reference_pipeline_v74 as runner
import solar_reference_watchdog_v74 as watchdog
ACTUAL_SOURCE_CLOSURE=m.source_closure


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False)if isinstance(value,(dict,list))else value)
    return str(path)


def patched_case_producer():
    import importlib.util
    spec=importlib.util.spec_from_file_location('case_eligibility_patch_v74',Path(__file__).with_name('solar_eval_data_v69.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self):self.elapsed=0.;self.base=datetime(2030,1,1,tzinfo=timezone.utc)
    def now(self):return self.base+timedelta(seconds=self.elapsed)
    def monotonic(self):return self.elapsed
    def advance(self,value):self.elapsed+=value


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);clock=Clock();monkeypatch.setattr(m,'now',clock.now);monkeypatch.setattr(runner,'now',clock.now)
    monkeypatch.setattr(runner.time,'monotonic',clock.monotonic)
    sources=set(m.PREP_SOURCES+m.LOOKUP_SOURCES+m.CASE_SOURCES+m.OWN_SOURCES+('verify_solar_development_v68.py',))
    for name in sources:put(name,Path(watchdog.__file__).read_text()if name=='solar_reference_watchdog_v74.py'else'# procedural source identity\n')
    source={n:m.sha(n)for n in sorted(sources)};monkeypatch.setattr(m,'source_closure',lambda:source)
    put('scene.npz','procedural full-scene identity only');put('raw-reference.npz','procedural raw-reference identity only')
    units=[];specs=[];approved={}
    for i in range(96):
        component=f'g{(i-64)//4:02}'if i>=64 else f'train{i//4:02}'
        split='model_selection'if i>=64 else'training';family='motion';pair=i%4
        units.append(dict(unit_index=i,component=component,split=split,eligible=True,qualification_reasons=[],oracle_outcomes=[dict(correct=True),dict(correct=True)]))
        specs.append(dict(site=dict(centre_latitude_deg=85.,component=component),camera_pairs=[dict(family=family,pair_index=pair)]))
        approved[f'{component}/{family}-{pair}']=dict(reference_geometry_path='raw-reference.npz',reference_geometry_sha256=m.sha('raw-reference.npz'),scene_mesh_path='scene.npz',scene_mesh_sha256=m.sha('scene.npz'),scene_maximum_height_m=120.)
    master=dict(units=specs);audit=dict(units=units,approved_reference_inputs=approved)
    put('master.json',master);put('data.json',dict(complete=True));put('audit.json',audit)
    gate=SimpleNamespace(audit=audit,master=master,original_master_sha256=m.sha('master.json'),dataset_summary_sha256=m.sha('data.json'),geometry_audit_sha256=m.sha('audit.json'))
    monkeypatch.setattr(m,'full_geometry_gate',lambda *args:gate)
    put('ledger.json',dict(accounting_complete=True,remaining_baseline_wall_seconds=66900,remaining_training_wall_seconds=20220))
    put(m.CANONICAL_JOURNAL,dict(ledger_sha256=m.sha('ledger.json'),reservations=[dict(job_id='historical-failed-control',category='baseline',seconds=300,refund_on_failure=False)]))
    put('blender-procedural-identity','not an executable');put('python-procedural-identity','not an executable')
    settings=dict(prepared_only=False,calibration_or_final_access=False,query_truth_in_primary_inference=False,original_master='master.json',dataset_summary='data.json',geometry_audit='audit.json',compute_ledger='ledger.json',reservation_journal=m.CANONICAL_JOURNAL,
        baseline_ceiling_seconds=30000,per_reference_cap_seconds=900,metadata_cap_seconds=600,online_lookup_cap_seconds=900,online_request_cap_seconds=300,publication_dispatch_lease_seconds=1800,dispatch_overhead_seconds=120,pilot_one_cap_seconds=1020,
        preparation_absolute_deadline_utc='2030-01-03T00:00:00+00:00',protocol_directory='protocols',run_directory='runs/physical',reference_assets_index='runs/reference-index.json',case_assets_output='runs/cases',stage_journal='runs/physical-state.json',blender_executable='blender-procedural-identity',python_executable='python-procedural-identity')
    put('settings.json',settings)
    return SimpleNamespace(clock=clock,settings=settings,gate=gate,contract='protocols/contract.json',source=source,monkeypatch=monkeypatch)


def initialize(s):return m.initialize('settings.json',s.contract)


def test_full_gate_precedes_all_protocol_outputs(setup):
    setup.monkeypatch.setattr(m,'full_geometry_gate',lambda *a:(_ for _ in()).throw(ValueError('full96 unavailable')))
    with pytest.raises(ValueError,match='full96'):initialize(setup)
    assert not Path('protocols').exists()and not Path('runs/physical').exists()


def test_all32_reference_protocols_frozen_and_pilot_spread(setup):
    c=initialize(setup)
    assert c['selection_units']==list(range(64,96))and c['pilot_units']==[64,72,80,88]
    assert len(list(Path('protocols/references').glob('*.json')))==64
    for index in c['selection_units']:
        pre,lookup=m.physical_protocols(c,index)
        assert pre['precompute']=='coarse'and pre['mesh']==pre['scene_mesh']=='scene.npz'
        assert pre['reference_geometry']=='raw-reference.npz'and pre['query_truth_used']is False
        assert set(lookup)=={'schema_version','cpu_threads','query_truth_used','max_lookup_calendar_seconds','max_request_seconds','mesh','mesh_sha256','scene_mesh','scene_mesh_sha256','reference_geometry','reference_geometry_sha256','geometry_audit','geometry_audit_sha256','approved_reference_key','source_sha256'}
        assert not any('query_'in k for k in pre if k!='query_truth_used')


def change_independent_eligibility(s,indices,reason='query:incomplete_finite_surface'):
    for i in indices:s.gate.audit['units'][i].update(eligible=False,qualification_reasons=[reason],oracle_outcomes=[])
    put('audit.json',s.gate.audit);s.gate.geometry_audit_sha256=m.sha('audit.json')


def test_31eligible_of32_generated_preserves_exclusion_and_pose_failure(setup):
    change_independent_eligibility(setup,[70])
    setup.gate.audit['units'][71]['oracle_outcomes'][0]['correct']=False
    put('audit.json',setup.gate.audit);setup.gate.geometry_audit_sha256=m.sha('audit.json')
    c=initialize(setup)
    assert len(c['selection_units'])==31 and 70 not in c['selection_units']and 71 in c['selection_units']
    assert len(c['generated_selection_units'])==32 and c['generated_selection_unit_count']==32 and c['eligible_selection_unit_count']==31
    assert c['excluded_selection_units']==[setup.gate.audit['units'][70]]and len(c['selection_groups'])==8
    assert len(list(Path('protocols/references').glob('*.json')))==62
    assert set(m.asset_protocol(c,'reference_asset_index')['reference_unit_protocols'])=={str(i)for i in c['selection_units']}


def test_zero_eligible_group_is_explicit_without_dropping_raw_group(setup):
    change_independent_eligibility(setup,range(64,68));c=initialize(setup)
    assert len(c['selection_units'])==28 and len(c['terrain_groups'])==8 and c['zero_eligible_terrain_groups']==['g00']
    assert len({c['reference_units'][str(i)]['component']for i in c['pilot_units']})==4


def test_four_group_pilot_shortfall_is_recorded_before_jobs(setup):
    change_independent_eligibility(setup,range(76,96))
    with pytest.raises(m.PilotShortfall,match='shortfall'):initialize(setup)
    r=m.read(setup.contract+'.SHORTFALL.json')
    assert r['physical_jobs_admitted']is False and r['generated_selection_unit_count']==32 and r['eligible_selection_unit_count']==12
    assert len(r['zero_eligible_terrain_groups'])==5 and not Path('protocols/references').exists()
    assert len(m.read(m.CANONICAL_JOURNAL)['reservations'])==1


def test_prepared_settings_have_no_admission(setup):
    put('settings.json',dict(setup.settings,prepared_only=True))
    with pytest.raises(ValueError,match='Prepared'):initialize(setup)


def test_pending_phase_is_immutable_and_not_retimed(setup):
    c=initialize(setup);first=m.prepare_next(setup.contract);saved=Path(first['protocol']).read_bytes()
    setup.clock.advance(20);second=m.prepare_next(setup.contract)
    assert first['protocol']==second['protocol']and saved==Path(second['protocol']).read_bytes()
    p=m.read(first['protocol']);m.validate_phase(c,first['protocol'],p)
    p['absolute_deadline_utc']='2030-01-02T00:00:00+00:00'
    with pytest.raises(ValueError,match='deadline'):m.validate_phase(c,first['protocol'],p)


@pytest.mark.parametrize('bad',[True,0,-1,float('nan'),float('inf')])
def test_measured_projection_rejects_fake_or_unbounded_cost(setup,bad):
    c=initialize(setup)
    with pytest.raises(ValueError):m.measured_cap(c,[dict(outer_calendar_seconds=bad)],28)


def test_projection_uses_two_times_max_external_not_internal_mean(setup):
    c=initialize(setup)
    assert m.measured_cap(c,[dict(outer_calendar_seconds=7,internal_calendar_seconds=1),dict(outer_calendar_seconds=10,internal_calendar_seconds=1)],28)==680


def test_canonical_history_and_no_refund(setup):
    c=initialize(setup);before=m.remaining(c);j=m.read(m.CANONICAL_JOURNAL)
    j['reservations'].append(dict(job_id='failed-physical',category='baseline',seconds=999,refund_on_failure=False));put(m.CANONICAL_JOURNAL,j)
    assert m.remaining(c)==before-999
    j['reservations'][0]['seconds']=1;put(m.CANONICAL_JOURNAL,j)
    with pytest.raises(ValueError,match='history'):m.remaining(c)


def test_canonical_alias_is_not_a_fresh_budget(setup):
    c=initialize(setup);c['reservation_journal']='some-other-journal.json'
    with pytest.raises(ValueError,match='canonical'):m.remaining(c)


def test_missing_publication_stops_before_reservation_or_executor(setup):
    initialize(setup);before=Path(m.CANONICAL_JOURNAL).read_bytes();called=[]
    def not_published(*a):raise m.PublicationPending('prepared, not published')
    result=runner.run(setup.contract,'a'*40,executor=lambda *a:called.append(a),publication_verifier=not_published)
    assert result['status']=='awaiting_source_publication'and not called and Path(m.CANONICAL_JOURNAL).read_bytes()==before
    assert not Path('runs/physical/stages/reference_one').exists()


def accept(contract,path,commit):
    c=m.load_contract(contract);p=m.read(path);m.validate_phase(c,path,p);return c,p


def test_failed_child_keeps_charge_and_prohibits_retry(setup):
    initialize(setup)
    def failed(*args):raise RuntimeError('procedural no-child failure')
    with pytest.raises(RuntimeError,match='procedural'):runner.run(setup.contract,'a'*40,executor=failed,publication_verifier=accept)
    j=m.read(m.CANONICAL_JOURNAL);assert j['reservations'][-1]['seconds']==1020 and j['reservations'][-1]['refund_on_failure']is False
    before=Path(m.CANONICAL_JOURNAL).read_bytes()
    with pytest.raises(ValueError,match='partial/failed'):runner.run(setup.contract,'a'*40,executor=failed,publication_verifier=accept)
    assert before==Path(m.CANONICAL_JOURNAL).read_bytes()


def test_expired_original_dispatch_lease_cannot_reset(setup):
    initialize(setup);m.prepare_next(setup.contract);setup.clock.advance(1801)
    with pytest.raises(TimeoutError,match='lease'):runner.run(setup.contract,'a'*40,executor=lambda *a:None,publication_verifier=accept)
    assert len(m.read(m.CANONICAL_JOURNAL)['reservations'])==1


def test_whitelisted_reference_command_has_external_python_exit_control():
    c=dict(blender_executable='/fake/blender',python_executable='/fake/python');j=dict(kind='reference',program='prepare_solar_reference_v63.py',protocol='p.json')
    cmd=watchdog.command(c,j,'a'*40)
    assert cmd[cmd.index('--threads')+1]=='2'and cmd[cmd.index('--python-exit-code')+1]=='3'
    j['program']='render_solar_development_v67.py'
    with pytest.raises(ValueError,match='No renderer'):watchdog.command(c,j,'a'*40)


class FakeChild:
    def __init__(self,clock,hang=False,term_resistant=False):self.clock=clock;self.pid=1001;self.returncode=None;self.hang=hang;self.term_resistant=term_resistant;self.killed=False;self.waits=[]
    def poll(self):
        if self.killed:self.returncode=-9
        elif not self.hang:self.returncode=0
        return self.returncode
    def wait(self,timeout=None):
        self.waits.append(timeout)
        if self.term_resistant and not self.killed:
            self.clock.advance(timeout);raise subprocess.TimeoutExpired(['dummy'],timeout)
        self.clock.advance(.01);self.returncode=-9 if self.killed else -15;return self.returncode
    def terminate(self):self.returncode=-15
    def kill(self):self.killed=True;self.returncode=-9


def fake_watchdog_env(setup,*,hang=False,term_resistant=False,interrupt_in_popen=False):
    clock=setup.clock;child=FakeChild(clock,hang,term_resistant);awake=FakeChild(clock,True);pending=[];calls=[];kills=[]
    def popen(argv,**kw):
        calls.append((argv,kw))
        if len(calls)==1:
            assert kw['start_new_session']is True
            if interrupt_in_popen:pending.append(signal.SIGTERM)
            clock.advance(.1);return child
        return awake
    def killpg(pid,sig):
        kills.append(sig)
        if sig==signal.SIGKILL:child.killed=True
        elif not term_resistant:child.returncode=-15;child.hang=True
    setup.monkeypatch.setattr(watchdog,'now',clock.now);setup.monkeypatch.setattr(watchdog.time,'monotonic',clock.monotonic)
    setup.monkeypatch.setattr(watchdog.time,'sleep',clock.advance);setup.monkeypatch.setattr(watchdog.subprocess,'Popen',popen);setup.monkeypatch.setattr(watchdog.os,'killpg',killpg)
    c=initialize(setup);ready=m.prepare_next(setup.contract);p=m.read(ready['protocol']);p['_protocol_sha256']=m.sha(ready['protocol']);p['_protocol_path']=ready['protocol']
    return c,p,p['jobs'][0],pending,child,calls,kills


def test_external_watchdog_success_is_procedural_receipt_only(setup):
    c,p,job,pending,child,calls,kills=fake_watchdog_env(setup)
    w=watchdog.execute(c,p,job,'a'*40,pending,setup.clock.monotonic())
    assert w['complete']is True and w['child_returncode']==0 and not kills and len(calls)==2
    assert all(v is not None for v in child.waits)
    assert calls[0][1]['env']['OMP_NUM_THREADS']=='2'
    m.validate_watchdog(c,p['_protocol_path'],m.read(p['_protocol_path']),job)
    # Exit zero alone is never a complete physical archive.
    with pytest.raises(FileNotFoundError):m.validate_reference(c,job['unit_index'],validate_arrays=False)


def test_external_watchdog_catches_startup_hang_and_escalates_boundedly(setup):
    c,p,job,pending,child,calls,kills=fake_watchdog_env(setup,hang=True,term_resistant=True)
    with pytest.raises(TimeoutError):watchdog.execute(c,p,job,'a'*40,pending,setup.clock.monotonic())
    assert kills==[signal.SIGTERM,signal.SIGKILL]and child.returncode==-9
    w=m.read(job['watchdog_receipt']);assert w['complete']is False and w['failure']['error_type']=='TimeoutError'
    assert all(v is not None and v<=3 for v in child.waits)
    assert w['calendar_seconds']<=p['max_calendar_seconds']


def test_pending_signal_during_popen_never_loses_child_cleanup(setup):
    c,p,job,pending,child,calls,kills=fake_watchdog_env(setup,hang=True,interrupt_in_popen=True)
    with pytest.raises(InterruptedError):watchdog.execute(c,p,job,'a'*40,pending,setup.clock.monotonic())
    assert kills==[signal.SIGTERM]and child.returncode==-15
    assert m.read(job['watchdog_receipt'])['complete']is False


def synthetic_executor(s):
    def execute(c,p,job,commit,pending,tick):
        start=s.clock.now();pre=m.read(job['protocol'])
        if job['kind']=='reference':
            paths=m.unit_paths(c,job['unit_index']);put(paths['archive'],'PROCEDURAL STUB ONLY: not an NPZ or physical result')
            row=c['reference_units'][str(job['unit_index'])]
            put(paths['archive']+'.json',dict(complete=True,output_sha256=m.sha(paths['archive']),protocol_sha256=m.sha(job['protocol']),source_sha256=pre['source_sha256'],publication_commit=commit,query_truth_used=False,neural_forwards=0,precompute='coarse',original_scene_mesh_sha256=row['approved']['scene_mesh_sha256'],original_scene_ceiling_m=row['approved']['scene_maximum_height_m'],map_mesh_sha256=pre['mesh_sha256'],calendar_seconds=.25))
        else:
            base=dict(complete=True,split='model_selection',calibration_or_final_access=False,source_sha256=pre['source_sha256'],protocol_sha256=m.sha(job['protocol']),geometry_audit_sha256=c['geometry_audit_sha256'],dataset_summary_sha256=c['dataset_summary_sha256'],original_master_sha256=c['original_master_sha256'],**m.selection_metadata(s.gate.audit))
            if p['stage']=='reference_index':put(c['reference_assets_index'],dict(base,schema_version=69,units=[dict(unit_index=i)for i in c['selection_units']]))
            else:
                root=Path(c['case_assets_output']);put(root/'manifest.json',dict(base,schema_version=70,eligible_unit_indices=c['selection_units'],positive_cases=6*len(c['selection_units']),negative_cases=6*len(c['selection_units']),cases=[dict(procedural=True)for _ in range(12*len(c['selection_units']))]))
                put(root/'scoring-stub.txt','Separate procedural scoring placeholder, no query inference archive')
                put(root/'COMPLETE.json',dict(sha256={str(x.relative_to(root)):m.sha(x)for x in root.rglob('*')if x.is_file()}))
        s.clock.advance(.5)
        put(job['watchdog_receipt'],dict(complete=True,failure=None,child_returncode=0,program=job['program'],kind=job['kind'],child_protocol_sha256=m.sha(job['protocol']),publication_commit=commit,watchdog_source_sha256=c['source_sha256']['solar_reference_watchdog_v74.py'],phase_protocol=p['_protocol_path'],phase_protocol_sha256=p['_protocol_sha256'],started_utc=start.isoformat(),ended_utc=s.clock.now().isoformat(),work_deadline_utc=min(start+timedelta(seconds=job['work_cap_seconds']),m.utc(p['absolute_deadline_utc'])-timedelta(seconds=5)).isoformat(),phase_hard_deadline_utc=p['absolute_deadline_utc'],work_cap_seconds=job['work_cap_seconds'],calendar_seconds=.5))
    return execute


def test_serial_protocol_chain_and_completed_physical_evidence_are_metadata_only(setup):
    import numpy as np
    import solar_eval_data_v69 as data
    setup.monkeypatch.setattr(data,'validate_reference_archive',lambda p:dict(solar_materialized_mask=np.array([True,False]),coarse_sun_mask=np.array([True,False])))
    setup.monkeypatch.setattr(data,'validate_generic_catalogue',lambda *a:True)
    initialize(setup)
    result=runner.run(setup.contract,'a'*40,executor=synthetic_executor(setup),publication_verifier=accept)
    assert result['reference_preparation_complete']is True and result['scientific_goal_complete']is False
    c=m.load_contract(setup.contract);fullpath,root=m.stage_paths(c,'reference_full')
    # This evidence replay must never parse our deliberately non-NPZ stub archives.
    setup.monkeypatch.setattr(data,'validate_reference_archive',lambda p:(_ for _ in()).throw(AssertionError('array parse forbidden during evidence replay')))
    setup.monkeypatch.setattr(m,'verify_publication',lambda *a,**kw:None)
    proof=m.completed_reference_preparation(fullpath,str(root/'summary.json'))
    assert len(proof['reference_archives'])==32 and proof['reference_unit_indices']==list(range(64,96))
    assert sum(proof['outer_calendar_seconds_by_archive'].values())==16 and sum(proof['internal_calendar_seconds_by_archive'].values())==8
    assert proof['baseline_reserved_seconds']==1020+123+148
    assert len(m.read(m.CANONICAL_JOURNAL)['reservations'])==6
    # Original deadline expiry closes launch authority but allows read-only completed evidence.
    setup.clock.advance(3*24*3600)
    with pytest.raises(TimeoutError):m.load_contract(setup.contract)
    assert m.completed_reference_preparation(fullpath,str(root/'summary.json'))==proof
    put('runs/cases/scoring-stub.txt','tampered')
    with pytest.raises(ValueError,match='checksum'):m.completion(c,'case_assets',True)


def test_reference_receipt_cannot_change_protocol_or_watchdog_source(setup):
    import numpy as np
    import solar_eval_data_v69 as data
    setup.monkeypatch.setattr(data,'validate_reference_archive',lambda p:dict(solar_materialized_mask=np.array([True]),coarse_sun_mask=np.array([True])))
    setup.monkeypatch.setattr(data,'validate_generic_catalogue',lambda *a:True)
    initialize(setup)
    def only_first(contract,path,commit):
        if m.read(path)['stage']!='reference_one':raise m.PublicationPending('later stage not yet published')
        return accept(contract,path,commit)
    runner.run(setup.contract,'a'*40,executor=synthetic_executor(setup),publication_verifier=only_first)
    c=m.load_contract(setup.contract);paths=m.unit_paths(c,64);w=m.read(paths['watchdog']);w['watchdog_source_sha256']='0'*64;put(paths['watchdog'],w)
    with pytest.raises(ValueError,match='watchdog provenance'):m.validate_reference(c,64,validate_arrays=False)


def test_source_closure_covers_module_imports_and_declared_lazy_entrypoints(setup):
    # Exercise the actual AST closure in a pure metadata sandbox, rather than invoking imports.
    seeds=set(m.PREP_SOURCES+m.LOOKUP_SOURCES+m.CASE_SOURCES+m.OWN_SOURCES+('solar_learning_pipeline_v73.py','solar_compute_budget_v67.py','solar_training_bank_v68.py','verify_solar_development_v68.py'))
    for n in seeds:
        if not Path(n).exists():put(n,'# procedural declared entrypoint\n')
    put('module_dependency.py','VALUE=1\n');put('function_only_dependency.py','VALUE=2\n')
    put('solar_map_v63.py','import module_dependency\ndef unused():\n    import function_only_dependency\n')
    closure=ACTUAL_SOURCE_CLOSURE()
    assert seeds<=set(closure)and 'module_dependency.py'in closure and 'function_only_dependency.py'not in closure


def test_actual_publication_verifier_pins_all_reference_protocols(setup):
    initialize(setup);ready=m.prepare_next(setup.contract);seen=[]
    def git_show(argv,**kwargs):
        path=argv[2].split(':',1)[1];seen.append(path)
        return Path(path).read_bytes()
    setup.monkeypatch.setattr(m.subprocess,'check_output',git_show)
    c,p=m.verify_publication(setup.contract,ready['protocol'],'a'*40)
    assert len([p for p in seen if '/references/'in p])==64
    lookup=m.unit_paths(c,95)['lookup']
    def changed_git_show(argv,**kwargs):
        path=argv[2].split(':',1)[1]
        return b'old public bytes'if path==lookup else Path(path).read_bytes()
    setup.monkeypatch.setattr(m.subprocess,'check_output',changed_git_show)
    with pytest.raises(m.PublicationPending,match='differs'):m.verify_publication(setup.contract,ready['protocol'],'a'*40)
    put(lookup,dict(m.read(lookup),query_truth_used=True))
    with pytest.raises(ValueError,match='no new information'):m.load_contract(setup.contract)


def test_too_small_metadata_allowance_is_rejected_before_output(setup):
    put('settings.json',dict(setup.settings,metadata_cap_seconds=4))
    with pytest.raises(ValueError,match='cleanup'):initialize(setup)
    assert not Path('protocols').exists()


def test_completed_pilot_remains_chain_evidence_after_its_own_dispatch_lease(setup):
    import numpy as np
    import solar_eval_data_v69 as data
    setup.monkeypatch.setattr(data,'validate_reference_archive',lambda p:dict(solar_materialized_mask=np.array([True]),coarse_sun_mask=np.array([True])))
    setup.monkeypatch.setattr(data,'validate_generic_catalogue',lambda *a:True)
    initialize(setup)
    def one_only(contract,path,commit):
        if m.read(path)['stage']!='reference_one':raise m.PublicationPending('next stage requires publication')
        return accept(contract,path,commit)
    runner.run(setup.contract,'a'*40,executor=synthetic_executor(setup),publication_verifier=one_only)
    c=m.load_contract(setup.contract);oldpath,_=m.stage_paths(c,'reference_one');newpath,_=m.stage_paths(c,'reference_four')
    # First pilot start lease ends at1800; its finished next stage's own lease ends at1800.5.
    setup.clock.advance(1799.75)
    assert setup.clock.now()>m.utc(m.read(oldpath)['latest_start_utc'])and setup.clock.now()<m.utc(m.read(newpath)['latest_start_utc'])
    prior=m.completion(c,'reference_one',True);assert prior['complete']is True
    old_charge=[r for r in m.read(m.CANONICAL_JOURNAL)['reservations']if r['job_id']==m.sha(oldpath)+'/reference_physics']
    def four_only(contract,path,commit):
        if m.read(path)['stage']!='reference_four':raise m.PublicationPending('full phase requires publication')
        return accept(contract,path,commit)
    result=runner.run(setup.contract,'a'*40,executor=synthetic_executor(setup),publication_verifier=four_only)
    assert result['stage']=='reference_full'and result['status']=='awaiting_source_publication'
    assert m.completion(c,'reference_four',True)['complete']is True
    assert [r for r in m.read(m.CANONICAL_JOURNAL)['reservations']if r['job_id']==m.sha(oldpath)+'/reference_physics']==old_charge


@pytest.mark.parametrize('excluded',[[70],list(range(64,68))])
def test_dynamic_eligible_pipeline_keeps_all_raw_inventory_and_case_denominator(setup,excluded):
    import numpy as np
    import solar_eval_data_v69 as data
    change_independent_eligibility(setup,excluded)
    setup.monkeypatch.setattr(data,'validate_reference_archive',lambda p:dict(solar_materialized_mask=np.array([True]),coarse_sun_mask=np.array([True])))
    setup.monkeypatch.setattr(data,'validate_generic_catalogue',lambda *a:True)
    initialize(setup);runner.run(setup.contract,'a'*40,executor=synthetic_executor(setup),publication_verifier=accept)
    c=m.load_contract(setup.contract);manifest=m.read(Path(c['case_assets_output'])/'manifest.json');n=32-len(excluded)
    assert manifest['positive_cases']==manifest['negative_cases']==6*n and len(manifest['cases'])==12*n
    assert manifest['generated_selection_unit_count']==32 and len(manifest['generated_selection_units'])==32 and len(manifest['terrain_groups'])==8
    assert [r['unit_index']for r in manifest['excluded_selection_units']]==excluded
    assert all(r['qualification_reasons']for r in manifest['excluded_selection_units'])
    fullpath,fullroot=m.stage_paths(c,'reference_full');p=m.read(fullpath)
    assert len(p['jobs'])==n-4 and p['max_calendar_seconds']==m.measured_cap(c,p['prior_reference_rows'],n-4)
    setup.monkeypatch.setattr(m,'verify_publication',lambda *a,**kw:None)
    evidence=m.completed_reference_preparation(fullpath,str(fullroot/'summary.json'))
    assert len(evidence['reference_archives'])==n and len(evidence['generated_selection_units'])==32
    assert evidence['zero_eligible_terrain_groups']==manifest['zero_eligible_terrain_groups']


def test_patched_case_gate_keeps31_eligible_with_failed_oracle_direction():
    from test_solar_eval_data_v69 import gate_fixture,unit_fixture
    data=patched_case_producer()
    p,d,a,master,assets=gate_fixture();a['units'][70].update(eligible=False,qualification_reasons=['query:incomplete_finite_surface'])
    a['units'][71]['oracle_outcomes']=[dict(correct=False),dict(correct=True)]
    assets['units']=[r for r in assets['units']if r['unit_index']!=70]
    a.update(oracle_pose_directions=190,correct_oracle_poses=189,oracle_success=189/190)
    selected=data.validate_gate(p,d,a,master,assets)
    assert len(selected)==31 and 71 in selected and not selected[71]['oracle_outcomes'][0]['correct']
    inventory=data.selection_metadata(a)
    assert inventory['generated_selection_unit_count']==32 and inventory['eligible_selection_unit_count']==31
    units,_=unit_fixture();del units[6]
    cases=data.plan_cases(list(units.values()));positives=[r for r in cases if r['positive']];negatives=[r for r in cases if not r['positive']]
    assert len(positives)==len(negatives)==31*6
    assert all(r['unit_index']in units and r['query_unit_index']in units for r in cases)


def test_patched_case_gate_reports_empty_selection_group_without_excluding_other_cases():
    from test_solar_eval_data_v69 import gate_fixture
    data=patched_case_producer()
    p,d,a,master,assets=gate_fixture()
    for i in range(64,68):a['units'][i].update(eligible=False,qualification_reasons=['query:incomplete_finite_surface'])
    assets['units']=[r for r in assets['units']if r['unit_index']not in range(64,68)]
    a.update(oracle_pose_directions=184,correct_oracle_poses=184,oracle_success=1.)
    assert len(data.validate_gate(p,d,a,master,assets))==28
    inventory=data.selection_metadata(a)
    assert len(inventory['terrain_groups'])==8 and inventory['zero_eligible_terrain_groups']==[a['units'][64]['component']]


def test_exactly_four_terrain_pilot_refs_need_no_repeated_full_stage_job(setup):
    import numpy as np
    import solar_eval_data_v69 as data
    keep={64,68,72,76};change_independent_eligibility(setup,[i for i in range(64,96)if i not in keep])
    setup.monkeypatch.setattr(data,'validate_reference_archive',lambda p:dict(solar_materialized_mask=np.array([True]),coarse_sun_mask=np.array([True])))
    setup.monkeypatch.setattr(data,'validate_generic_catalogue',lambda *a:True)
    initialize(setup)
    def pilot_only(contract,path,commit):
        if m.read(path)['stage']not in('reference_one','reference_four'):raise m.PublicationPending('full phase requires publication')
        return accept(contract,path,commit)
    result=runner.run(setup.contract,'a'*40,executor=synthetic_executor(setup),publication_verifier=pilot_only)
    c=m.load_contract(setup.contract);p=m.read(result['protocol']);m.validate_phase(c,result['protocol'],p)
    assert result['stage']=='reference_full'and p['jobs']==[] and p['max_calendar_seconds']==c['dispatch_overhead_seconds']
    assert set(c['selection_units'])==keep and len(c['generated_selection_units'])==32 and len(c['zero_eligible_terrain_groups'])==4
