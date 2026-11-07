"""Fake-artifact end-to-end workflow. No image exists and no model is loaded."""
import ast
import copy
from datetime import datetime, timezone, timedelta
import inspect
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
REAL_CHECK_OUTPUT=subprocess.check_output

HERE=Path(__file__).resolve().parent
REPO=HERE if(HERE/'solar_frontend_v63.py').exists()else HERE.parents[1]/'outputs'/'lunar-image-matching'
sys.path.insert(0,str(HERE));sys.path.append(str(REPO))
import solar_learning_pipeline_v73 as pipeline
import run_solar_learning_pipeline_v73 as runner


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n');return str(path)


@pytest.fixture
def workspace(tmp_path,monkeypatch):
    # Source code and tiny JSON/identity files only. Missing fake PNGs are an
    # access guard: a builder that attempts scientific decoding cannot pass.
    for source in REPO.glob('*.py'):shutil.copy2(source,tmp_path/source.name)
    for name in pipeline.PIPELINE_SOURCES+('run_solar_job_v70.py',):shutil.copy2(HERE/name,tmp_path/name)
    monkeypatch.chdir(tmp_path)
    from north_render_plan_v31 import plan
    sites=[s for s in plan()['sites']if s['split']in('training','model_selection')]
    lighting=write('reports/lighting.json',dict(procedural_fixture=True))
    units=[];records=[];checked=[]
    for site in sites:
        for pair in range(4):
            i=len(units);folder=Path('data')/str(i)
            images=[dict(role=role,light=light,path=f'MUST_NOT_OPEN_{role}_{light}.png',sha256='0'*64)for role in('reference','query')for light in range(24)]
            summary=write(folder/'summary.json',dict(complete=True,rendered_images=48,images=images,matcher_forwards=0,optimizer_updates=0))
            seal=write(folder/'COMPLETE.json',dict(procedural_fixture=True))
            units.append(dict(site=site,camera_pairs=[dict(family='training_motion',pair_index=pair)]))
            r=dict(unit_index=i,output=str(folder),summary_sha256=pipeline.sha(summary),seal_sha256=pipeline.sha(seal));records.append(r)
            checked.append(dict(unit_index=i,component=site['component'],split=site['split'],eligible=True,
                source_summary_sha256=r['summary_sha256'],source_seal_sha256=r['seal_sha256']))
    master=write('reports/master.json',dict(units=units,calibration_or_final_access=False,source_schedule_sha256=pipeline.sha(lighting)))
    dataset=write('reports/dataset.json',dict(complete=True,units=records,original_master_sha256=pipeline.sha(master),calibration_or_final_access=False))
    audit=write('reports/audit.json',dict(passed=True,complete_dataset=True,expected_units=96,completed_units=96,
        terrain_counts=dict(training=16,model_selection=8),original_master_sha256=pipeline.sha(master),completion_summary_sha256=pipeline.sha(dataset),
        oracle_success=1.,oracle_pose_directions=192,correct_oracle_poses=192,units=checked,audit_source_sha256=pipeline.sha('verify_solar_development_v68.py')))
    initializer=Path('runs/initializer');initializer.mkdir(parents=True)
    (initializer/'selected.pt').write_bytes(b'PROCEDURAL IDENTITY ONLY, NOT A TORCH CHECKPOINT')
    write(initializer/'training.json',dict(procedural_fixture=True))
    write(initializer/'COMPLETE.json',dict(checkpoint_sha256=pipeline.sha(initializer/'selected.pt'),training_sha256=pipeline.sha(initializer/'training.json')))
    ledger=write('reports/ledger.json',dict(accounting_complete=True,remaining_training_wall_seconds=20220,remaining_baseline_wall_seconds=67200))
    write(pipeline.CANONICAL_JOURNAL,dict(ledger_sha256=pipeline.sha(ledger),reservations=[dict(job_id='historical/candidate',category='baseline',seconds=300,refund_on_failure=False)]))
    settings=dict(prepared_only=False,calibration_or_final_access=False,training_ceiling_seconds=20220,baseline_ceiling_seconds=66900,
        original_master=master,dataset_summary=dataset,geometry_audit=audit,lighting_schedule=lighting,
        compute_ledger=ledger,reservation_journal=pipeline.CANONICAL_JOURNAL,initializer=str(initializer),expected_frozen_descriptor_sha256='f'*64,
        protocol_directory='reports/learning',run_directory='runs/learning',stage_journal='runs/learning-state.json',
        publication_dispatch_lease_seconds=1800,
        teacher_pilot_cap_seconds=100,training_pilot_cap_seconds=100,training_replay_cap_seconds=100,
        maximum_points_per_unit=64,teacher_batch_size=64,trial_pose_jitter=dict(translation_sigma_m=1.,rotation_sigma_deg=.25),
        selection_evidence={mode:dict(protocol=f'reports/selection-{mode}.json',summary=f'runs/selection-{mode}/summary.json',metric_replay=f'reports/replay-{mode}.json')for mode in('sh','pooled')})
    settings_path=write('reports/settings.json',settings)
    return dict(root=tmp_path,settings=settings,settings_path=settings_path,contract='reports/learning-contract.json',published={},calls=[])


def initialize(w):return pipeline.initialize(w['settings_path'],w['contract'])


def publish(w):
    # In-memory publication snapshot; never git commit/push or remote publish.
    for path in Path('.').rglob('*'):
        if path.is_file()and(path.suffix in('.py','.json')):w['published'][path.as_posix()]=path.read_bytes()


def fake_git(w):
    def verify(argv,**kwargs):
        if argv[:2]!=['git','show']:return REAL_CHECK_OUTPUT(argv,**kwargs)
        name=argv[2].split(':',1)[1]
        if name not in w['published']:raise subprocess.CalledProcessError(128,argv)
        return w['published'][name]
    return verify


def fake_executor(w,training_elapsed=2.,fail=False):
    # Signature intentionally EXACTLY matches the real run_solar_job_v70 API.
    def execute(protocol_path,commit,python,program,replay_root=None,replay_output=None):
        from solar_compute_budget_v67 import reserve
        p=pipeline.read(protocol_path);mode='replay'if replay_root else'run';stage=p['pipeline_stage']
        cap=p['replay_cap_seconds']if replay_root else p['max_calendar_seconds'];suffix='teacher_bank'if program=='build_solar_training_bank_v68.py'else'exact_replay'if replay_root else'train'
        reserve(p['compute_ledger'],p['compute_ledger_sha256'],p['reservation_journal'],pipeline.sha(protocol_path)+'/'+suffix,'training',cap)
        w['calls'].append((stage,mode,program,replay_root,replay_output))
        if fail:
            Path(p['watchdog_log_prefix']+mode+'.log').write_text('PROCEDURAL FAILURE')
            raise RuntimeError('procedural child failure after reservation')
        out=Path(p['output']);out.mkdir(parents=True,exist_ok=True)
        if replay_root:
            assert program=='train_solar_response_v63.py'and replay_output==p['exact_replay_output']and replay_root==p['output']
            write(replay_output,dict(passed=True,updates=100,exact_model_optimizer_rng_loss_gradient_sampler=True,
                protocol_sha256=pipeline.sha(protocol_path),pilot_summary_sha256=pipeline.sha(out/'summary.json'),calendar_seconds=training_elapsed))
        elif program=='build_solar_training_bank_v68.py':
            import cv2,torch,numpy as np
            c=pipeline.read(w['contract']);indices=c['eligible_training_unit_indices']if p['phase']=='full_bank'else[
                min(int(i)for i,g in c['training_unit_components'].items()if g==group)for group in c['training_groups'][:4]]
            records=[dict(unit_index=i,component=c['training_unit_components'][str(i)],selected_physical_points=64)for i in indices]
            summary=dict(complete=True,phase=p['phase'],protocol_sha256=pipeline.sha(protocol_path),heldout_image_reads=0,
                records=records,fresh_teacher_records=len(records)if p['phase']=='cost_pilot'else 60,reused_complete_records=0 if p['phase']=='cost_pilot'else 4,
                calendar_seconds=1.,maximum_points_per_unit=64,teacher_batch_size=64,trial_pose_jitter=p['trial_pose_jitter'],geometry_gate_sha256=p['geometry_audit_sha256'],
                teacher_checkpoint_sha256=p['teacher_checkpoint_sha256'],source_sha256={n:p['source_sha256'][n]for n in pipeline.TEACHER_SOURCES})
            summary['environment']=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,torch=torch.__version__,platform=platform.platform(),processor=platform.processor(),torch_num_threads=2)
            manifest_sha=None
            if p['phase']=='full_bank':
                write(p['manifest'],dict(schema_version=63,split='training',records=records,terrain_groups=c['training_groups'],
                    producer_completion_path=str(out/'COMPLETE.json'),geometry_audit_path=c['geometry_audit'],geometry_audit_sha256=c['geometry_audit_sha256']))
                manifest_sha=pipeline.sha(p['manifest']);summary['manifest_sha256']=manifest_sha
            write(out/'summary.json',summary)
            write(out/'COMPLETE.json',dict(complete=True,protocol_sha256=pipeline.sha(protocol_path),summary_sha256=pipeline.sha(out/'summary.json'),manifest_sha256=manifest_sha))
        else:
            assert program=='train_solar_response_v63.py';checkpoints={}
            for step in((0,100)if p['updates']==100 else(0,100,500,1000,1500,2000)):
                cp=out/f'step_{step}.pt';cp.write_bytes(f'PROCEDURAL CHECKPOINT ONLY {stage}/{step}'.encode());checkpoints[cp.name]=pipeline.sha(cp)
            hist=write(out/f'history_{p["updates"]}.json',dict(procedural_fixture=True,updates=p['updates']))
            write(out/'summary.json',dict(complete=True,protocol_sha256=pipeline.sha(protocol_path),updates=p['updates'],seed=p['seed'],pooled=p['pooled'],
                calendar_seconds=training_elapsed,checkpoint_sha256=checkpoints,all_history_sha256=pipeline.sha(hist)))
        start=datetime.now(timezone.utc);original=pipeline.utc(p['replay_absolute_deadline_utc']if replay_root else p['absolute_deadline_utc'])
        write(p['watchdog_receipts'][mode],dict(complete=True,mode=mode,program=program,protocol_sha256=pipeline.sha(protocol_path),
            source_sha256=p['source_sha256']['run_solar_job_v70.py'],started_utc=start.isoformat(),absolute_deadline_utc=min(start+timedelta(seconds=cap),original).isoformat(),
            cap_seconds=cap,calendar_seconds=max(3.,training_elapsed+.1),child_returncode=0,failure=None))
    return execute


def advance_to_selection(w,monkeypatch,training_elapsed=2.):
    initialize(w);monkeypatch.setattr(pipeline.subprocess,'check_output',fake_git(w))
    result=runner.run(w['contract'],'f'*40,executor=fake_executor(w,training_elapsed))
    while result.get('awaiting_source_publication'):
        publish(w);result=runner.run(w['contract'],'f'*40,executor=fake_executor(w,training_elapsed))
    return result


def create_selection(w):
    from solar_evaluation_v70 import checkpoint_profiles
    from solar_compute_budget_v67 import reserve
    c=pipeline.read(w['contract'])
    for mode in('sh','pooled'):
        study=pipeline.completion(c,mode+'_study',True);spec=c['selection_evidence'][mode]
        profiles=checkpoint_profiles(Path(study['summary']).parent,pooled=mode=='pooled',training_protocol=study['protocol'])
        p=dict(split='model_selection',purpose='checkpoint_selection',tracks=['estimated_sun'],calibration_or_final_access=False,case_ids=['positive','negative'],profiles=profiles,
            source_sha256={'verify_solar_evaluation_v70.py':pipeline.sha('verify_solar_evaluation_v70.py')})
        write(spec['protocol'],p);rows=[]
        for profile in profiles:
            for positive in(True,False):
                rows.append(dict(profile=profile['name'],case_id='positive'if positive else'negative',track='estimated_sun',positive=positive,
                    metrics=dict(correct=positive and mode=='sh'and profile['training_step']==1000,correspondence={'2':dict(correct=10)}if positive else{})))
        write(spec['summary'],dict(complete=True,purpose='checkpoint_selection',protocol_sha256=pipeline.sha(spec['protocol']),rows=rows,manifest_sha256='same-procedural-manifest'))
        write(Path(spec['summary']).parent/'COMPLETE.json',dict(complete=True,external_watchdog_passed=True,summary_sha256=pipeline.sha(spec['summary']),protocol_sha256=pipeline.sha(spec['protocol'])))
        write(spec['metric_replay'],dict(passed=True,summary_sha256=pipeline.sha(spec['summary']),replayed_rows=len(rows),replay_source_sha256=p['source_sha256']['verify_solar_evaluation_v70.py']))
        reserve(c['compute_ledger'],c['compute_ledger_sha256'],c['reservation_journal'],pipeline.sha(spec['protocol'])+'/evaluation','baseline',50)


def test_complete_fake_workflow_stops_for_publication_and_pose_selection_then_repeats_selected_mode(workspace,monkeypatch):
    w=workspace;result=advance_to_selection(w,monkeypatch)
    assert result['awaiting_development_selection']and len(w['calls'])==8
    c=pipeline.read(w['contract']);assert pipeline.remaining(c)['baseline']==66900
    # No fake PNG or actual tensor was ever created, opened or decoded.
    assert not list(Path('data').rglob('*.png'))
    create_selection(w)
    result=runner.run(w['contract'],'f'*40,executor=fake_executor(w))
    while result.get('awaiting_source_publication'):
        publish(w);result=runner.run(w['contract'],'f'*40,executor=fake_executor(w))
    assert result['learning_complete']and not result['scientific_goal_complete']and len(w['calls'])==10
    for stage in('seed_20261102','seed_20261103'):
        p=pipeline.read(pipeline.protocol_path(c,stage));assert p['seed']==int(stage[5:])and p['pooled']is False and p['fixed_reporting_step']==1000
    journal=pipeline.read(pipeline.CANONICAL_JOURNAL);training=[r for r in journal['reservations']if r['category']=='training']
    assert len(training)==10 and len({r['job_id']for r in training})==10
    assert pipeline.remaining(c)==dict(training=20220-sum(r['seconds']for r in training),baseline=66800)
    # Actual source validators accept the generated pilot and study CLI/schema.
    from solar_training_admission_v70 import validate_training_contract
    publish(w)
    for stage in('sh_pilot','pooled_pilot','sh_study','pooled_study','seed_20261102','seed_20261103'):
        assert validate_training_contract(pipeline.protocol_path(c,stage),'f'*40)['pipeline_stage']==stage


def test_geometry_gate_must_really_pass_and_no64pair_claim_after_shortfall(workspace):
    w=workspace;a=pipeline.read(w['settings']['geometry_audit']);a['passed']=False;write(w['settings']['geometry_audit'],a)
    with pytest.raises(ValueError,match='full96'):initialize(w)
    assert not Path(w['contract']).exists()


def test_independent_camera_exclusion_preserves96inventory_and_all_other_training_pairs(workspace):
    w=workspace;a=pipeline.read(w['settings']['geometry_audit'])
    removed=next(u for u in a['units']if u['split']=='training');removed['eligible']=False;removed['qualification_reasons']=['procedural cropped native support']
    a['oracle_pose_directions']=a['correct_oracle_poses']=190
    write(w['settings']['geometry_audit'],a)
    c=initialize(w)
    assert c['generated_training_unit_count']==64 and c['eligible_training_unit_count']==63
    assert len(c['excluded_training_units'])==1 and c['excluded_training_units'][0]['qualification_reasons']


def test_generated_full_teacher_protocol_passes_actual_preflight_before_model_construction(workspace,monkeypatch):
    w=workspace;initialize(w);ready=pipeline.prepare_next(w['contract']);publish(w);monkeypatch.setattr(pipeline.subprocess,'check_output',fake_git(w))
    # First fake pilot yields measured data used by the real producer preflight.
    fake_executor(w)(ready['protocol'],'f'*40,sys.executable,'build_solar_training_bank_v68.py')
    full=pipeline.prepare_next(w['contract']);p=pipeline.read(full['protocol'])
    p['output']='runs/preflight-only';p['manifest']='reports/preflight-only-manifest.json';path=write('reports/preflight-only.json',p);publish(w)
    import compact_lunar
    class ModelConstructionForbidden(RuntimeError):pass
    def no_model(*args,**kwargs):raise ModelConstructionForbidden('passed real metadata/source/cost gates; no model constructed')
    monkeypatch.setattr(compact_lunar,'load_checkpoint',no_model)
    from build_solar_training_bank_v68 import run as real_producer
    with pytest.raises(ModelConstructionForbidden,match='no model constructed'):
        real_producer(path,'f'*40)
    assert not Path(p['output']).exists()and not list(Path('data').rglob('*.png'))


def test_canonical_journal_cannot_be_replaced_to_restore_allowance(workspace):
    w=workspace;s=w['settings'];s['reservation_journal']='runs/another-journal.json';write(w['settings_path'],s)
    with pytest.raises(ValueError,match='canonical reservation journal'):initialize(w)


def test_original_canonical_charges_cannot_be_deleted_after_contract_freeze(workspace):
    w=workspace;c=initialize(w);j=pipeline.read(pipeline.CANONICAL_JOURNAL);j['reservations']=[];write(pipeline.CANONICAL_JOURNAL,j)
    with pytest.raises(ValueError,match='deleted, reordered or rewritten'):pipeline.remaining(c)


def test_source_hash_change_requires_a_new_explicit_contract_not_silent_reuse(workspace):
    w=workspace;initialize(w);Path('solar_response_v63.py').write_text('# PROCEDURAL CHANGED SOURCE\n')
    with pytest.raises(ValueError,match='source changed'):pipeline.prepare_next(w['contract'])


def test_source_publication_is_required_before_first_job(workspace,monkeypatch):
    w=workspace;initialize(w);monkeypatch.setattr(pipeline.subprocess,'check_output',fake_git(w))
    result=runner.run(w['contract'],'f'*40,executor=fake_executor(w))
    assert result['status']=='awaiting_source_publication'and not w['calls']
    assert len(pipeline.read(pipeline.CANONICAL_JOURNAL)['reservations'])==1


def test_failed_job_keeps_charge_and_cannot_restart(workspace,monkeypatch):
    w=workspace;initialize(w);pipeline.prepare_next(w['contract']);publish(w);monkeypatch.setattr(pipeline.subprocess,'check_output',fake_git(w))
    with pytest.raises(RuntimeError,match='procedural child failure'):
        runner.run(w['contract'],'f'*40,executor=fake_executor(w,fail=True))
    charge=pipeline.read(pipeline.CANONICAL_JOURNAL)['reservations'];assert len(charge)==2
    with pytest.raises(ValueError,match='active/partial|no automatic restart'):
        runner.run(w['contract'],'f'*40,executor=fake_executor(w))
    assert pipeline.read(pipeline.CANONICAL_JOURNAL)['reservations']==charge


def test_expired_published_deadline_never_resets(workspace,monkeypatch):
    w=workspace;c=initialize(w);ready=pipeline.prepare_next(w['contract']);before=Path(ready['protocol']).read_bytes();publish(w)
    p=pipeline.read(ready['protocol'])
    monkeypatch.setattr(pipeline.subprocess,'check_output',fake_git(w));monkeypatch.setattr(runner,'now',lambda:pipeline.utc(p['latest_start_utc'])+timedelta(seconds=1))
    with pytest.raises(TimeoutError,match='no reset'):runner.run(w['contract'],'f'*40,executor=fake_executor(w))
    assert not w['calls']and Path(ready['protocol']).read_bytes()==before


def test_publication_queue_is_separate_from_execution_cap_and_not_added_to_compute(workspace):
    w=workspace;c=initialize(w);ready=pipeline.prepare_next(w['contract']);p=pipeline.read(ready['protocol'])
    assert(pipeline.utc(p['latest_start_utc'])-pipeline.utc(p['created_utc'])).total_seconds()==1800
    assert(pipeline.utc(p['absolute_deadline_utc'])-pipeline.utc(p['latest_start_utc'])).total_seconds()==p['max_calendar_seconds']==100
    assert pipeline.remaining(c)==dict(training=20220,baseline=66900)
    pipeline.verify_protocol_lease(c,p)
    p['absolute_deadline_utc']=(pipeline.utc(p['absolute_deadline_utc'])+timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError,match='lease or execution deadline changed'):pipeline.verify_protocol_lease(c,p)


def test_real_watchdog_refuses_a_new_lease_protocol_after_latest_start_without_launching(workspace,monkeypatch):
    w=workspace;initialize(w);ready=pipeline.prepare_next(w['contract']);p=pipeline.read(ready['protocol'])
    # A real watchdog call with fake source publication stops before Popen.
    import importlib.util
    spec=importlib.util.spec_from_file_location('procedural_lease_watchdog',HERE/'run_solar_job_v70.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Clock:
        @staticmethod
        def now(zone):return pipeline.utc(p['latest_start_utc'])+timedelta(seconds=1)
        fromisoformat=staticmethod(datetime.fromisoformat)
    monkeypatch.setattr(module,'datetime',Clock)
    monkeypatch.setattr(module.subprocess,'check_output',lambda *args,**kwargs:Path(args[0][2].split(':',1)[1]).read_bytes())
    monkeypatch.setattr(module.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('Expired lease launched a scientific process'))
    with pytest.raises(TimeoutError,match='dispatch lease expired'):
        module.run(ready['protocol'],'f'*40,sys.executable,'build_solar_training_bank_v68.py')
    assert len(pipeline.read(pipeline.CANONICAL_JOURNAL)['reservations'])==1


def test_measured_full_study_forecast_includes_both_modes_and_two_seed_repetitions(workspace,monkeypatch):
    w=workspace;w['settings']['training_ceiling_seconds']=1000;write(w['settings_path'],w['settings'])
    with pytest.raises(ValueError,match='selected-seed repetitions'):
        advance_to_selection(w,monkeypatch,training_elapsed=5.)
    assert all(stage not in('sh_study','pooled_study')for stage,*_ in w['calls'])


def test_failed_exact_replay_watchdog_blocks_new_studies_even_with_passed_replay_json(workspace,monkeypatch):
    w=workspace;initialize(w);monkeypatch.setattr(pipeline.subprocess,'check_output',fake_git(w))
    result=runner.run(w['contract'],'f'*40,executor=fake_executor(w))
    while len(w['calls'])<6:
        publish(w);result=runner.run(w['contract'],'f'*40,executor=fake_executor(w))
    assert result['stage']=='sh_study'and len(w['calls'])==6
    c=pipeline.read(w['contract']);p=pipeline.read(pipeline.protocol_path(c,'sh_replay'));receipt=p['watchdog_receipts']['replay']
    r=pipeline.read(receipt);r['complete']=False;write(receipt,r)
    before=pipeline.read(pipeline.CANONICAL_JOURNAL)
    with pytest.raises(ValueError,match='external scientific watchdog'):
        runner.run(w['contract'],'f'*40,executor=fake_executor(w))
    assert len(w['calls'])==6 and pipeline.read(pipeline.CANONICAL_JOURNAL)==before


def test_selected_mode_cannot_be_chosen_from_different_case_sets(workspace,monkeypatch):
    w=workspace;advance_to_selection(w,monkeypatch);create_selection(w);c=pipeline.read(w['contract']);spec=c['selection_evidence']['pooled']
    summary=pipeline.read(spec['summary']);summary['manifest_sha256']='different-terrain';write(spec['summary'],summary)
    seal=Path(spec['summary']).parent/'COMPLETE.json';s=pipeline.read(seal);s['summary_sha256']=pipeline.sha(spec['summary']);write(seal,s)
    replay=pipeline.read(spec['metric_replay']);replay['summary_sha256']=pipeline.sha(spec['summary']);write(spec['metric_replay'],replay)
    with pytest.raises(ValueError,match='same complete cases'):pipeline.selection(c)


def test_cli_flags_and_api_names_match_actual_watchdog(workspace):
    w=workspace;c=initialize(w);ready=pipeline.prepare_next(w['contract'])
    from run_solar_job_v70 import run as real_api
    assert tuple(inspect.signature(real_api).parameters)==('protocol_path','commit','python','program','replay_root','replay_output')
    tree=ast.parse(Path('run_solar_job_v70.py').read_text());flags=set()
    for n in ast.walk(tree):
        if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='add_argument':
            flags.update(a.value for a in n.args if isinstance(a,ast.Constant)and isinstance(a.value,str)and a.value.startswith('--'))
    argv=runner.command_for_stage(ready['protocol'],'teacher_pilot','f'*40,sys.executable)
    assert {x for x in argv if x.startswith('--')}<=flags
    assert '--program'in argv and '--publication-commit'in argv and '--contract'not in argv
