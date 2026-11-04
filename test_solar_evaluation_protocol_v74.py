"""Metadata-only tests. All receipts/checkpoints below are TEMPORARY SYNTHETIC
unit-test fixtures, not actual learning or evaluation evidence. No model, image,
ray, reserve or scientific process is constructed/executed.
"""
import copy,json,sys
from types import SimpleNamespace
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pytest
import solar_evaluation_protocol_v74 as freeze

REPO=Path(__file__).resolve().parent
HERE=Path(__file__).resolve().parent
NOW=datetime(2026,11,4,12,tzinfo=timezone.utc)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return str(path)


def workspace(tmp_path,monkeypatch,excluded=(),pose_failure=None):
    # Read-only source view. Only metadata fixtures are written under tmp_path.
    for p in REPO.glob('*.py'):(tmp_path/p.name).symlink_to(p)
    for name in('solar_evaluation_protocol_v74.py','build_solar_evaluation_protocol_v74.py','run_solar_evaluation_v70.py'):
        p=tmp_path/name
        if p.exists():p.unlink()
        p.symlink_to(HERE/name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(freeze.runner,'software_identity',lambda:dict(scope='PROCEDURAL UNIT TEST; NO SCIENTIFIC EXECUTION'))
    monkeypatch.setattr(freeze.runner,'Backend',lambda *args:pytest.fail('Metadata builder constructed a model'))
    import test_solar_evaluation_v70 as old_tests
    manifest,audit=old_tests.procedural_manifest()
    for unit in audit['units']:
        if unit['unit_index']in excluded:
            unit['eligible']=False;unit['qualification_reasons']=['SYNTHETIC independent missing map coverage']
    audit['oracle_pose_directions']=2*sum(unit['eligible']for unit in audit['units'])
    if pose_failure is not None:
        next(unit for unit in audit['units']if unit['unit_index']==pose_failure)['oracle_outcomes']=[dict(pose_correct=False,scope='SYNTHETIC oracle failure; independent eligibility stays true')]
        audit['oracle_success']=(audit['oracle_pose_directions']-1)/audit['oracle_pose_directions']
    eligible=[unit['unit_index']for unit in audit['units']if unit['split']=='model_selection'and unit['eligible']]
    manifest['cases']=[case for case in manifest['cases']if case['unit_index']in eligible]
    for case in manifest['cases']:
        if not case['positive']and case['query_unit_index']not in eligible:
            endpoint=next(unit for unit in audit['units']if unit['unit_index']in eligible and unit['component']!=case['component'])
            case['query_unit_index']=endpoint['unit_index'];case['query_component']=endpoint['component']
            group=int(endpoint['component'].replace('select',''));case['nonoverlap_certificate']['query_bounds']=[group*1000,0,group*1000+100,100]
    audit['audit_source_sha256']=freeze.sha('verify_solar_development_v68.py');write('metadata/audit.json',audit)
    prep={}
    for ui in eligible:
        geometry=f'assets/reference-{ui}.npz';receipt=f'metadata/prep-{ui}.json'
        write(receipt,dict(complete=True,calendar_seconds=.25,scope='SYNTHETIC UNIT FIXTURE'))
        prep[geometry]=dict(receipt_path=receipt,sha256=freeze.sha(receipt),calendar_seconds=.25)
    for c in manifest['cases']:
        ui,qi=c['unit_index'],c['query_unit_index'];c.update(reference=f'assets/ref-{ui}-{c["light_a"]}.png',query=f'assets/query-{qi}-{c["light_b"]}.png',reference_geometry=f'assets/reference-{ui}.npz',solar_lookup_protocol=f'assets/lookup-{ui}.json')
        paths=[c[k]for k in('reference','query','reference_geometry','solar_lookup_protocol')]
        if c['positive']:
            token=c['case_id'].replace('/','-');c.update(scoring_labels=f'assets/labels-{token}.npz',scoring_geometry=f'assets/geometry-{token}.npz');paths += [c['scoring_labels'],c['scoring_geometry']]
        c['input_sha256']={p:'f'*64 for p in paths}
    manifest.update(complete=True,eligible_unit_indices=eligible,geometry_audit_sha256=freeze.sha('metadata/audit.json'),geometry_audit='metadata/audit.json',reference_precompute_receipts=prep)
    write('metadata/manifest.json',manifest)
    physical_protocol=write('metadata/reference-preparation-protocol.json',dict(scope='SYNTHETIC UNIT TEST, NOT EXECUTED'))
    physical_summary=write('metadata/reference-preparation-summary.json',dict(scope='SYNTHETIC UNIT TEST, NOT EXECUTED'))
    physical=dict(pinned_inputs={physical_protocol:freeze.sha(physical_protocol),physical_summary:freeze.sha(physical_summary)},outer_calendar_seconds_by_archive={p:.5 for p in prep},internal_calendar_seconds_by_archive={p:.25 for p in prep},reference_archives=list(prep),reference_unit_indices=eligible,baseline_reserved_seconds=100,phase_calendar_seconds=dict(reference_one=.5,reference_four=2.,reference_full=float(len(eligible))),scope='SYNTHETIC UNIT TEST DEPENDENCY')
    physical.update(freeze.selection_inventory(audit))
    physical_module=tmp_path/'solar_reference_pipeline_v74.py'
    if physical_module.exists():physical_module.unlink()
    physical_module.write_text('# Temporary unit dependency identity, no physical code executed.\n')
    monkeypatch.setitem(sys.modules,'solar_reference_pipeline_v74',SimpleNamespace(completed_reference_preparation=lambda protocol,summary:copy.deepcopy(physical)))
    for name in('master','dataset','lights'):write(f'metadata/{name}.json',dict(scope='SYNTHETIC UNIT FIXTURE'))
    write('metadata/ledger.json',dict(accounting_complete=True,remaining_training_wall_seconds=20220,remaining_baseline_wall_seconds=67200))
    journal=dict(ledger_sha256=freeze.sha('metadata/ledger.json'),reservations=[]);write(freeze.learning.CANONICAL_JOURNAL,journal)
    c=dict(schema_version=73,prepared_only=False,calibration_or_final_access=False,training_ceiling_seconds=20220,baseline_ceiling_seconds=66900,initial_reservations=[],initial_charged=dict(training=0,baseline=0),source_sha256={'run_solar_job_v70.py':freeze.sha('run_solar_job_v70.py')},reservation_journal=freeze.learning.CANONICAL_JOURNAL,compute_ledger='metadata/ledger.json',compute_ledger_sha256=freeze.sha('metadata/ledger.json'),protocol_directory='reports/learning',run_directory='runs/learning',selection_evidence={mode:dict(protocol=f'reports/{mode}-selection.json',summary=f'runs/{mode}-selection/summary.json',metric_replay=f'reports/{mode}-selection-replay.json')for mode in('sh','pooled')})
    for key,name in [('original_master','master'),('dataset_summary','dataset'),('geometry_audit','audit'),('lighting_schedule','lights')]:c[key]=f'metadata/{name}.json';c[key+'_sha256']=freeze.sha(c[key])
    for mode in('sh','pooled'):
        run=f'runs/learning/{mode}_study';Path(run).mkdir(parents=True);receipt=f'metadata/{mode}-watchdog.json'
        protocol=dict(updates=2000,seed=20261101,pooled=mode=='pooled',output=run,max_calendar_seconds=60,absolute_deadline_utc=(NOW+timedelta(seconds=120)).isoformat(),latest_start_utc=(NOW+timedelta(seconds=60)).isoformat(),watchdog_receipts=dict(run=receipt),source_sha256=c['source_sha256'])
        path=f'reports/learning/{mode}_study.json';write(path,protocol)
        checkpoints={}
        for step in(0,100,500,1000,1500,2000):
            cp=Path(run)/f'step_{step}.pt';cp.write_text('SYNTHETIC UNIT CHECKPOINT IDENTITY; NOT A MODEL:'+str(step));checkpoints[cp.name]=freeze.sha(cp)
        history=write(Path(run)/'history_2000.json',dict(scope='SYNTHETIC UNIT FIXTURE'))
        write(Path(run)/'summary.json',dict(complete=True,updates=2000,seed=20261101,pooled=mode=='pooled',protocol_sha256=freeze.sha(path),checkpoint_sha256=checkpoints,all_history_sha256=freeze.sha(history),calendar_seconds=5.))
        write(receipt,dict(complete=True,mode='run',program='train_solar_response_v63.py',child_returncode=0,failure=None,protocol_sha256=freeze.sha(path),source_sha256=c['source_sha256']['run_solar_job_v70.py'],started_utc=NOW.isoformat(),absolute_deadline_utc=(NOW+timedelta(seconds=60)).isoformat(),cap_seconds=60,calendar_seconds=6.))
        journal['reservations'].append(dict(job_id=freeze.sha(path)+'/train',category='training',seconds=60,reserved_utc=NOW.isoformat(),refund_on_failure=False))
    write(freeze.learning.CANONICAL_JOURNAL,journal);write('metadata/learning-contract.json',c)
    pilot_positives=[]
    for case in manifest['cases']:
        if case['positive']and(case['light_a'],case['light_b'])==(0,0)and all(case['component']!=prior['component']for prior in pilot_positives):pilot_positives.append(case)
        if len(pilot_positives)==2:break
    pilot_ids=[key for case in pilot_positives for key in(case['case_id'],case['case_id']+'/negative')]
    s=dict(prepared_only=False,mode='checkpoint_selection_sh',calibration_or_final_access=False,repository_root='.',learning_contract='metadata/learning-contract.json',manifest='metadata/manifest.json',geometry_audit='metadata/audit.json',tracks=['estimated_sun'],inference_configuration=copy.deepcopy(freeze.DEFAULT_CONFIG),pilot_case_ids=pilot_ids,pilot_cap_seconds=600,maximum_full_calendar_seconds=10000,publication_dispatch_lease_seconds=1800,output_directory='runs/evaluation-sh',blender_executable='/NOT_EXECUTED/Blender',cost_pilot_evidence={},reference_preparation_evidence=dict(protocol=physical_protocol,summary=physical_summary))
    write('metadata/settings.json',s)
    return s,c,manifest,journal


def completed_cost_pilot(s,c,m,journal,monkeypatch):
    q=freeze.prepare('metadata/settings.json','cost_pilot','reports/eval-pilot.json',NOW)
    root=Path(q['output']);root.mkdir(parents=True);by={case['case_id']:case for case in m['cases']};rows=[]
    for profile in q['profiles']:
        for track in q['tracks']:
            for case_id in q['pilot_case_ids']:
                case=by[case_id];rows.append(dict({key:case[key]for key in('case_id','component','query_component','positive','unit_index','query_unit_index','light_a','light_b')},profile=profile['name'],track=track,inference_failure=None,case_calendar_seconds=.1,profile_load_seconds=.01,reused_case_cache=False))
    summary=dict(complete=True,purpose='cost_pilot',protocol_sha256=freeze.sha('reports/eval-pilot.json'),manifest_sha256=freeze.sha(s['manifest']),software=freeze.runner.software_identity(),comparison_identity=freeze.runner.comparison_identity(q,m),profiles=q['profiles'],case_ids=q['pilot_case_ids'],fresh_inference_cases=len(rows),cached_cases=0,rows=rows,calendar_seconds=5.,source_publication_commit='a'*40)
    write(root/'summary.json',summary);write(root/'COMPLETE.json',dict(complete=True,external_watchdog_passed=True,summary_sha256=freeze.sha(root/'summary.json'),protocol_sha256=freeze.sha('reports/eval-pilot.json'),observed_calendar_seconds=6.,original_allowance_seconds=600))
    entry=dict(job_id=freeze.sha('reports/eval-pilot.json')+'/evaluation',category='baseline',seconds=600,reserved_utc=NOW.isoformat(),refund_on_failure=False);journal['reservations'].append(entry);write(freeze.learning.CANONICAL_JOURNAL,journal)
    write(root/'reservation.json',dict(reservation=entry));write(root/'watchdog_admission.json',dict(protocol_sha256=freeze.sha('reports/eval-pilot.json'),reservation_sha256=freeze.sha(root/'reservation.json'),started_utc=NOW.isoformat(),absolute_deadline_utc=(NOW+timedelta(seconds=600)).isoformat()))
    write(root/'execution.json',dict(protocol_sha256=freeze.sha('reports/eval-pilot.json'),publication_commit='a'*40,software=summary['software'],final_reserve_access=False))
    replay='metadata/pilot-replay.json';write(replay,dict(passed=True,summary_sha256=freeze.sha(root/'summary.json'),replayed_rows=len(rows),replay_source_sha256=q['source_sha256']['verify_solar_evaluation_v70.py']))
    s['cost_pilot_evidence']=dict(protocol='reports/eval-pilot.json',summary=str(root/'summary.json'),metric_replay=replay);write('metadata/settings.json',s)
    monkeypatch.setattr(freeze.subprocess,'check_output',lambda argv:Path(argv[2].split(':',1)[1]).read_bytes())
    return q,summary


def test_prepared_template_cannot_become_an_executable_claim(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);write('template.json',dict(prepared_only=True))
    with pytest.raises(ValueError,match='actual qualified'):freeze.prepare('template.json','cost_pilot','reports/out.json',NOW)
    assert not Path('reports/out.json').exists()


def test_checkpoint_builder_keeps_all384cases_five_steps_and_never_opens_images(tmp_path,monkeypatch):
    s,c,m,journal=workspace(tmp_path,monkeypatch)
    p=freeze.prepare('metadata/settings.json','cost_pilot','reports/eval-pilot.json',NOW)
    assert len(p['case_ids'])==384 and len(set(p['case_ids']))==384
    assert [x['training_step']for x in p['profiles']]==[0,500,1000,1500,2000]
    assert p['tracks']==['estimated_sun']and not p['calibration_or_final_access']and p['case_cache_seals']=={}
    assert not Path('assets').exists()and not Path(p['output']).exists()and p['source_publication_required']
    assert p['reservation_journal']==freeze.learning.CANONICAL_JOURNAL


def test_incomplete_geometry_or_case_denominator_blocks_freeze(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch);m['cases']=m['cases'][2:];write(s['manifest'],m)
    with pytest.raises(ValueError,match='All eligible cameras'):freeze.prepare('metadata/settings.json','cost_pilot','reports/eval.json',NOW)
    assert not Path('reports/eval.json').exists()


def test_missing_or_failed_actual_study_watchdog_cannot_select_checkpoints(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch);Path('metadata/sh-watchdog.json').unlink()
    with pytest.raises(ValueError,match='active/partial'):freeze.prepare('metadata/settings.json','cost_pilot','reports/eval.json',NOW)


def test_measured_full_protocol_requires_sealed_replayed_charged_pilot(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch);q,summary=completed_cost_pilot(s,c,m,j,monkeypatch)
    p=freeze.prepare('metadata/settings.json','full','reports/eval-full.json',NOW+timedelta(minutes=1))
    assert p['purpose']=='checkpoint_selection'and len(p['case_ids'])==384 and p['profiles']==q['profiles']
    assert p['max_calendar_seconds']==289 and p['cost_pilot_summary_sha256']==freeze.sha(s['cost_pilot_evidence']['summary'])
    assert s['cost_pilot_evidence']['metric_replay']in p['input_sha256']and str(Path(q['output'])/'execution.json')in p['input_sha256']
    assert not Path(p['output']).exists()


@pytest.mark.parametrize('defect',['failed','late','unreplayed','duplicate','cached','wrong_endpoint'])
def test_bad_pilot_evidence_does_not_generate_full_study(tmp_path,monkeypatch,defect):
    s,c,m,j=workspace(tmp_path,monkeypatch);q,summary=completed_cost_pilot(s,c,m,j,monkeypatch);root=Path(q['output'])
    if defect=='failed':write(root/'FAILED.json',dict(scope='SYNTHETIC FAILURE FIXTURE'))
    elif defect=='late':
        seal=freeze.read(root/'COMPLETE.json');seal['observed_calendar_seconds']=601;write(root/'COMPLETE.json',seal)
    elif defect=='unreplayed':
        replay=freeze.read(s['cost_pilot_evidence']['metric_replay']);replay['passed']=False;write(s['cost_pilot_evidence']['metric_replay'],replay)
    else:
        if defect=='duplicate':summary['rows'][1]=copy.deepcopy(summary['rows'][0])
        elif defect=='cached':summary['rows'][0]['reused_case_cache']=True
        else:summary['rows'][0]['query_component']='wrong'
        write(root/'summary.json',summary)
        seal=freeze.read(root/'COMPLETE.json');seal['summary_sha256']=freeze.sha(root/'summary.json');write(root/'COMPLETE.json',seal)
        replay=freeze.read(s['cost_pilot_evidence']['metric_replay']);replay['summary_sha256']=freeze.sha(root/'summary.json');write(s['cost_pilot_evidence']['metric_replay'],replay)
    with pytest.raises(ValueError):freeze.prepare('metadata/settings.json','full','reports/full.json',NOW)
    assert not Path('reports/full.json').exists()


def test_expired_dispatch_and_alternate_journal_are_rejected(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch);p=freeze.prepare('metadata/settings.json','cost_pilot','reports/pilot.json',NOW)
    with pytest.raises(ValueError,match='lease expired'):freeze.validate_runtime(p,'reports/pilot.json',NOW+timedelta(seconds=1801))
    p['reservation_journal']='runs/alternate-budget.json'
    with pytest.raises(ValueError,match='Canonical'):freeze.validate_runtime(p,'reports/pilot.json',NOW)


def test_existing_partial_output_is_never_overwritten_or_restarted(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch);Path(s['output_directory']+'/cost_pilot').mkdir(parents=True)
    with pytest.raises(FileExistsError,match='Preserve every'):freeze.prepare('metadata/settings.json','cost_pilot','reports/pilot.json',NOW)


def test_caps_cannot_spend_more_than_remaining_original_budget(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch);s['maximum_full_calendar_seconds']=66900;write('metadata/settings.json',s)
    with pytest.raises(ValueError,match='remaining canonical'):freeze.prepare('metadata/settings.json','cost_pilot','reports/pilot.json',NOW)


def test_source_publication_check_only_reads_and_never_executes(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch);p=freeze.prepare('metadata/settings.json','cost_pilot','reports/pilot.json',NOW)
    monkeypatch.setattr(freeze,'validate_runtime',lambda *args:dict())
    monkeypatch.setattr(freeze.subprocess,'check_output',lambda argv:b'wrong byte identity')
    with pytest.raises(ValueError,match='publication barrier'):freeze.verify_publication('reports/pilot.json','a'*40)
    monkeypatch.setattr(freeze.subprocess,'check_output',lambda argv:Path(argv[2].split(':',1)[1]).read_bytes())
    value=freeze.verify_publication('reports/pilot.json','a'*40)
    assert value['scientific_job_started']is False and not Path(p['output']).exists()


def completed_selections(s,c,m):
    """Synthetic negative-result fixtures: every checkpoint has zero successes.
    They exercise complete-row ranking and linked evidence, not scientific data.
    """
    for mode in('sh','pooled'):
        profiles,stage=freeze._study_profiles(c,mode)
        for p in profiles:p['configuration'].update(copy.deepcopy(freeze.DEFAULT_CONFIG))
        spec=c['selection_evidence'][mode];root=Path(spec['summary']).parent;root.mkdir(parents=True)
        protocol=dict(split='model_selection',purpose='checkpoint_selection',tracks=['estimated_sun'],calibration_or_final_access=False,profiles=profiles,case_ids=[case['case_id']for case in m['cases']],source_sha256=freeze.source_closure('.'))
        write(spec['protocol'],protocol);rows=[]
        for profile in profiles:
            for case in m['cases']:
                rows.append(dict(profile=profile['name'],case_id=case['case_id'],track='estimated_sun',positive=case['positive'],metrics=dict(correct=False,correspondence={'2':dict(correct=0)})))
        summary=dict(complete=True,purpose='checkpoint_selection',protocol_sha256=freeze.sha(spec['protocol']),manifest_sha256=freeze.sha(s['manifest']),calibration_or_final_access=False,rows=rows)
        write(spec['summary'],summary)
        write(root/'COMPLETE.json',dict(complete=True,external_watchdog_passed=True,protocol_sha256=freeze.sha(spec['protocol']),summary_sha256=freeze.sha(spec['summary'])))
        write(spec['metric_replay'],dict(passed=True,summary_sha256=freeze.sha(spec['summary']),replayed_rows=len(rows),replay_source_sha256=protocol['source_sha256']['verify_solar_evaluation_v70.py']))


def test_principal_matrix_reuses_actual_pose_selected_weights_and_explicit_pooled_study(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch);completed_selections(s,c,m)
    s['mode']='principal_ablations';s['tracks']=['estimated_sun','known_sun'];s['output_directory']='runs/principal';write('metadata/settings.json',s)
    p=freeze.prepare('metadata/settings.json','cost_pilot','reports/principal-pilot.json',NOW)
    assert len(p['profiles'])==9 and len(p['case_ids'])==384
    by={profile['name']:profile for profile in p['profiles']};base=by['A1B1C1']
    assert all(by[name]['checkpoint']==base['checkpoint']for name in by if name!='pooled_A1B1C1')
    assert all(profile['training_step']==0 for profile in p['profiles']) # honestly selected initialization
    assert by['pooled_A1B1C1']['configuration']['pooled']and not base['configuration']['pooled']
    assert by['pooled_A1B1C1']['training_evidence']['summary']!=base['training_evidence']['summary']
    freeze.evaluation.validate_profiles(p['profiles'],'principal_ablations')


@pytest.mark.parametrize('mode',['additional_controls_sh','additional_controls_pooled'])
def test_additional_controls_preserve_declared_budgets_policies_and_selected_mode(tmp_path,monkeypatch,mode):
    s,c,m,j=workspace(tmp_path,monkeypatch);completed_selections(s,c,m);s['mode']=mode;write('metadata/settings.json',s)
    p=freeze.prepare('metadata/settings.json','cost_pilot','reports/additional-pilot.json',NOW)
    assert len(p['profiles'])==13 and p['profiles'][0]['configuration']['solar_mode']=='independent'
    assert {(profile['configuration']['budget'],profile['configuration']['policy'])for profile in p['profiles'][1:]}=={(budget,policy)for budget in(32,64,128)for policy in('active','uniform','appearance','geometry')}
    assert all(profile['configuration']['pooled']==mode.endswith('_pooled')for profile in p['profiles'])
    assert len({profile['checkpoint']for profile in p['profiles']})==1


def test_post_selection_pose_policy_change_requires_new_selection_evidence(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch);completed_selections(s,c,m);s['mode']='principal_ablations'
    s['inference_configuration'].update(pose_initialization='quantization_aware',inference_revision='solar-v73-quantized-init');write('metadata/settings.json',s)
    with pytest.raises(ValueError,match='after checkpoint selection'):freeze.prepare('metadata/settings.json','cost_pilot','reports/principal.json',NOW)


def test_physical_preparation_costs_include_external_startup_and_are_pinned(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch);p=freeze.prepare('metadata/settings.json','cost_pilot','reports/pilot.json',NOW)
    costs=p['reference_preparation_costs'];assert len(costs['outer_calendar_seconds_by_archive'])==32
    assert all(costs['outer_calendar_seconds_by_archive'][archive]>.25 for archive in costs['outer_calendar_seconds_by_archive'])
    for path in s['reference_preparation_evidence'].values():assert p['input_sha256'][path]==freeze.sha(path)
    module=sys.modules['solar_reference_pipeline_v74'];original=module.completed_reference_preparation
    def wrong(protocol,summary):
        value=original(protocol,summary);value['reference_archives']=value['reference_archives'][:-1];return value
    monkeypatch.setattr(module,'completed_reference_preparation',wrong)
    with pytest.raises(ValueError,match='exact eligible'):freeze.prepare('metadata/settings.json','cost_pilot','reports/other.json',NOW)


def test_registered_worker_reservation_does_not_get_charged_twice_by_metadata_admission(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch);p=freeze.prepare('metadata/settings.json','cost_pilot','reports/pilot.json',NOW)
    root=Path(p['output']);root.mkdir(parents=True)
    j['reservations'].append(dict(job_id=freeze.sha('reports/pilot.json')+'/evaluation',category='baseline',seconds=p['max_calendar_seconds'],reserved_utc=NOW.isoformat(),refund_on_failure=False));write(freeze.learning.CANONICAL_JOURNAL,j)
    write(root/'watchdog_admission.json',dict(protocol_sha256=freeze.sha('reports/pilot.json'),started_utc=NOW.isoformat(),absolute_deadline_utc=(NOW+timedelta(seconds=600)).isoformat()))
    value=freeze.validate_runtime(p,'reports/pilot.json',NOW+timedelta(seconds=20))
    assert value['already_reserved']and len(freeze.read(freeze.learning.CANONICAL_JOURNAL)['reservations'])==3
    with pytest.raises(ValueError,match='deadline has expired'):freeze.validate_runtime(p,'reports/pilot.json',NOW+timedelta(seconds=601))


def test_legacy_schema_cannot_start_a_new_scientific_evaluation(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);write('legacy.json',dict(prepared_only=False,schema_version=70))
    with pytest.raises(ValueError,match='schema74'):freeze.runner.admission('legacy.json','not-used')


def test_outer_setup_cost_cannot_be_replaced_with_internal_sidecar_time(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch);module=sys.modules['solar_reference_pipeline_v74'];original=module.completed_reference_preparation
    def wrong(protocol,summary):
        result=original(protocol,summary);first=result['reference_archives'][0];result['outer_calendar_seconds_by_archive'][first]=.1;return result
    monkeypatch.setattr(module,'completed_reference_preparation',wrong)
    with pytest.raises(ValueError,match='omit cold startup'):freeze.prepare('metadata/settings.json','cost_pilot','reports/pilot.json',NOW)


def test_completed_pilot_replays_after_lease_with_less_current_budget_than_its_old_cap(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch);q,summary=completed_cost_pilot(s,c,m,j,monkeypatch);root=Path(q['output'])
    # The actual start meets the launch lease, while its successful6second
    # completion is AFTER latest_start and BEFORE its execution deadline.
    started=freeze.utc(q['latest_start_utc'])-timedelta(seconds=1)
    watchdog=freeze.read(root/'watchdog_admission.json');watchdog.update(started_utc=started.isoformat(),absolute_deadline_utc=(started+timedelta(seconds=600)).isoformat());write(root/'watchdog_admission.json',watchdog)
    # Other fixed-cap work is already spent. Only300seconds remain now, less
    # than this historical pilot's600seconds but enough for the289second study.
    j['reservations'].append(dict(job_id='OTHER_SYNTHETIC_PRIOR_BASELINE_CHARGE',category='baseline',seconds=66000,reserved_utc=started.isoformat(),refund_on_failure=False));write(freeze.learning.CANONICAL_JOURNAL,j)
    p=freeze.prepare('metadata/settings.json','full','reports/full-month-later.json',NOW+timedelta(days=31))
    assert p['max_calendar_seconds']==289 and p['baseline_admission']['remaining_at_preparation_seconds']==300
    assert freeze.utc(p['created_utc'])>freeze.utc(q['absolute_deadline_utc'])
    assert freeze.validate_runtime(p,'reports/full-month-later.json',freeze.utc(p['created_utc'])+timedelta(seconds=1))['already_reserved']is False


def test_31_eligible_cameras_freeze_all372cases_and_retain_pose_failure(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch,excluded=(95,),pose_failure=94)
    p=freeze.prepare('metadata/settings.json','cost_pilot','reports/pilot-31.json',NOW)
    assert len(p['case_ids'])==372 and p['generated_selection_unit_count']==32 and p['eligible_selection_unit_count']==31
    assert [unit['unit_index']for unit in p['excluded_selection_units']]==[95]
    assert len(p['terrain_groups'])==8 and p['zero_eligible_terrain_groups']==[]
    assert len([case for case in m['cases']if case['unit_index']==94])==12
    assert next(unit for unit in p['generated_selection_units']if unit['unit_index']==94)['eligible']is True
    assert len(p['reference_preparation_costs']['outer_calendar_seconds_by_archive'])==31


def test_zero_eligible_group_remains_in_raw_inventory_without_invented_cases(tmp_path,monkeypatch):
    s,c,m,j=workspace(tmp_path,monkeypatch,excluded=(64,65,66,67))
    p=freeze.prepare('metadata/settings.json','cost_pilot','reports/pilot-empty-group.json',NOW)
    assert len(p['case_ids'])==336 and len(p['generated_selection_units'])==32
    assert p['zero_eligible_terrain_groups']==['select0']and len(p['eligible_terrain_groups'])==7 and len(p['terrain_groups'])==8
    assert len(p['excluded_selection_units'])==4
    assert not any(case['component']=='select0'or case['query_component']=='select0'for case in m['cases'])
