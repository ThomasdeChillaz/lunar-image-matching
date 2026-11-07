"""Artifact-driven, source-publication-gated lunar learning orchestration.

This module prepares protocols and checks receipts; it never decodes images,
constructs a model, runs rays, publishes source, or accesses final/calibration
data. The existing bounded scientific entrypoints perform admitted jobs.
"""
import ast
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import subprocess

CANONICAL_JOURNAL = 'runs/solar-compute-reservations-v67.json'
MAX_TRAINING = 20220
MAX_BASELINE = 66900
STAGES = ('teacher_pilot','teacher_full','sh_pilot','sh_replay','pooled_pilot','pooled_replay',
          'sh_study','pooled_study','seed_20261102','seed_20261103')
PIPELINE_SOURCES = ('solar_learning_pipeline_v73.py','build_solar_learning_protocol_v73.py','run_solar_learning_pipeline_v73.py')
TEACHER_SOURCES = ('build_solar_training_bank_v68.py','solar_training_bank_v68.py','solar_native_v63.py',
    'solar_physics_v63.py','solar_frontend_v63.py','lunar_geometry_v5.py','lunar_geometry_v2.py',
    'compact_lunar.py','contrastive_terrain.py','north_render_plan_v31.py','north_terrain_pilot_v32.py',
    'solar_training_v63.py','solar_compute_budget_v67.py','verify_solar_development_v68.py')
TRAINING_SOURCES = ('train_solar_response_v63.py','solar_training_admission_v70.py','run_solar_job_v70.py',
    'solar_training_v63.py','solar_response_v63.py','compact_lunar.py','contrastive_terrain.py',
    'solar_compute_budget_v67.py','dataset.py','matchers.py','geometry.py')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc)
def utc(value):
    result=datetime.fromisoformat(value)
    if result.tzinfo is None: raise ValueError('Timezone-aware original deadlines required')
    return result
def dump_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def pinned(path,digest):
    if sha(path)!=digest: raise ValueError('Frozen artifact changed: '+str(path))
    return read(path)
def integer(value,maximum=None):
    if type(value)is not int or value<=0 or maximum is not None and value>maximum:
        raise ValueError('Positive explicit bounded integer allowance required')
    return value


def local_source_closure():
    """Module-level import closure plus explicitly executed lazy entrypoints."""
    seeds=set(TEACHER_SOURCES+TRAINING_SOURCES+PIPELINE_SOURCES+('solar_evaluation_v70.py',))
    def imports(node):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): return
        if isinstance(node,ast.Import):
            for name in node.names: yield name.name.split('.')[0]+'.py'
        if isinstance(node,ast.ImportFrom)and node.module: yield node.module.split('.')[0]+'.py'
        for child in ast.iter_child_nodes(node): yield from imports(child)
    todo=list(seeds);seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        path=Path(name)
        if not path.is_file():raise ValueError('Learning implementation not integrated: '+name)
        seen.add(name)
        todo.extend(n for n in imports(ast.parse(path.read_text()))if Path(n).is_file()and n not in seen)
    return {name:sha(name)for name in sorted(seen)}


def journal_state(ledger_path,ledger_sha,journal_path):
    if str(journal_path)!=CANONICAL_JOURNAL or Path(journal_path).resolve()!=Path(CANONICAL_JOURNAL).absolute():
        raise ValueError('The one canonical reservation journal is mandatory; aliases/symlinks cannot create free budget')
    ledger=pinned(ledger_path,ledger_sha)
    if not ledger['accounting_complete']:raise ValueError('Historical cumulative compute must be reconstructed')
    journal=read(journal_path)if Path(journal_path).exists()else dict(ledger_sha256=ledger_sha,reservations=[])
    if journal['ledger_sha256']!=ledger_sha:raise ValueError('Canonical journal belongs to another ledger')
    ids=set();charged=dict(training=0,baseline=0)
    for r in journal['reservations']:
        if r['job_id']in ids or r['category']not in charged or r.get('refund_on_failure')is not False:
            raise ValueError('Canonical journal has duplicate, invalid or refunded charges')
        integer(r['seconds']);ids.add(r['job_id']);charged[r['category']]+=r['seconds']
    return ledger,journal,charged


def remaining(contract):
    ledger,journal,charged=journal_state(contract['compute_ledger'],contract['compute_ledger_sha256'],contract['reservation_journal'])
    initial=contract['initial_reservations']
    if journal['reservations'][:len(initial)]!=initial:
        raise ValueError('Canonical reservation history was deleted, reordered or rewritten')
    result={}
    for category in ('training','baseline'):
        result[category]=min(ledger['remaining_'+category+'_wall_seconds']-charged[category],
            contract[category+'_ceiling_seconds']-(charged[category]-contract['initial_charged'][category]))
        if result[category]<0:raise ValueError('Original cumulative '+category+' allowance exhausted')
    return result


def full_geometry_gate(master,dataset,audit):
    # Existing verifier reads metadata/seals only, never an image or model.
    from solar_training_bank_v68 import verify_geometry_gate
    gate=verify_geometry_gate(master,dataset,audit)
    if gate.audit['audit_source_sha256']!=sha('verify_solar_development_v68.py'):
        raise ValueError('Passed geometry audit source no longer matches the implementation')
    return gate


def initialize(settings_path,contract_path):
    s=read(settings_path)
    if s.get('prepared_only',False):raise ValueError('Prepared settings cannot admit learning')
    if s['calibration_or_final_access']is not False:raise ValueError('Development-only learning contract required')
    integer(s['training_ceiling_seconds'],MAX_TRAINING);integer(s['baseline_ceiling_seconds'],MAX_BASELINE)
    gate=full_geometry_gate(s['original_master'],s['dataset_summary'],s['geometry_audit'])
    # Keep all independently eligible cameras. The verifier preserves all four
    # generated cameras/group and forbids eligibility based on pose success.
    integer(s['publication_dispatch_lease_seconds'])
    integer(s['teacher_pilot_cap_seconds'],3600);integer(s['training_pilot_cap_seconds']);integer(s['training_replay_cap_seconds'],900)
    if type(s['maximum_points_per_unit'])is not int or not 8<=s['maximum_points_per_unit']<=256:
        raise ValueError('Predeclared 8..256 point teacher budget required')
    integer(s['teacher_batch_size'],128)
    jitter=s['trial_pose_jitter']
    if set(jitter)!={'translation_sigma_m','rotation_sigma_deg'}or any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or v<0 for v in jitter.values()):
        raise ValueError('Explicit finite nonnegative physical trial-pose jitter required')
    if sha(s['lighting_schedule'])!=gate.master['source_schedule_sha256']:
        raise ValueError('Original physical lighting schedule differs')
    ledger_sha=sha(s['compute_ledger']);ledger,journal,charged=journal_state(s['compute_ledger'],ledger_sha,s['reservation_journal'])
    teacher=Path(s['initializer']);complete=read(teacher/'COMPLETE.json')
    if sha(teacher/'selected.pt')!=complete['checkpoint_sha256']or sha(teacher/'training.json')!=complete['training_sha256']:
        raise ValueError('Frozen original initializer artifacts changed')
    if len(s['expected_frozen_descriptor_sha256'])!=64:raise ValueError('Preexisting frozen descriptor identity required')
    source=local_source_closure()
    result=dict(s,schema_version=73,settings_path=str(settings_path),settings_sha256=sha(settings_path),source_sha256=source,
        original_master_sha256=sha(s['original_master']),dataset_summary_sha256=sha(s['dataset_summary']),geometry_audit_sha256=sha(s['geometry_audit']),
        lighting_schedule_sha256=sha(s['lighting_schedule']),compute_ledger_sha256=ledger_sha,
        initializer_sha256={name:sha(teacher/name)for name in('selected.pt','training.json','COMPLETE.json')},
        initial_reservations=journal['reservations'],initial_charged=charged,
        training_groups=list(gate.training_groups),eligible_training_unit_indices=list(gate.eligible_training_indices),
        training_unit_components={str(i):gate.master['units'][i]['site']['component']for i in gate.eligible_training_indices},
        generated_training_unit_count=64,eligible_training_unit_count=len(gate.eligible_training_indices),
        excluded_training_units=[u for u in gate.audit['units']if u['split']=='training'and not u['eligible']],
        geometry_jitter_absent=not any(jitter.values()),published_protocols_required_before_each_invocation=True,
        scope='Training bank and bounded development learning only; no automatic publication, model promotion, calibration, reserve, email or submission.')
    remaining(result);dump_new(contract_path,result);return result


def load_contract(path):
    c=read(path)
    if c['schema_version']!=73 or c.get('prepared_only',False)or c['calibration_or_final_access']is not False:
        raise ValueError('Initialized development-only learning contract required')
    for name in ('original_master','dataset_summary','geometry_audit','lighting_schedule','compute_ledger'):
        pinned(c[name],c[name+'_sha256'])
    for name,h in c['source_sha256'].items():
        if sha(name)!=h:raise ValueError('Learning source changed after its contract freeze: '+name)
    if str(c['reservation_journal'])!=CANONICAL_JOURNAL:raise ValueError('Canonical budget journal changed')
    integer(c['training_ceiling_seconds'],MAX_TRAINING);integer(c['baseline_ceiling_seconds'],MAX_BASELINE)
    remaining(c);return c


def protocol_path(c,stage):
    owner=stage.replace('_replay','_pilot')if stage in('sh_replay','pooled_replay')else stage
    return str(Path(c['protocol_directory'])/(owner+'.json'))


def stage_output(c,stage):
    owner=stage.replace('_replay','_pilot')if stage in('sh_replay','pooled_replay')else stage
    return str(Path(c['run_directory'])/owner)


def watchdog_valid(p,path,mode,program):
    r=read(path)
    cap=p['replay_cap_seconds']if mode=='replay'else p['max_calendar_seconds']
    original=utc(p['replay_absolute_deadline_utc']if mode=='replay'else p['absolute_deadline_utc'])
    if r['complete']is not True or r['mode']!=mode or r['program']!=program or r['child_returncode']!=0 or r.get('failure')is not None or r['protocol_sha256']!=p['_protocol_sha256']or r['source_sha256']!=p['source_sha256']['run_solar_job_v70.py']:
        raise ValueError('Incomplete, failed or unrelated external scientific watchdog receipt')
    start=utc(r['started_utc']);deadline=utc(r['absolute_deadline_utc']);elapsed=r['calendar_seconds']
    latest=utc(p['replay_latest_start_utc']if mode=='replay'else p['latest_start_utc'])
    if start>latest or r['cap_seconds']!=cap or deadline!=min(start+timedelta(seconds=cap),original)or isinstance(elapsed,bool)or not isinstance(elapsed,(int,float))or not math.isfinite(elapsed)or not 0<elapsed<=min(cap,(deadline-start).total_seconds()):
        raise ValueError('Scientific invocation exceeded or changed its original deadline')
    return r


def charged_job(c,p,stage):
    _,journal,_=journal_state(c['compute_ledger'],c['compute_ledger_sha256'],c['reservation_journal'])
    suffix='teacher_bank'if stage.startswith('teacher_')else'exact_replay'if stage.endswith('_replay')else'train'
    cap=p['replay_cap_seconds']if stage.endswith('_replay')else p['max_calendar_seconds']
    expected=p['_protocol_sha256']+'/'+suffix
    matches=[r for r in journal['reservations']if r['job_id']==expected]
    if len(matches)!=1 or matches[0]['category']!='training'or matches[0]['seconds']!=cap:
        raise ValueError('Scientific invocation is not charged exactly once in the canonical journal')


def completion(c,stage,require=False):
    path=protocol_path(c,stage)
    if not Path(path).exists():
        if require:raise ValueError('Prior stage protocol does not exist: '+stage)
        return None
    p=read(path);p['_protocol_sha256']=sha(path)
    mode='replay'if stage.endswith('_replay')else'run';receipt_path=p['watchdog_receipts'][mode]
    output=Path(p['output'])
    if not Path(receipt_path).exists():
        if (mode=='run'and output.exists())or Path(p['watchdog_log_prefix']+mode+'.log').exists():
            raise ValueError('Prior invocation is active/partial; preserve it, never restart: '+stage)
        if require:raise ValueError('Prior stage has not run: '+stage)
        return None
    program='build_solar_training_bank_v68.py'if stage.startswith('teacher_')else'train_solar_response_v63.py'
    receipt=watchdog_valid(p,receipt_path,mode,program);charged_job(c,p,stage)
    if (output/'FAILED.json').exists():raise ValueError('Prior scientific failure is preserved: '+stage)
    summary=read(output/'summary.json')
    if not summary['complete']or summary['protocol_sha256']!=p['_protocol_sha256']:
        raise ValueError('Scientific summary does not match its source-published protocol')
    if stage.startswith('teacher_'):
        seal=read(output/'COMPLETE.json')
        if not seal['complete']or seal['summary_sha256']!=sha(output/'summary.json')or seal['protocol_sha256']!=p['_protocol_sha256']or summary['heldout_image_reads']!=0:
            raise ValueError('Teacher stage is not sealed or read heldout training lights')
        expected=4 if stage=='teacher_pilot'else len(c['eligible_training_unit_indices'])
        if len(summary['records'])!=expected:raise ValueError('Teacher stage physical-pair coverage incomplete')
        wanted=c['eligible_training_unit_indices']if stage=='teacher_full'else[
            min(int(i)for i,g in c['training_unit_components'].items()if g==group)for group in c['training_groups'][:4]]
        if sorted(r['unit_index']for r in summary['records'])!=sorted(wanted):
            raise ValueError('Teacher stage substituted/dropped the frozen physical camera records')
        if stage=='teacher_pilot'and(summary['fresh_teacher_records']!=4 or summary['reused_complete_records']):
            raise ValueError('Teacher cost pilot must perform four fresh physical records')
        if stage=='teacher_full':
            bank=read(p['manifest'])
            if seal['manifest_sha256']!=sha(p['manifest'])or len(bank['records'])!=expected or len(bank['terrain_groups'])!=16:
                raise ValueError('Complete sealed all-eligible/16group teacher bank required')
    else:
        if summary['updates']!=p['updates']or summary['pooled']!=p['pooled']or summary['seed']!=p['seed']:
            raise ValueError('Training study configuration mismatch')
        for name,h in summary['checkpoint_sha256'].items():
            if sha(output/name)!=h:raise ValueError('Training checkpoint changed')
        if sha(output/f'history_{p["updates"]}.json')!=summary['all_history_sha256']:
            raise ValueError('Training history changed')
        if mode=='replay':
            replay=read(p['exact_replay_output'])
            if not replay['passed']or replay['updates']!=100 or not replay['exact_model_optimizer_rng_loss_gradient_sampler']or replay['protocol_sha256']!=p['_protocol_sha256']or replay['pilot_summary_sha256']!=sha(output/'summary.json'):
                raise ValueError('Actual exact100update pilot replay did not pass')
    return dict(protocol=path,protocol_sha256=p['_protocol_sha256'],summary=str(output/'summary.json'),summary_sha256=sha(output/'summary.json'),
        watchdog_receipt=receipt_path,watchdog_receipt_sha256=sha(receipt_path),receipt=receipt,summary_data=summary,protocol_data=p)


def common_protocol(c,contract_path,stage,cap):
    integer(cap)
    if cap>remaining(c)['training']:raise ValueError('Stage cap exceeds remaining cumulative training budget')
    output=stage_output(c,stage);watchprefix=output+'-watchdog-'
    created=now();lease=c['publication_dispatch_lease_seconds'];latest=created+timedelta(seconds=lease)
    return dict(cpu_threads=2,max_calendar_seconds=cap,created_utc=created.isoformat(),latest_start_utc=latest.isoformat(),
        publication_dispatch_lease_seconds=lease,queue_allowance_seconds=lease,
        absolute_deadline_utc=(latest+timedelta(seconds=cap)).isoformat(),output=output,
        source_sha256=c['source_sha256'],compute_ledger=c['compute_ledger'],compute_ledger_sha256=c['compute_ledger_sha256'],reservation_journal=c['reservation_journal'],
        watchdog_receipts=dict(run=watchprefix+'run.json',replay=watchprefix+'replay.json'),watchdog_log_prefix=watchprefix,
        learning_pipeline_contract=str(contract_path),learning_pipeline_contract_sha256=sha(contract_path),pipeline_stage=stage,
        calibration_or_final_access=False)


def pilot_links(c,mode):
    pilot=completion(c,mode+'_pilot',True);completion(c,mode+'_replay',True)
    p=pilot['protocol_data'];replay=p['exact_replay_output']
    return dict(pilot_protocol=pilot['protocol'],pilot_protocol_sha256=pilot['protocol_sha256'],
        pilot_summary=pilot['summary'],pilot_summary_sha256=pilot['summary_sha256'],
        pilot_replay_receipt=replay,pilot_replay_receipt_sha256=sha(replay),
        pilot_watchdog_receipt=p['watchdog_receipts']['run'],pilot_watchdog_receipt_sha256=sha(p['watchdog_receipts']['run']),
        replay_watchdog_receipt=p['watchdog_receipts']['replay'],replay_watchdog_receipt_sha256=sha(p['watchdog_receipts']['replay']))


def study_cap(c,mode):
    summary=completion(c,mode+'_pilot',True)['summary_data']
    measured=2*summary['calendar_seconds']*2000/100
    if not math.isfinite(measured)or measured<=0:raise ValueError('Positive actually measured study projection required')
    return math.ceil(measured),measured


def selection(c):
    """Select the repeated mode from complete, independently replayed poses."""
    from solar_evaluation_v70 import checkpoint_ranking,completed_training_evidence
    outcomes=[];evidence={};comparison=None
    missing=[spec[key]for spec in c['selection_evidence'].values()for key in('protocol','summary','metric_replay')if not Path(spec[key]).exists()]
    if missing:raise SelectionPending(missing)
    for mode in('sh','pooled'):
        spec=c['selection_evidence'][mode]
        protocol=read(spec['protocol']);summary=read(spec['summary']);replay=read(spec['metric_replay']);seal=read(Path(spec['summary']).parent/'COMPLETE.json')
        if protocol['split']!='model_selection'or protocol['purpose']!='checkpoint_selection'or protocol['tracks']!=['estimated_sun']or protocol['calibration_or_final_access']or not summary['complete']or summary.get('calibration_or_final_access',False)or summary['purpose']!='checkpoint_selection'or summary['protocol_sha256']!=sha(spec['protocol']):
            raise ValueError('Frozen primary-track development pose selection required')
        if (Path(spec['summary']).parent/'FAILED.json').exists()or not seal['complete']or not seal['external_watchdog_passed']or seal['protocol_sha256']!=sha(spec['protocol'])or seal['summary_sha256']!=sha(spec['summary'])or not replay['passed']or replay['summary_sha256']!=sha(spec['summary'])or replay['replayed_rows']!=len(summary['rows']):
            raise ValueError('Actual complete pose selection and independent metric replay required')
        if replay['replay_source_sha256']!=protocol['source_sha256']['verify_solar_evaluation_v70.py']:
            raise ValueError('Selection metric replay source differs from its published protocol')
        study=completion(c,mode+'_study',True)
        for profile in protocol['profiles']:
            completed_training_evidence(profile)
            if profile['training_evidence']['summary_sha256']!=study['summary_sha256']or profile['configuration']['pooled']!=(mode=='pooled'):
                raise ValueError('Selection profiles belong to a different study')
        configuration={k:v for k,v in protocol['profiles'][0]['configuration'].items()if k not in('checkpoint_sha256','pooled')}
        identity=dict(case_ids=protocol['case_ids'],manifest_sha256=summary['manifest_sha256'],source_sha256=protocol['source_sha256'],
            configuration=configuration,case_labels={r['case_id']:r['positive']for r in summary['rows']})
        if comparison is not None and comparison!=identity:
            raise ValueError('SH and pooled selection must share the same complete cases, sources and inference settings')
        comparison=identity
        ranking=checkpoint_ranking(summary['rows'],protocol['profiles'],protocol['case_ids'])
        outcomes.append(dict(mode=mode,**ranking['selected']))
        evidence[mode]={key:dict(path=spec[key],sha256=sha(spec[key]))for key in('protocol','summary','metric_replay')}
    outcomes.sort(key=lambda x:(-x['correct_poses'],-x['two_pixel_correspondences'],x['step'],x['mode']!='sh'))
    return dict(selected=outcomes[0],outcomes=outcomes,evidence=evidence,scope='Raw development pose selection; no acceptance/final claim.')


def prepare_next(contract_path):
    c=load_contract(contract_path)
    for stage in STAGES:
        if completion(c,stage)is not None:continue
        path=protocol_path(c,stage)
        if Path(path).exists():return dict(stage=stage,protocol=path,protocol_sha256=sha(path),awaiting_source_publication=True)
        # Earlier completed stages and their original external caps are checked
        # above; no new pilot receipt is manufactured by this builder.
        if stage.startswith('teacher_'):
            cap=c['teacher_pilot_cap_seconds']
            extra={}
            if stage=='teacher_full':
                pilot=completion(c,'teacher_pilot',True);summary=pilot['summary_data'];points=min(r['selected_physical_points']for r in summary['records'])
                measured=2*summary['calendar_seconds']*len(c['eligible_training_unit_indices'])/4*c['maximum_points_per_unit']/points
                if not math.isfinite(measured)or measured<=0:raise ValueError('Measured positive teacher extraction projection required')
                cap=math.ceil(measured);integer(cap,3600)
                extra=dict(cost_pilot=pilot['summary'],cost_pilot_sha256=pilot['summary_sha256'],factor_two_cost_projection_seconds=measured,
                    cache_roots=[stage_output(c,'teacher_pilot')],teacher_pilot_watchdog_receipt=pilot['watchdog_receipt'],teacher_pilot_watchdog_receipt_sha256=pilot['watchdog_receipt_sha256'])
            p=common_protocol(c,contract_path,stage,cap)
            p.update(phase='cost_pilot'if stage=='teacher_pilot'else'full_bank',manifest=str(Path(c['protocol_directory'])/(stage+'-bank.json')),
                original_master=c['original_master'],original_master_sha256=c['original_master_sha256'],dataset_summary=c['dataset_summary'],dataset_summary_sha256=c['dataset_summary_sha256'],
                geometry_audit=c['geometry_audit'],geometry_audit_sha256=c['geometry_audit_sha256'],lighting_schedule=c['lighting_schedule'],lighting_schedule_sha256=c['lighting_schedule_sha256'],
                seed=20261101,maximum_points_per_unit=c['maximum_points_per_unit'],teacher_batch_size=c['teacher_batch_size'],trial_pose_jitter=c['trial_pose_jitter'],
                teacher_directory=c['initializer'],teacher_checkpoint_sha256=c['initializer_sha256']['selected.pt'],teacher_training_sha256=c['initializer_sha256']['training.json'],
                expected_frozen_descriptor_sha256=c['expected_frozen_descriptor_sha256'],cache_roots=[])
            p.update(extra)
        else:
            bankstage=completion(c,'teacher_full',True);bankpath=bankstage['protocol_data']['manifest'];bank=read(bankpath)
            pooled=stage.startswith('pooled_');seed=20261101;updates=100 if stage.endswith('_pilot')else 2000;extra={}
            cap=c['training_pilot_cap_seconds']
            if updates==2000:
                # Before either primary study, account for BOTH modes and two
                # repetitions of the more expensive possible selected mode.
                caps={m:study_cap(c,m)[0]for m in('sh','pooled')}
                pending=[m for m in('sh','pooled')if completion(c,m+'_study')is None]
                reserve_future=sum(caps[m]for m in pending)+2*max(caps.values())
                if stage.startswith('seed_'):
                    try:selected=selection(c)
                    except SelectionPending as exc:return dict(stage=stage,awaiting_development_selection=True,required_artifacts=exc.paths,
                        instruction='Complete source-published model-selection pose comparisons and independent metric replay. No seed repetition or reserve inference was started.')
                    pooled=selected['selected']['mode']=='pooled';seed=int(stage[5:]);extra['development_selection']=selected
                    extra['fixed_reporting_step']=selected['selected']['step']
                    future=sum(completion(c,s)is None for s in('seed_20261102','seed_20261103'))*caps[selected['selected']['mode']]
                    reserve_future=future
                if reserve_future>remaining(c)['training']:
                    raise ValueError('Measured remaining primary studies and selected-seed repetitions do not fit the cumulative budget')
                mode='pooled'if pooled else'sh';cap,measured=study_cap(c,mode);extra.update(pilot_links(c,mode));extra['measured_training_projection_seconds']=measured
            p=common_protocol(c,contract_path,stage,cap)
            p.update(updates=updates,seed=seed,pooled=pooled,bank=bankpath,bank_sha256=sha(bankpath),bank_producer_completion_sha256=sha(bank['producer_completion_path']),
                initializer=c['initializer'],initializer_sha256=c['initializer_sha256'])
            if updates==100:
                replay_latest=utc(p['absolute_deadline_utc'])+timedelta(seconds=c['publication_dispatch_lease_seconds'])
                p.update(replay_cap_seconds=c['training_replay_cap_seconds'],replay_queue_allowance_seconds=c['publication_dispatch_lease_seconds'],
                    replay_latest_start_utc=replay_latest.isoformat(),
                    replay_absolute_deadline_utc=(replay_latest+timedelta(seconds=c['training_replay_cap_seconds'])).isoformat(),
                    exact_replay_output=stage_output(c,stage)+'-exact-replay.json')
            p.update(extra)
        Path(p['output']).parent.mkdir(parents=True,exist_ok=True)
        dump_new(path,p)
        return dict(stage=stage,protocol=path,protocol_sha256=sha(path),awaiting_source_publication=True,
            instruction='Publish this exact protocol and its frozen sources; then run the serial dispatcher with that publication commit. No publication or job was started.')
    return dict(learning_complete=True,scientific_goal_complete=False,scope='Bounded learning stages complete; evaluation, failures, calibration and final reserve remain separate.')


class PublicationPending(ValueError): pass
class SelectionPending(ValueError):
    def __init__(self,paths):self.paths=paths;super().__init__('Completed development selection evidence is not available')


def verify_publication(contract_path,protocol_path,commit):
    if len(commit or'')!=40 or any(x not in'0123456789abcdef'for x in commit):raise ValueError('Exact published source commit required')
    c=load_contract(contract_path);p=read(protocol_path)
    if p['learning_pipeline_contract_sha256']!=sha(contract_path)or p['reservation_journal']!=CANONICAL_JOURNAL or p['source_sha256']!=c['source_sha256']:
        raise ValueError('Prepared protocol differs from learning contract/canonical budget')
    verify_protocol_lease(c,p)
    for name in [str(contract_path),str(protocol_path),*p['source_sha256']]:
        try: public=subprocess.check_output(['git','show',commit+':'+name],stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:raise PublicationPending('Required source publication barrier not satisfied: '+name)from exc
        if public!=Path(name).read_bytes():raise PublicationPending('Required source publication barrier not satisfied: '+name)
    return p


def verify_protocol_lease(c,p):
    """Published queue allowance is finite and never added to scientific caps."""
    lease=c['publication_dispatch_lease_seconds'];created=utc(p['created_utc']);latest=utc(p['latest_start_utc'])
    if p['publication_dispatch_lease_seconds']!=lease or p['queue_allowance_seconds']!=lease or latest!=created+timedelta(seconds=lease)or utc(p['absolute_deadline_utc'])!=latest+timedelta(seconds=p['max_calendar_seconds']):
        raise ValueError('Original published dispatch lease or execution deadline changed')
    if p.get('updates')==100:
        replay_latest=utc(p['absolute_deadline_utc'])+timedelta(seconds=lease)
        if p['replay_queue_allowance_seconds']!=lease or utc(p['replay_latest_start_utc'])!=replay_latest or utc(p['replay_absolute_deadline_utc'])!=replay_latest+timedelta(seconds=p['replay_cap_seconds']):
            raise ValueError('Original published exact-replay lease changed')
