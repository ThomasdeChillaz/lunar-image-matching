"""Artifact-driven reference-only physical preparation and selection case assets.

All execution is separately published, bounded and charged to the one baseline
journal. This module itself performs metadata checks and protocol preparation;
it never starts Blender, reads an image, calls a model or runs a solar ray.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,hashlib,json,math,subprocess
from solar_learning_pipeline_v73 import (CANONICAL_JOURNAL,MAX_BASELINE,journal_state,utc,integer,
    pinned,sha,read,dump_new,now,PublicationPending,full_geometry_gate)
PREP_SOURCES=('prepare_solar_reference_v63.py','solar_map_v63.py','solar_physics_v63.py','solar_active_v63.py','north_terrain_pilot_v32.py','solar_sun_search_v68.py')
LOOKUP_SOURCES=('solar_lookup_v70.py','solar_lookup_worker_v70.py','solar_map_v63.py','solar_physics_v63.py','solar_active_v63.py','north_terrain_pilot_v32.py')
CASE_SOURCES=('solar_eval_data_v69.py','build_solar_eval_data_v69.py','index_solar_reference_assets_v69.py','solar_physics_v63.py','solar_sun_search_v68.py')
OWN_SOURCES=('solar_reference_pipeline_v74.py','build_solar_reference_protocol_v74.py','run_solar_reference_pipeline_v74.py','solar_reference_watchdog_v74.py')
STAGES=('reference_one','reference_four','reference_full','reference_index','case_assets')


class PilotShortfall(ValueError):
    def __init__(self,inventory):
        super().__init__('Fewer than four terrain-spread independently eligible references: record data shortfall before physical jobs')
        self.inventory=inventory


def bounded_time(value,maximum,label):
    if isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)or not 0<value<=maximum:
        raise ValueError('Finite measured '+label+' inside its original allowance required')
    return float(value)


def repo_relative(value):
    p=Path(value)
    if p.is_absolute()or '..'in p.parts or str(p)in('','.'):raise ValueError('Published protocols and outputs require explicit repository-relative paths')
    return str(p)


def source_closure():
    seeds=set(PREP_SOURCES+LOOKUP_SOURCES+CASE_SOURCES+OWN_SOURCES+('solar_learning_pipeline_v73.py',
        'solar_compute_budget_v67.py','solar_training_bank_v68.py','verify_solar_development_v68.py'))
    def imports(node):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):return
        if isinstance(node,ast.Import):
            for x in node.names:yield x.name.split('.')[0]+'.py'
        if isinstance(node,ast.ImportFrom)and node.module:yield node.module.split('.')[0]+'.py'
        for child in ast.iter_child_nodes(node):yield from imports(child)
    todo=list(seeds);seen=set()
    while todo:
        name=todo.pop()
        if name in seen:continue
        path=Path(name)
        if not path.is_file():raise ValueError('Reference pipeline implementation not integrated:'+name)
        seen.add(name);todo.extend(n for n in imports(ast.parse(path.read_text()))if Path(n).is_file()and n not in seen)
    return {n:sha(n)for n in sorted(seen)}


def schedule(audit):
    selected=sorted((u for u in audit['units']if u['split']=='model_selection'),key=lambda u:u['unit_index'])
    groups=sorted({u['component']for u in selected})
    if len(groups)!=8 or len(selected)!=32 or any(sum(u['component']==g for u in selected)!=4 for g in groups):raise ValueError('Original eight groups/four cameras each required')
    if any(type(u['eligible'])is not bool or type(u['unit_index'])is not int for u in selected)or len({u['unit_index']for u in selected})!=32:raise ValueError('Every generated camera and independent boolean eligibility is required')
    eligible=[u for u in selected if u['eligible']];eligible_groups=sorted({u['component']for u in eligible})
    ids=[u['unit_index']for u in eligible]
    if len(eligible_groups)<4:raise PilotShortfall(selection_metadata(audit))
    # Four deterministic quantiles over available terrain groups. Oracle/pose
    # outcomes never participate in the camera or pilot selection.
    pilot=[min(u['unit_index']for u in eligible if u['component']==eligible_groups[j*len(eligible_groups)//4])for j in range(4)]
    return ids,groups,pilot


def selection_metadata(audit):
    rows=sorted((u for u in audit['units']if u['split']=='model_selection'),key=lambda u:u['unit_index'])
    groups=sorted({u['component']for u in rows});eligible_groups=sorted({u['component']for u in rows if u['eligible']})
    return dict(terrain_groups=groups,generated_selection_units=rows,generated_selection_unit_count=len(rows),
        eligible_selection_unit_count=sum(u['eligible']for u in rows),excluded_selection_units=[u for u in rows if not u['eligible']],
        eligible_terrain_groups=eligible_groups,zero_eligible_terrain_groups=sorted(set(groups)-set(eligible_groups)))


def remaining(c):
    ledger,journal,charged=journal_state(c['compute_ledger'],c['compute_ledger_sha256'],c['reservation_journal'])
    initial=c['initial_reservations']
    if journal['reservations'][:len(initial)]!=initial:raise ValueError('Canonical reservation history changed')
    value=min(ledger['remaining_baseline_wall_seconds']-charged['baseline'],c['baseline_ceiling_seconds']-(charged['baseline']-c['initial_charged']['baseline']))
    if value<0:raise ValueError('Original cumulative baseline budget exhausted')
    return value


def unit_paths(c,index):
    name=f'u{index:03d}'
    return dict(precompute=str(Path(c['protocol_directory'])/'references'/(name+'-precompute.json')),
        lookup=str(Path(c['protocol_directory'])/'references'/(name+'-lookup.json')),
        archive=str(Path(c['run_directory'])/'references'/(name+'.npz')),
        watchdog=str(Path(c['run_directory'])/'watchdogs'/(name+'.json')),log=str(Path(c['run_directory'])/'logs'/(name+'.log')))


def physical_protocols(c,index):
    from solar_physics_v63 import lighting_schedule
    row=c['reference_units'][str(index)];approved=row['approved'];paths=unit_paths(c,index)
    common=dict(cpu_threads=2,query_truth_used=False,reference_geometry=approved['reference_geometry_path'],
        reference_geometry_sha256=approved['reference_geometry_sha256'],mesh=approved['scene_mesh_path'],mesh_sha256=approved['scene_mesh_sha256'],
        scene_mesh=approved['scene_mesh_path'],scene_mesh_sha256=approved['scene_mesh_sha256'],geometry_audit=c['geometry_audit'],geometry_audit_sha256=c['geometry_audit_sha256'],approved_reference_key=row['approved_reference_key'])
    first=lighting_schedule(row['latitude_deg'])[0]
    pre=dict(common,max_calendar_seconds=c['per_reference_cap_seconds'],output=paths['archive'],precompute='coarse',
        latitude_deg=row['latitude_deg'],reference_hour_angle_deg=first['hour_angle_deg'],reference_subsolar_latitude_deg=first['subsolar_latitude_deg'],source_sha256={n:c['source_sha256'][n]for n in PREP_SOURCES})
    lookup=dict(common,schema_version=70,max_lookup_calendar_seconds=c['online_lookup_cap_seconds'],max_request_seconds=c['online_request_cap_seconds'],source_sha256={n:c['source_sha256'][n]for n in LOOKUP_SOURCES})
    return pre,lookup


def initialize(settings_path,contract_path):
    s=read(settings_path)
    if s.get('prepared_only',False):raise ValueError('Prepared reference settings cannot initialize jobs')
    if s['calibration_or_final_access']is not False or s['query_truth_in_primary_inference']is not False:raise ValueError('Reference-only development preparation required')
    # This real gate precedes reference inputs/protocols, budget admission and all output construction.
    gate=full_geometry_gate(s['original_master'],s['dataset_summary'],s['geometry_audit'])
    try:ids,groups,pilot=schedule(gate.audit)
    except PilotShortfall as exc:
        dump_new(str(contract_path)+'.SHORTFALL.json',dict(complete=False,physical_jobs_admitted=False,reason=str(exc),
            original_master_sha256=gate.original_master_sha256,dataset_summary_sha256=gate.dataset_summary_sha256,geometry_audit_sha256=gate.geometry_audit_sha256,**exc.inventory))
        raise
    integer(s['baseline_ceiling_seconds'],MAX_BASELINE);integer(s['per_reference_cap_seconds'],900)
    integer(s['metadata_cap_seconds'],1800);integer(s['online_lookup_cap_seconds'],900);integer(s['online_request_cap_seconds'],300)
    if s['metadata_cap_seconds']<10:raise ValueError('Metadata phase must retain bounded work and cleanup time')
    integer(s['publication_dispatch_lease_seconds'],3600);integer(s['dispatch_overhead_seconds'],600)
    if s['pilot_one_cap_seconds']<s['per_reference_cap_seconds']+10:raise ValueError('One-reference phase must cover startup/work/termination and setup overhead')
    integer(s['pilot_one_cap_seconds'],1800)
    if utc(s['preparation_absolute_deadline_utc'])<=now():raise TimeoutError('Original reference-preparation deadline already expired')
    ledger_sha=sha(s['compute_ledger']);_,journal,charged=journal_state(s['compute_ledger'],ledger_sha,s['reservation_journal'])
    for key in('protocol_directory','run_directory','reference_assets_index','case_assets_output','stage_journal'):
        repo_relative(s[key])
    repo_relative(contract_path)
    if Path(s['reference_assets_index']).exists()or Path(s['case_assets_output']).exists()or Path(s['stage_journal']).exists():raise FileExistsError('Preserve every previous metadata artifact/state; no automatic retry')
    c=dict(s,schema_version=74,contract_path=str(contract_path),settings_path=str(settings_path),settings_sha256=sha(settings_path),source_sha256=source_closure(),
        original_master_sha256=gate.original_master_sha256,dataset_summary_sha256=gate.dataset_summary_sha256,geometry_audit_sha256=gate.geometry_audit_sha256,
        compute_ledger_sha256=ledger_sha,initial_reservations=journal['reservations'],initial_charged=charged,
        canonical_journal_absolute=str(Path(CANONICAL_JOURNAL).absolute()),selection_units=ids,selection_groups=groups,pilot_units=pilot,
        reference_units={},published_protocols_required_before_each_invocation=True,
        query_truth_used_for_physical_prep=False,scoring_labels_materialized_only_in_case_stage=True,
        scope='Every independently eligible selection reference, all32generated cameras retained in metadata, generic coarse Sun and schema70 cases; no matcher, teacher, training, calibration or final reserve inference.',**selection_metadata(gate.audit))
    for binary in('blender_executable','python_executable'):c[binary+'_sha256']=sha(c[binary])
    for index in ids:
        # Audit row order need not imply index order.
        row=next(r for r in gate.audit['units']if r['unit_index']==index)
        spec=gate.master['units'][index];pair=spec['camera_pairs'][0];key=row['component']+'/'+pair['family']+'-'+str(pair['pair_index'])
        c['reference_units'][str(index)]=dict(component=row['component'],latitude_deg=spec['site']['centre_latitude_deg'],approved_reference_key=key,approved=gate.audit['approved_reference_inputs'][key])
    if c['pilot_one_cap_seconds']+2*c['metadata_cap_seconds']>remaining(c):raise ValueError('Initial physical pilot and mandatory metadata stages exceed canonical budget')
    # Freeze ALL reference cameras/physical protocols before the first pilot.
    for index in ids:
        paths=unit_paths(c,index);pre,lookup=physical_protocols(c,index)
        if any(Path(paths[k]).exists()for k in('precompute','lookup','archive','watchdog','log')):raise FileExistsError('Preserve previous physical protocols/assets; use a declared version')
    dump_new(contract_path,c)
    for index in ids:
        paths=unit_paths(c,index);pre,lookup=physical_protocols(c,index);dump_new(paths['precompute'],pre);dump_new(paths['lookup'],lookup)
    return c


def load_contract(path,*,allow_expired=False):
    c=read(path)
    if c.get('prepared_only',False)or c['schema_version']!=74 or c['contract_path']!=str(path)or c['calibration_or_final_access']is not False or c['query_truth_in_primary_inference']is not False:raise ValueError('Initialized reference-only physical preparation contract required')
    for key in('original_master','dataset_summary','geometry_audit','compute_ledger'):pinned(c[key],c[key+'_sha256'])
    audit=read(c['geometry_audit']);ids,groups,pilot=schedule(audit)
    if (c['selection_units'],c['selection_groups'],c['pilot_units'])!=(ids,groups,pilot)or any(c[k]!=v for k,v in selection_metadata(audit).items()):raise ValueError('Full generated inventory or independent eligibility changed')
    for n,h in c['source_sha256'].items():
        if sha(n)!=h:raise ValueError('Frozen physical/asset source changed:'+n)
    if c['canonical_journal_absolute']!=str(Path(CANONICAL_JOURNAL).absolute()):raise ValueError('Canonical journal location changed')
    for binary in('blender_executable','python_executable'):
        if sha(c[binary])!=c[binary+'_sha256']:raise ValueError('Frozen scientific executable changed:'+binary)
    for index in c['selection_units']:
        paths=unit_paths(c,index)
        if (read(paths['precompute']),read(paths['lookup']))!=physical_protocols(c,index):raise ValueError('Original reference-only physical protocols changed; no new information may be injected')
    if allow_expired:
        ledger,journal,charged=journal_state(c['compute_ledger'],c['compute_ledger_sha256'],c['reservation_journal'])
        if charged['baseline']>ledger['remaining_baseline_wall_seconds']:raise ValueError('Cumulative baseline ledger exceeded its original allowance')
    else:remaining(c)
    if not allow_expired and now()>=utc(c['preparation_absolute_deadline_utc']):raise TimeoutError('Original physical-preparation deadline expired; no reset')
    return c


def stage_paths(c,stage):
    return str(Path(c['protocol_directory'])/(stage+'.json')),Path(c['run_directory'])/'stages'/stage


def validate_phase(c,path,p):
    stage=p['stage']
    if stage not in STAGES or (p['schema_version'],p['cpu_threads'])!=(74,2)or p['calibration_or_final_access']is not False:raise ValueError('Reference-only fixed scientific phase required')
    expected_path,root=stage_paths(c,stage)
    if str(path)!=expected_path or p['output']!=str(root)or p['contract']!=c['contract_path']or p['contract_sha256']!=sha(c['contract_path'])or p['source_sha256']!=c['source_sha256']:raise ValueError('Immutable phase/contract/source identity differs')
    for k in('compute_ledger','compute_ledger_sha256','reservation_journal'):
        if p[k]!=c[k]:raise ValueError('Phase changed its original canonical budget identity')
    created,latest,deadline=map(utc,(p['created_utc'],p['latest_start_utc'],p['absolute_deadline_utc']))
    if latest-created!=timedelta(seconds=c['publication_dispatch_lease_seconds'])or deadline-latest!=timedelta(seconds=p['max_calendar_seconds'])or deadline>utc(c['preparation_absolute_deadline_utc']):raise ValueError('Phase changed its original lease/deadline')
    if stage=='reference_one':ids=c['pilot_units'][:1];prior=[];cap=c['pilot_one_cap_seconds']
    elif stage=='reference_four':ids=c['pilot_units'][1:];prior=c['pilot_units'][:1];cap=measured_cap(c,p['prior_reference_rows'],3)
    elif stage=='reference_full':ids=[i for i in c['selection_units']if i not in c['pilot_units']];prior=c['pilot_units'];cap=measured_cap(c,p['prior_reference_rows'],len(ids))
    else:ids=[];prior=[];cap=c['metadata_cap_seconds']
    if p['max_calendar_seconds']!=cap or [r['unit_index']for r in p['prior_reference_rows']]!=prior:raise ValueError('Measured pilot admission or prior schedule differs')
    jobs=p['jobs']
    if stage in STAGES[:3]:
        if len(jobs)!=len(ids):raise ValueError('Every declared reference must be retained')
        for job,index in zip(jobs,ids):
            paths=unit_paths(c,index)
            if job!=dict(kind='reference',unit_index=index,program='prepare_solar_reference_v63.py',protocol=paths['precompute'],protocol_sha256=sha(paths['precompute']),work_cap_seconds=c['per_reference_cap_seconds'],log=paths['log'],watchdog_receipt=paths['watchdog']):raise ValueError('Fixed reference job identity differs')
    else:
        childpath=str(Path(c['protocol_directory'])/(stage+'-child.json'))
        expected=dict(kind='python',program='index_solar_reference_assets_v69.py'if stage=='reference_index'else'build_solar_eval_data_v69.py',protocol=childpath,protocol_sha256=sha(childpath),work_cap_seconds=cap-5,log=str(root)+'-child.log',watchdog_receipt=str(root)+'-child-watchdog.json')
        if jobs!=[expected]or read(childpath)!=asset_protocol(c,'reference_asset_index'if stage=='reference_index'else'development_case_assets'):raise ValueError('Fixed asset job/protocol identity differs')
    return p


def validate_watchdog(c,phase_path,phase,job):
    w=read(job['watchdog_receipt'])
    if (w['complete']is not True or w['failure']is not None or w['child_returncode']!=0 or w['child_protocol_sha256']!=job['protocol_sha256']or
        w['program']!=job['program']or w['kind']!=job['kind']or w['watchdog_source_sha256']!=c['source_sha256']['solar_reference_watchdog_v74.py']or
        w['phase_protocol']!=str(phase_path)or w['phase_protocol_sha256']!=sha(phase_path)or w['work_cap_seconds']!=job['work_cap_seconds']or
        w['phase_hard_deadline_utc']!=phase['absolute_deadline_utc']):raise ValueError('Successful original external watchdog provenance required')
    started,ended,work=map(utc,(w['started_utc'],w['ended_utc'],w['work_deadline_utc']))
    elapsed=bounded_time(w['calendar_seconds'],phase['max_calendar_seconds'],'startup-inclusive child time')
    if (not utc(phase['created_utc'])<=started<ended<=work or work>started+timedelta(seconds=job['work_cap_seconds'])or
        work>utc(phase['absolute_deadline_utc'])-timedelta(seconds=5)or abs((ended-started).total_seconds()-elapsed)>1):raise ValueError('Original external child time/deadline interval differs')
    if len(w['publication_commit']or'')!=40 or any(x not in'0123456789abcdef'for x in w['publication_commit']):raise ValueError('Exact watchdog source publication revision required')
    return w


def validate_reference(c,index,*,validate_arrays=True):
    paths=unit_paths(c,index);pre=read(paths['precompute']);lookup=read(paths['lookup']);receipt=read(paths['archive']+'.json');watch=read(paths['watchdog']);row=c['reference_units'][str(index)]
    if (pre,lookup)!=physical_protocols(c,index):raise ValueError('Reference physical protocol changed from its frozen construction')
    phasepath=watch['phase_protocol'];phase=read(phasepath);validate_phase(c,phasepath,phase)
    jobs=[j for j in phase['jobs']if j['kind']=='reference'and j['unit_index']==index]
    if len(jobs)!=1:raise ValueError('A successful original phase must own this reference')
    validate_watchdog(c,phasepath,phase,jobs[0])
    if (receipt['complete']is not True or receipt['protocol_sha256']!=sha(paths['precompute'])or receipt['output_sha256']!=sha(paths['archive'])or
        receipt['source_sha256']!=pre['source_sha256']or receipt['publication_commit']!=watch['publication_commit']or receipt['query_truth_used']is not False or receipt['neural_forwards']!=0 or receipt['precompute']!='coarse'or
        receipt['map_mesh_sha256']!=pre['mesh_sha256']or receipt['original_scene_mesh_sha256']!=row['approved']['scene_mesh_sha256']or receipt['original_scene_ceiling_m']!=row['approved']['scene_maximum_height_m']):raise ValueError('Complete measured reference-only physical archive required')
    if validate_arrays:
        import numpy as np
        from solar_eval_data_v69 import validate_reference_archive,validate_generic_catalogue
        geometry=validate_reference_archive(paths['archive']);validate_generic_catalogue(geometry,row['latitude_deg'])
        if not np.array_equal(geometry['solar_materialized_mask'],geometry['coarse_sun_mask']):raise ValueError('Pilot/full primary tables must compute generic COARSE states only')
    elapsed=watch['calendar_seconds'];bounded_time(receipt['calendar_seconds'],min(pre['max_calendar_seconds'],elapsed),'internal ray preparation time')
    return dict(unit_index=index,archive=paths['archive'],archive_sha256=sha(paths['archive']),program_receipt=paths['archive']+'.json',program_receipt_sha256=sha(paths['archive']+'.json'),
        watchdog_receipt=paths['watchdog'],watchdog_receipt_sha256=sha(paths['watchdog']),outer_calendar_seconds=elapsed,internal_calendar_seconds=receipt['calendar_seconds'])


def metadata_outputs(c,stage):
    if stage=='reference_index':
        path=Path(c['reference_assets_index']);result=read(path)
        if (result['complete']is not True or result['schema_version']!=69 or result['split']!='model_selection'or result['calibration_or_final_access']is not False or
            {r['unit_index']for r in result['units']}!=set(c['selection_units'])or len(result['units'])!=len(c['selection_units'])):raise ValueError('Exactly every independently eligible reference asset is required')
        files={str(path):sha(path)}
    elif stage=='case_assets':
        root=Path(c['case_assets_output']);result=read(root/'manifest.json');seal=read(root/'COMPLETE.json')
        if (result['complete']is not True or result['schema_version']!=70 or result['split']!='model_selection'or result['calibration_or_final_access']is not False or
            result['terrain_groups']!=c['selection_groups']or result['eligible_unit_indices']!=c['selection_units']or result['positive_cases']!=6*len(c['selection_units'])or result['negative_cases']!=6*len(c['selection_units'])or len(result['cases'])!=12*len(c['selection_units'])):raise ValueError('Every eligible camera/sixlights and matched nonoverlap case is required')
        if (root/'FAILED.json').exists():raise ValueError('Failed case output is preserved, never reusable as completed')
        files={str(p.relative_to(root)):sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p.name!='COMPLETE.json'}
        if seal['sha256']!=files:raise ValueError('Every saved case/scoring artifact must retain its completion checksum')
        files={str(root/k):v for k,v in files.items()};files[str(root/'COMPLETE.json')]=sha(root/'COMPLETE.json')
    else:return {}
    if any(result[k]!=c[k]for k in selection_metadata(read(c['geometry_audit']))):raise ValueError('Original32camera inventory, exclusion reasons or eight raw groups were lost')
    childpath=str(Path(c['protocol_directory'])/(stage+'-child.json'));child=read(childpath)
    if result['protocol_sha256']!=sha(childpath)or result['source_sha256']!=child['source_sha256']or result['geometry_audit_sha256']!=c['geometry_audit_sha256']or result['dataset_summary_sha256']!=c['dataset_summary_sha256']or result['original_master_sha256']!=c['original_master_sha256']:raise ValueError('Actual child output provenance differs')
    return files


def completion(c,stage,required=False):
    path,root=stage_paths(c,stage)
    if not(root/'COMPLETE.json').exists():
        if root.exists():raise ValueError('A partial/failed physical stage is preserved; no automatic retry:'+stage)
        if required:raise ValueError('Prior measured stage has not completed:'+stage)
        return None
    p=read(path);validate_phase(c,path,p);summary=read(root/'summary.json');seal=read(root/'COMPLETE.json')
    if (root/'FAILED.json').exists()or seal['complete']is not True or seal['summary_sha256']!=sha(root/'summary.json')or seal['protocol_sha256']!=sha(path)or summary['complete']is not True or summary['protocol_sha256']!=sha(path):
        raise ValueError('Physical stage completion seal differs')
    elapsed=bounded_time(summary['calendar_seconds'],p['max_calendar_seconds'],'whole physical/metadata phase time')
    started,ended=utc(summary['started_utc']),utc(summary['ended_utc'])
    if (summary['max_calendar_seconds']!=p['max_calendar_seconds']or summary['absolute_deadline_utc']!=p['absolute_deadline_utc']or
        summary['source_sha256']!=c['source_sha256']or not utc(p['created_utc'])<=started<=utc(p['latest_start_utc'])or not started<ended<=utc(p['absolute_deadline_utc'])or
        abs((ended-started).total_seconds()-elapsed)>1):raise ValueError('Completed phase exceeded/changed its original time/source lease')
    _,journal,_=journal_state(c['compute_ledger'],c['compute_ledger_sha256'],c['reservation_journal'])
    charges=[r for r in journal['reservations']if r['job_id']==sha(path)+'/reference_physics']
    if len(charges)!=1 or charges[0]['category']!='baseline'or charges[0]['seconds']!=p['max_calendar_seconds']:raise ValueError('Physical stage must be charged exactly once, without refund')
    expected=c['pilot_units'][:1]if stage=='reference_one'else c['pilot_units']if stage=='reference_four'else c['pilot_units']+[i for i in c['selection_units']if i not in c['pilot_units']]if stage=='reference_full'else[]
    if [r['unit_index']for r in summary['reference_rows']]!=expected:raise ValueError('No eligible reference may be dropped or repeated')
    for row in summary['reference_rows']:
        if validate_reference(c,row['unit_index'],validate_arrays=False)!=row:raise ValueError('Immutable completed reference archive/receipt changed')
    for job in p['jobs']:
        w=validate_watchdog(c,path,p,job)
        if w['publication_commit']!=summary['publication_commit']:raise ValueError('Child and phase source publications differ')
    if summary['output_sha256']!=metadata_outputs(c,stage):raise ValueError('Completed index/case artifact changed')
    return summary


def measured_cap(c,rows,count):
    values=[r['outer_calendar_seconds']for r in rows]
    if not values or any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or v<=0 for v in values):raise ValueError('Actual measured positive external pilot intervals required')
    return math.ceil(2*max(values)*count+c['dispatch_overhead_seconds'])


def asset_protocol(c,phase):
    from solar_eval_data_v69 import LIGHT_PAIRS
    p=dict(schema_version=69,phase=phase,split='model_selection',calibration_or_final_access=False,cpu_threads=2,max_calendar_seconds=1800,
        light_pairs=[list(x)for x in LIGHT_PAIRS],original_master=c['original_master'],original_master_sha256=c['original_master_sha256'],dataset_summary=c['dataset_summary'],dataset_summary_sha256=c['dataset_summary_sha256'],
        geometry_audit=c['geometry_audit'],geometry_audit_sha256=c['geometry_audit_sha256'],geometry_audit_source='verify_solar_development_v68.py',geometry_audit_source_sha256=c['source_sha256']['verify_solar_development_v68.py'],source_sha256={n:c['source_sha256'][n]for n in CASE_SOURCES})
    if phase=='reference_asset_index':
        p.update(output=c['reference_assets_index'],reference_unit_protocols={str(i):dict(reference_precompute_protocol=unit_paths(c,i)['precompute'],reference_precompute_protocol_sha256=sha(unit_paths(c,i)['precompute']),solar_lookup_protocol=unit_paths(c,i)['lookup'],solar_lookup_protocol_sha256=sha(unit_paths(c,i)['lookup']))for i in c['selection_units']})
    else:p.update(output=c['case_assets_output'],reference_assets_index=c['reference_assets_index'],reference_assets_index_sha256=sha(c['reference_assets_index']))
    return p


def prepare_next(contract_path):
    c=load_contract(contract_path)
    for stage in STAGES:
        if completion(c,stage)is not None:continue
        path,root=stage_paths(c,stage)
        if Path(path).exists():return dict(stage=stage,protocol=path,awaiting_source_publication=True)
        prior=[];jobs=[]
        if stage=='reference_one':ids=c['pilot_units'][:1];cap=c['pilot_one_cap_seconds']
        elif stage=='reference_four':
            prior=completion(c,'reference_one',True)['reference_rows'];ids=c['pilot_units'][1:];cap=measured_cap(c,prior,len(ids))
        elif stage=='reference_full':
            completion(c,'reference_one',True);prior=completion(c,'reference_four',True)['reference_rows'];ids=[i for i in c['selection_units']if i not in c['pilot_units']];cap=measured_cap(c,prior,len(ids))
        else:
            completion(c,'reference_full',True);ids=[];cap=c['metadata_cap_seconds']
            if stage=='case_assets':completion(c,'reference_index',True)
        if cap+(2 if stage in STAGES[:3]else 0)*c['metadata_cap_seconds']>remaining(c):raise ValueError('Measured physical preparation plus mandatory metadata phases exceed remaining canonical baseline budget')
        created=now();latest=created+timedelta(seconds=c['publication_dispatch_lease_seconds']);absolute=min(latest+timedelta(seconds=cap),utc(c['preparation_absolute_deadline_utc']))
        if (absolute-latest).total_seconds()<cap:raise TimeoutError('Original overall preparation deadline cannot fit the next fixed phase')
        if stage in STAGES[:3]:
            for i in ids:jobs.append(dict(kind='reference',unit_index=i,program='prepare_solar_reference_v63.py',protocol=unit_paths(c,i)['precompute'],protocol_sha256=sha(unit_paths(c,i)['precompute']),work_cap_seconds=c['per_reference_cap_seconds'],log=unit_paths(c,i)['log'],watchdog_receipt=unit_paths(c,i)['watchdog']))
        else:
            childpath=str(Path(c['protocol_directory'])/(stage+'-child.json'));child=asset_protocol(c,'reference_asset_index'if stage=='reference_index'else'development_case_assets');dump_new(childpath,child)
            jobs=[dict(kind='python',program='index_solar_reference_assets_v69.py'if stage=='reference_index'else'build_solar_eval_data_v69.py',protocol=childpath,protocol_sha256=sha(childpath),work_cap_seconds=cap-5,log=str(root)+'-child.log',watchdog_receipt=str(root)+'-child-watchdog.json')]
        p=dict(schema_version=74,stage=stage,contract=str(contract_path),contract_sha256=sha(contract_path),source_sha256=c['source_sha256'],cpu_threads=2,max_calendar_seconds=cap,
            created_utc=created.isoformat(),latest_start_utc=latest.isoformat(),absolute_deadline_utc=absolute.isoformat(),output=str(root),jobs=jobs,prior_reference_rows=prior,
            compute_ledger=c['compute_ledger'],compute_ledger_sha256=c['compute_ledger_sha256'],reservation_journal=c['reservation_journal'],calibration_or_final_access=False,
            scope='No automatic publication or retry. Metadata case stage writes scoring-only query labels; physical preparation never receives query truth.')
        dump_new(path,p);return dict(stage=stage,protocol=path,awaiting_source_publication=True,instruction='Publish exact contract, sources, all reference protocols and this stage/child protocol before dispatch.')
    return dict(reference_preparation_complete=True,scientific_goal_complete=False,case_manifest=str(Path(c['case_assets_output'])/'manifest.json'))


def verify_publication(contract_path,phase_path,commit,*,historical_only=False):
    c=load_contract(contract_path,allow_expired=historical_only);p=read(phase_path);validate_phase(c,phase_path,p)
    if len(commit or'')!=40 or any(x not in'0123456789abcdef'for x in commit):raise ValueError('Exact published source revision required')
    if p['contract_sha256']!=sha(contract_path):raise ValueError('Phase belongs to another immutable contract')
    files=[contract_path,phase_path,*c['source_sha256']]
    for i in c['selection_units']:files.extend(unit_paths(c,i)[k]for k in('precompute','lookup'))
    files.extend(j['protocol']for j in p['jobs'])
    for path in dict.fromkeys(files):
        try:public=subprocess.check_output(['git','show',commit+':'+str(path)],stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:raise PublicationPending('Next exact reference protocol/source is not published:'+str(path))from exc
        if public!=Path(path).read_bytes():raise PublicationPending('Next exact reference protocol/source differs from published bytes:'+str(path))
    return c,p


def completed_reference_preparation(protocol_path,summary_path):
    """Hash/metadata-only evidence replay; never authorizes a launch or reads NPZ arrays."""
    p=read(protocol_path)
    if p['stage']!='reference_full':raise ValueError('The complete independently eligible reference phase is required')
    c=load_contract(p['contract'],allow_expired=True);validate_phase(c,protocol_path,p)
    if str(summary_path)!=str(Path(p['output'])/'summary.json'):raise ValueError('Exact full-reference summary required')
    inputs={p['contract']:sha(p['contract'])};reserved=0;phase_times={}
    for stage in STAGES[:3]:
        phasepath,root=stage_paths(c,stage);summary=completion(c,stage,True)
        verify_publication(c['contract_path'],phasepath,summary['publication_commit'],historical_only=True)
        for name in(phasepath,str(root/'summary.json'),str(root/'COMPLETE.json')):inputs[name]=sha(name)
        reserved+=read(phasepath)['max_calendar_seconds']
        phase_times[stage]=summary['calendar_seconds']
    rows=read(summary_path)['reference_rows'];outer={};inner={}
    for row in rows:
        paths=unit_paths(c,row['unit_index']);archive=row['archive'];outer[archive]=row['outer_calendar_seconds'];inner[archive]=row['internal_calendar_seconds']
        for name in(archive,row['program_receipt'],row['watchdog_receipt'],paths['precompute'],paths['lookup']):inputs[name]=sha(name)
    return dict(pinned_inputs=inputs,outer_calendar_seconds_by_archive=outer,internal_calendar_seconds_by_archive=inner,
        reference_archives=sorted(outer),reference_unit_indices=sorted(r['unit_index']for r in rows),baseline_reserved_seconds=reserved,phase_calendar_seconds=phase_times,
        scope='Completed physical reference preparation only. Outer costs include Blender startup and child shutdown; full phase costs also include serial validation. Historical replay grants no job admission.',**selection_metadata(read(c['geometry_audit'])))
