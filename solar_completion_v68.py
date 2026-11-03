"""Missing-only data completion after the preserved v67 preflight failure.

Only camera rotation replay gets the existing audited1e-12 cross-runtime
allowance. Stored matrices drive rendering; all physical draws stay exact.
This is a new fixed-cap completion, never an extension of the expired v67 job.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
from solar_data_v65 import schedule
from north_camera_contract_v59 import verify_schedule

RENDER_SOURCES=['solar_completion_v68.py','render_solar_completion_v68.py','run_solar_completion_v68.py',
 'render_solar_data_v65.py','solar_data_v65.py','solar_physics_v63.py','north_camera_contract_v59.py',
 'north_camera_scene_v59.py','north_camera_correspondences_v59.py','north_development_plan_v35.py',
 'north_camera_variation_v59.py','north_render_plan_v31.py','north_terrain_pilot_v32.py','blender_camera_control_v31.py']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def camera_replay(unit):
    if len(unit['camera_pairs'])!=1:raise ValueError('One frozen camera pair per unit required')
    actual=unit['camera_pairs'][0];key=tuple(actual[k]for k in('component','family','pair_index'))
    generated=schedule(unit['site'],unit['core_max_height'],2)
    matches=[x for x in generated if tuple(x[k]for k in('component','family','pair_index'))==key]
    if len(matches)!=1:raise ValueError('Camera identity not in original deterministic schedule')
    return verify_schedule({'pairs':[actual]},{'pairs':matches})


def verify_seal(folder):
    folder=Path(folder);seal=read(folder/'COMPLETE.json');expected=set(seal['sha256'])
    actual={str(p.relative_to(folder))for p in folder.rglob('*')if p.is_file()and p!=folder/'COMPLETE.json'}
    if actual!=expected:raise ValueError('Complete-unit file inventory differs')
    for n,h in seal['sha256'].items():
        if sha(folder/n)!=h:raise ValueError('Sealed unit changed:'+n)
    summary=read(folder/'summary.json')
    if not summary['complete']or summary['rendered_images']!=48 or len(summary['images'])!=48:raise ValueError('Complete48image unit required')
    return summary


def require_renderer_summary(folder,contract_digest):
    folder=Path(folder);path=folder/'summary.json'
    if not path.is_file():raise RuntimeError('Renderer exited without an explicit complete unit summary')
    summary=read(path)
    if not summary['complete']or summary['rendered_images']!=48 or len(summary['images'])!=48 or summary['completion_contract_sha256']!=contract_digest:raise ValueError('Renderer completion receipt incomplete or wrong admission')
    seen=set();paths=set()
    for row in summary['images']:
        key=(row['role'],row['light'])
        if key in seen or row['path']in paths:raise ValueError('Repeated physical image acquisition record')
        seen.add(key);paths.add(row['path'])
        if sha(folder/row['path'])!=row['sha256']:raise ValueError('Newly rendered image absent or changed')
    if seen!={(role,light)for role in('reference','query')for light in range(24)}:raise ValueError('Two views/all24physical lights required')
    return summary


def verify_original(contract):
    for key in('original_master','original_failed_state','snapshot','schedule_replay'):
        if sha(contract[key])!=contract[key+'_sha256']:raise ValueError('Preserved input changed:'+key)
    master=read(contract['original_master']);snapshot=read(contract['snapshot'])
    if read(contract['original_failed_state'])['state']!='failed':raise ValueError('Original failed attempt must remain failed')
    if master['absolute_deadline_utc']!=contract['original_shared_deadline_utc']or contract['original_deadline_met']:raise ValueError('Original expired deadline must not be extended or relabelled')
    root=Path(contract['original_root'])
    actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()}
    if actual!=set(snapshot['original_output_files_sha256']):raise ValueError('Original raw dataset inventory changed')
    for n,h in snapshot['original_output_files_sha256'].items():
        if sha(root/n)!=h:raise ValueError('Original raw file changed:'+n)
    complete=[r['unit_index']for r in snapshot['complete_units']]
    if complete!=list(range(10))or contract['missing_unit_indices']!=list(range(10,96)):raise ValueError('Exactly ten reused and86missing units required')
    if Path(master['units'][10]['output']).exists():raise ValueError('Original preflight-failed unit10 unexpectedly has output')
    if len(master['units'])!=96 or contract['reused_image_count']!=480 or contract['new_image_count']!=4128:raise ValueError('Original camera/image schedule must remain complete')
    for row in snapshot['complete_units']:
        if sha(Path(row['output'])/'COMPLETE.json')!=row['seal_sha256']:raise ValueError('Original copied seal changed')
        verify_seal(row['output'])
    return master,snapshot


def verify_publication(contract_path,commit):
    contract=read(contract_path)
    names=RENDER_SOURCES+[contract_path,contract['snapshot'],contract['schedule_replay']]
    for n in names:
        if subprocess.check_output(['git','show',commit+':'+str(n)])!=Path(n).read_bytes():raise ValueError('Completion source/protocol not source-published:'+str(n))
    for n in RENDER_SOURCES:
        if contract['render_source_sha256'][n]!=sha(n):raise ValueError('Frozen rendering dependency changed:'+n)
    return contract


def admit(contract,index,now):
    if contract['phase']!='missing_only_completion'or contract['completion_cap_seconds']!=50400 or contract['per_unit_cap_seconds']!=900:raise ValueError('Distinct fixed completion allowance required')
    if contract['original_deadline_met']or contract['cpu_threads']!=2 or contract['calibration_or_final_access']or contract['matcher_forwards_allowed']or contract['optimizer_updates_allowed']!=0:raise ValueError('Data-only completion contract required')
    deadline=datetime.fromisoformat(contract['completion_absolute_deadline_utc'])
    started=datetime.fromisoformat(contract['completion_contract_created_utc'])
    if (deadline-started).total_seconds()!=50400 or now>=deadline:raise TimeoutError('Original NEW completion deadline expired; never reset')
    if contract['factor_two_max_projection_seconds']+contract['copy_and_admission_margin_seconds']>50400:raise ValueError('Measured completion cost does not fit its fixed cap')
    if type(index)is not int or index not in contract['missing_unit_indices']:raise ValueError('Never render a reused complete unit')
    if sha(contract['original_master'])!=contract['original_master_sha256']:raise ValueError('Original camera schedule changed')
    master=read(contract['original_master']);unit=master['units'][index]
    if unit['site']['split']not in('training','model_selection')or unit['calibration_or_final_access']:raise ValueError('Reserve terrain forbidden')
    camera_replay(unit)
    relative=Path(unit['output']).relative_to(contract['original_root'])
    destination=Path(contract['new_output_root'])/relative
    return unit,destination,deadline


def make_contract(snapshot_path,schedule_replay_path,now=None,new_output_root='runs/solar-v68-completion'):
    """Prepare once after software QA; publish the resulting bytes before launch."""
    from datetime import timedelta
    now=datetime.now(timezone.utc)if now is None else now
    snapshot=read(snapshot_path);original=read(snapshot['original_master'])
    observed=[r['observed_outer_seconds']for r in snapshot['complete_units']]
    factor=2*len(snapshot['remaining_units'])*max(observed)
    failed=read(snapshot['original_failed_state']);failed_time=datetime.fromisoformat(failed['updated_utc'])
    if now<=datetime.fromisoformat(original['absolute_deadline_utc']):raise ValueError('This declaration records an expired original attempt; inspect clock before preparing')
    contract=dict(schema_version=68,phase='missing_only_completion',original_master=snapshot['original_master'],original_master_sha256=snapshot['original_master_sha256'],original_root='runs/solar-v67-development',original_failed_state=snapshot['original_failed_state'],original_failed_state_sha256=snapshot['original_failed_state_sha256'],snapshot=str(snapshot_path),snapshot_sha256=sha(snapshot_path),schedule_replay=str(schedule_replay_path),schedule_replay_sha256=sha(schedule_replay_path),original_shared_deadline_utc=original['absolute_deadline_utc'],original_deadline_met=False,completion_contract_created_utc=now.isoformat(),completion_absolute_deadline_utc=(now+timedelta(seconds=50400)).isoformat(),completion_cap_seconds=50400,per_unit_cap_seconds=900,cpu_threads=2,new_output_root=new_output_root,state_path='runs/solar-completion-v68-state.json',log_prefix='runs/solar-v68-unit-',reused_unit_indices=list(range(10)),missing_unit_indices=list(range(10,96)),reused_image_count=480,new_image_count=4128,factor_two_max_projection_seconds=factor,copy_and_admission_margin_seconds=2700,original_observed_unit_outer_seconds=sum(observed),original_observed_unit_internal_seconds=sum(r['internal_seconds']for r in snapshot['complete_units']),original_failed_preflight_seconds=None,calendar_gap_after_original_failure_seconds=(now-failed_time).total_seconds(),render_source_sha256={n:sha(n)for n in RENDER_SOURCES},calibration_or_final_access=False,matcher_forwards_allowed=False,optimizer_updates_allowed=0,paid_compute=False,physical_schedule_changed=False,original_failed_unit10_rendered_images=0,timing_scope='Original observed tenunit child-phase and internal intervals retained separately; failedpreflight exactdurationunknown. Postfailurecalendar gap is idle/external elapsed time, NOT chargedCPU or reclaimed oldbudget. Newcompletion intervals include copy/admission and86newunits. Not a successful original10hour attempt.',coverage_scope='Same original16training/eightselection terrain groups; fourcamera pairs/24lights are repeatedconditions, not new independent geography.',source_guard_scope='Exact rendering/admission dependencies only. Matcher, teacher and training files outside thisclosure may evolve during rendering; no renderingdependency may change.',camera_replay='Reuse audited north_camera_contract_v59: all randomdraws/translation/intrinsics exact; only3x3rotation basis permitsabs1e-12. Always render STORED originalcamera matrices.',blender_failure_policy='--python-exit-code3; nonzeroexit and exit0-without-complete-summary both fail. Preserve everypartialoutput/log, no automaticretry.',scope='Separately declared50400second data-only missingcompletion after originalv67failure/expiry. Copytencompleteunits byte-for-byte including theirseals and failedcases; renderonlyremaining86. Originalstate/log/deadline and NN48h/72hbudgets stayunchanged. Fullgeometry95%posecontrol audit before anylearning.')
    if factor+2700>50400:raise ValueError('Measuredfactor-two max cost andcopy/admissionmargin exceed fixedcompletion allowance')
    return contract
