"""Lineage-preserving completion of the failed v59 camera pilot.

Reuse every complete pair, including failures. Never copy incomplete pairs or
interpret missing old per-image timing as zero render time. No matcher here.
"""
from pathlib import Path
import hashlib,json,shutil
from datetime import datetime

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def inventory(folder):
    folder=Path(folder)
    if any(p.is_symlink()for p in folder.rglob('*')):
        raise ValueError('Symlinked prior data are not admissible')
    return {str(p.relative_to(folder)):sha(p)for p in sorted(folder.rglob('*'))if p.is_file()}


def completed_pairs(prior,snapshot,schedule):
    """Authenticate the full original attempt before reusing any complete pair."""
    prior=Path(prior)
    if inventory(prior)!=snapshot['partial_files_sha256']:
        raise ValueError('Original partial attempt differs from its terminal snapshot')
    state=snapshot['terminal_controller_state']
    if state['state']!='failed' or state['error_type']!='TimeoutError' or state['error']!='Original fixed camera phase allowance exhausted:render':
        raise ValueError('Actual original rendering timeout required')
    if sha(prior/'protocol.json')!=snapshot['original_protocol_sha256']:
        raise ValueError('Original protocol differs')
    protocol=read(prior/'protocol.json')
    if (protocol['max_calendar_seconds']!=1800 or protocol['schedule_sha256']!=sha(schedule)
            or protocol['publication_commit']!=snapshot['source_publication_commit']):
        raise ValueError('Original frozen camera schedule/source/budget differs')
    schedule=read(schedule);expected={p['component']+'/'+p['family']+'-'+str(p['pair_index']):p for p in schedule['pairs']}
    records={}
    for key,spec in expected.items():
        folder=prior/key;path=folder/'summary.json'
        if not path.exists():continue
        row=read(path)
        if (row['component']!=spec['component']or row['pair_id']!=key.split('/')[1]
                or row['family']!=spec['family']or row['pair_index']!=spec['pair_index']
                or set(row['views'])!={'reference','query'}):
            raise ValueError('Completed pair metadata differ from original schedule')
        for role in ('reference','query'):
            geometry=prior/row['views'][role]['path']
            if geometry!=folder/role/'geometry.npz' or sha(geometry)!=row['views'][role]['sha256']:
                raise ValueError('Original completed camera geometry differs')
            for light in range(6):
                if not(folder/role/f'light-{light}'/'image.png').is_file():
                    raise ValueError('Completed pair lacks a declared image')
        for label in row['labels']:
            p=prior/label['path']
            if p.parent!=folder or sha(p)!=label['sha256']:
                raise ValueError('Original completed labels differ')
        records[key]=dict(component=spec['component'],pair_id=row['pair_id'],files_sha256=inventory(folder))
    if (len(records)!=snapshot['completed_camera_pairs']
            or sorted(records)!=sorted(snapshot['completed_pair_ids'])):
        raise ValueError('All original completed pairs must be reused without selection')
    return records


def copy_pair(source,destination,record):
    source,destination=Path(source),Path(destination)
    if destination.exists():raise FileExistsError('Preserve every completion attempt')
    if inventory(source)!=record['files_sha256']:
        raise ValueError('Prior complete pair bytes changed')
    shutil.copytree(source,destination)
    if inventory(destination)!=record['files_sha256']:
        raise ValueError('Copied pair is not byte-identical')
    return read(destination/'summary.json')


def reused_images(root,row):
    root=Path(root);result=[]
    for role in ('reference','query'):
        for light in range(6):
            name=Path(row['component'])/row['pair_id']/role/f'light-{light}'/'image.png'
            result.append(dict(component=row['component'],pair_id=row['pair_id'],role=role,light=light,
                image=str(name),sha256=sha(root/name),seconds=None,reused_from_prior_attempt=True,
                timing_scope='Original per-image timing was not durably saved; unknown, never zero.'))
    return result


def verify_completion_lineage(root,contract_path):
    """Authenticate copied bytes, missing-only generation and separate budgets."""
    root=Path(root);contract=read(contract_path);p=read(root/'protocol.json');s=read(root/'summary.json')
    snapshot=read(contract['snapshot_path']);prior=Path(contract['prior_output'])
    if sha(contract['prior_failed_state'])!=contract['prior_failed_state_sha256'] or read(contract['prior_failed_state'])!=snapshot['terminal_controller_state']:
        raise ValueError('Original terminal controller evidence changed')
    if sha(contract['snapshot_path'])!=contract['snapshot_sha256']:
        raise ValueError('Original terminal snapshot changed')
    records=completed_pairs(prior,snapshot,contract['schedule_path'])
    if records!=contract['complete_pairs']:
        raise ValueError('Frozen complete-pair inventory differs')
    if (p['completion_contract_sha256']!=sha(contract_path)or p['original_v59_deadline_met']
            or p['max_calendar_seconds']!=contract['completion_cap_seconds']
            or not 0<=s['calendar_seconds']<=contract['completion_cap_seconds']):
        raise ValueError('Separate completion clock or original failure was relabelled')
    for key,record in records.items():
        if inventory(root/key)!=record['files_sha256']:
            raise ValueError('Completed original pair was modified or rerendered')
    rows=s['images'];keys=[(r['component'],r['pair_id'],r['role'],r['light'])for r in rows]
    if len(keys)!=576 or len(set(keys))!=576:raise ValueError('Exactly576 unique image outputs required')
    reused=[r for r in rows if r.get('reused_from_prior_attempt',False)]
    fresh=[r for r in rows if not r.get('reused_from_prior_attempt',False)]
    for r in rows:
        expected_name=Path(r['component'])/r['pair_id']/r['role']/f"light-{r['light']}"/'image.png'
        if r['role']not in('reference','query')or r['light']not in range(6)or r['image']!=str(expected_name):
            raise ValueError('Image path differs from its native camera/light identity')
        if sha(root/r['image'])!=r['sha256']:raise ValueError('Completion image changed')
        key=r['component']+'/'+r['pair_id'];old=key in records
        if old!=bool(r.get('reused_from_prior_attempt',False)):
            raise ValueError('Completed-pair images must be reused and missing-pair images generated')
        if old and r['seconds']is not None:raise ValueError('Unknown old render timing must stay unknown')
        if not old and not(0<r['seconds']<contract['completion_cap_seconds']):
            raise ValueError('New image render timing is invalid')
    original_protocol=read(prior/'protocol.json')
    observed_original=(datetime.fromisoformat(snapshot['terminal_controller_state']['updated_utc'])-datetime.fromisoformat(original_protocol['created_utc'])).total_seconds()
    if not 0<observed_original<1900:raise ValueError('Original observed render interval invalid')
    if (len(reused)!=s['reused_images']or len(fresh)!=s['newly_rendered_images']
            or len(fresh)!=contract['new_image_count']or len(reused)!=contract['reused_image_count']
            or abs(s['original_render_program_observed_seconds']-observed_original)>1e-6
            or abs(s['sum_observed_render_program_intervals_seconds']-(observed_original+s['calendar_seconds']))>1e-6):
        raise ValueError('Original/new/combined render accounting differs')
    return dict(passed=True,original_attempt_deadline_met=False,reused_pairs=len(records),
        reused_images=len(reused),newly_rendered_images=len(fresh),
        completion_calendar_seconds=s['calendar_seconds'],
        sum_observed_render_program_intervals_seconds=s['sum_observed_render_program_intervals_seconds'],
        timing_limit='Sum of original observed render-program interval and supplemental program interval, excluding admissions/imports and idle time between attempts. Not total CPU cost; outer fixed-deadline receipts retained separately.',
        original_per_image_timings_unknown=True,all_reused_bytes_exact=True)
