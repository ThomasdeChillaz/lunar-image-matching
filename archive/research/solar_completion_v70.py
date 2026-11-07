"""Same-deadline, missing-only continuation after the preserved v68 seal failure.

The failed v68 payload and erroneous self-seal stay untouched. Existing unit10
is copied, verified byte-for-byte and correctly sealed in a NEW output root;
only units11..95 may be rendered. The original 14-hour clock never resets.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess
from solar_completion_v68 import (sha,read,verify_original,require_renderer_summary,
                                  verify_seal,verify_publication as verify_base_publication)
PARENT_SOURCES=['solar_completion_v70.py','run_solar_completion_v70.py']
EMPTY_SHA256=hashlib.sha256(b'').hexdigest()
UNCHANGED_BASE_KEYS=('original_master','original_master_sha256','original_root','original_failed_state',
 'original_failed_state_sha256','snapshot','snapshot_sha256','schedule_replay','schedule_replay_sha256',
 'original_shared_deadline_utc','original_deadline_met','completion_contract_created_utc',
 'completion_absolute_deadline_utc','completion_cap_seconds','per_unit_cap_seconds','cpu_threads',
 'reused_unit_indices','missing_unit_indices','reused_image_count','new_image_count',
 'factor_two_max_projection_seconds','copy_and_admission_margin_seconds','render_source_sha256',
 'calibration_or_final_access','matcher_forwards_allowed','optimizer_updates_allowed','paid_compute')


def payload_hashes(folder):
    folder=Path(folder)
    return {str(p.relative_to(folder)):sha(p)for p in sorted(folder.rglob('*'))if p.is_file()and p!=folder/'COMPLETE.json'}


def seal_payload(folder):
    """Hash the complete payload BEFORE opening the new seal; never self-hash."""
    folder=Path(folder);path=folder/'COMPLETE.json'
    if path.exists():raise FileExistsError('Never overwrite any existing seal')
    inventory=payload_hashes(folder)
    if 'COMPLETE.json'in inventory:raise ValueError('A seal must never include itself')
    with path.open('x')as f:json.dump(dict(sha256=inventory),f,indent=2);f.write('\n')
    return inventory


def verify_amendment(contract):
    snapshot_path=contract['v68_failure_snapshot'];snapshot=read(snapshot_path)
    if sha(snapshot_path)!=contract['v68_failure_snapshot_sha256']:raise ValueError('Frozen v68 failure snapshot changed')
    for key in('v68_contract','v68_failed_state'):
        if sha(snapshot[key])!=snapshot[key+'_sha256']:raise ValueError('Preserved v68 contract/state changed')
    base=read(snapshot['v68_contract']);failed=read(snapshot['v68_failed_state'])
    if failed['state']!='failed'or failed['completion_absolute_deadline_utc']!=base['completion_absolute_deadline_utc']:raise ValueError('Failed v68 deadline/failure must remain preserved')
    for key in UNCHANGED_BASE_KEYS:
        if contract[key]!=base[key]:raise ValueError('Amendment changes original completion contract:'+key)
    if contract['continuation_remaining_unit_indices']!=list(range(11,96))or contract['repaired_payload_unit_indices']!=[10]:raise ValueError('Only existing unit10 payload may be repaired; render only11..95')
    if Path(contract['new_output_root'])==Path(base['new_output_root'])or Path(contract['state_path'])==Path(base['state_path']):raise ValueError('Continuation needs NEW root/state; failed v68 cannot be overwritten')
    oldroot=Path(base['new_output_root']);actual={str(p.relative_to(oldroot))for p in oldroot.rglob('*')if p.is_file()}
    if actual!=set(snapshot['v68_output_files_sha256']):raise ValueError('Failed v68 root inventory changed')
    for n,h in snapshot['v68_output_files_sha256'].items():
        if sha(oldroot/n)!=h:raise ValueError('Failed v68 file changed:'+n)
    unit=Path(snapshot['unit10_output']);summary=require_renderer_summary(unit,snapshot['v68_contract_sha256'])
    if summary['original_unit_index']!=10:raise ValueError('Repair payload is not original unit10')
    bad=read(unit/'COMPLETE.json');payload=payload_hashes(unit)
    if set(bad['sha256'])!=set(payload)|{'COMPLETE.json'}or bad['sha256']['COMPLETE.json']!=EMPTY_SHA256:raise ValueError('Only the diagnosed empty self-hash defect is repairable')
    if any(bad['sha256'][n]!=h for n,h in payload.items())or payload!=snapshot['unit10_payload_sha256']:raise ValueError('Unit10 payload does not match every original rendered byte')
    if len(payload)!=54:raise ValueError('Existing complete48-image/geometry/labels/execution/summary payload required')
    if datetime.now(timezone.utc)>=datetime.fromisoformat(contract['completion_absolute_deadline_utc']):raise TimeoutError('Original v68 completion deadline expired; no continuation clock reset')
    return snapshot


def verify_publication(contract_path,commit):
    contract=verify_base_publication(contract_path,commit)
    for n in PARENT_SOURCES+[contract['v68_failure_snapshot']]:
        if subprocess.check_output(['git','show',commit+':'+str(n)])!=Path(n).read_bytes():raise ValueError('Continuation parent/amendment not source-published:'+str(n))
    for n in PARENT_SOURCES:
        if sha(n)!=contract['parent_source_sha256'][n]:raise ValueError('Frozen continuation source changed:'+n)
    verify_amendment(contract)
    return contract


def repair_copy_unit10(contract,contract_path,commit):
    snapshot=verify_amendment(contract);master,_=verify_original(contract)
    relative=Path(master['units'][10]['output']).relative_to(contract['original_root']);destination=Path(contract['new_output_root'])/relative
    if destination.exists():
        if not(destination/'COMPLETE.json').exists():raise FileExistsError('Partial repair copy preserved; no automatic overwrite')
        verify_seal(destination)
        provenance=read(destination/'REPAIRED_FROM_V68.json')
        if provenance['v68_failure_snapshot_sha256']!=contract['v68_failure_snapshot_sha256']or provenance['continuation_contract_sha256']!=sha(contract_path):raise ValueError('Existing repaired copy belongs to another admission')
        return provenance
    source=Path(snapshot['unit10_output'])
    shutil.copytree(source,destination,copy_function=shutil.copy2,
                    ignore=lambda folder,names:['COMPLETE.json']if Path(folder)==source else[])
    if payload_hashes(destination)!=snapshot['unit10_payload_sha256']:raise ValueError('Copied unit10 payload is not byte-identical')
    provenance=dict(unit_index=10,source_output=str(source),source_summary_sha256=sha(source/'summary.json'),
        preserved_bad_self_seal_sha256=sha(source/'COMPLETE.json'),v68_failure_snapshot_sha256=contract['v68_failure_snapshot_sha256'],
        continuation_contract_sha256=sha(contract_path),parent_publication_commit=commit,rendered_again=False,
        source_payload_files=54,images_copied=48,scope='Only original failed self-seal excluded from copied payload; all54payload bytes preserved. The badseal and complete failedv68root remain untouched.')
    with(destination/'REPAIRED_FROM_V68.json').open('x')as f:json.dump(provenance,f,indent=2);f.write('\n')
    seal_payload(destination);verify_seal(destination)
    return provenance


def make_contract(base_contract_path,snapshot_path,new_output_root='runs/solar-v70-completion',now=None):
    base=read(base_contract_path);snapshot=read(snapshot_path);now=datetime.now(timezone.utc)if now is None else now
    if sha(base_contract_path)!=snapshot['v68_contract_sha256']:raise ValueError('Snapshot/base contract differs')
    remaining=(datetime.fromisoformat(base['completion_absolute_deadline_utc'])-now).total_seconds()
    original_snapshot=read(base['snapshot']);projection=2*85*max(r['observed_outer_seconds']for r in original_snapshot['complete_units'])
    margin=1200
    if remaining<=projection+margin:raise TimeoutError('Remaining ORIGINAL clock cannot fit factor-two remaining projection and1200s copy/admission margin')
    c=dict(base,schema_version=70,new_output_root=new_output_root,state_path='runs/solar-completion-v70-state.json',log_prefix='runs/solar-v70-unit-',
        v68_failure_snapshot=str(snapshot_path),v68_failure_snapshot_sha256=sha(snapshot_path),
        repaired_payload_unit_indices=[10],continuation_remaining_unit_indices=list(range(11,96)),
        continuation_declared_utc=now.isoformat(),continuation_remaining_clock_seconds_at_declaration=remaining,
        continuation_factor_two_max_projection_seconds=projection,continuation_copy_admission_margin_seconds=margin,
        parent_source_sha256={n:sha(n)for n in PARENT_SOURCES},
        deadline_reset=False,v68_failure_preserved=True,unit10_rendered_again=False,
        continuation_scope='Missing-only continuation under ORIGINAL v68createdtime/50400cap/absolute deadline. Copy originalten immutable units, copy existingunit10payload/reseal innewroot, render ONLY11..95. Failedv68allfiles/badselfseal/state preserved. No time refund, paidcompute, matcherforward or optimizerupdate.')
    verify_amendment(c)
    return c
