"""Source-published, budgeted construction of the even-light3Dteacher bank.

Run a small extraction-cost pilot AFTER the complete dataset's independent
geometry gate. The full bank then includes every geometrically eligible training
camera pair. No model-selection, calibration, final or odd training PNG is read.
A completed record cache is immutable and keyed by weights, geometry, source,
canonicalization, selection settings and software versions. Partial caches are
preserved and never overwritten automatically.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from solar_training_bank_v68 import (EVEN,ARRAY_KEYS,EvenImageReader,verify_geometry_gate,
    unit_geometries,exact_surface_normals,sample_geometry,build_unit_arrays,read,sha)
from solar_native_v63 import state_digest

SOURCES=('build_solar_training_bank_v68.py','solar_training_bank_v68.py',
    'solar_native_v63.py','solar_physics_v63.py','solar_frontend_v63.py',
    'lunar_geometry_v5.py','lunar_geometry_v2.py','compact_lunar.py',
    'contrastive_terrain.py','north_render_plan_v31.py','north_terrain_pilot_v32.py',
    'solar_training_v63.py','solar_compute_budget_v67.py','verify_solar_development_v68.py')


def write_new(path,value):
    with Path(path).open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def verify_published(protocol_path,protocol,commit):
    if len(commit)!=40 or any(c not in'0123456789abcdef'for c in commit):raise ValueError('Exact public source commit required')
    for name in SOURCES:
        if name not in protocol['source_sha256']or sha(name)!=protocol['source_sha256'][name]or subprocess.check_output(['git','show',commit+':'+name])!=Path(name).read_bytes():raise ValueError('Bank source differs from published protocol:'+name)
    if subprocess.check_output(['git','show',commit+':'+str(protocol_path)])!=Path(protocol_path).read_bytes():raise ValueError('Bank protocol was not published before extraction')


def cache_key(spec,unit_record,audit_sha,teacher_checkpoint_sha,teacher_training_sha,teacher_descriptor_sha,configuration,environment,sources):
    record=dict(version=68,component=spec['site']['component'],split=spec['site']['split'],source_unit_index=unit_record['unit_index'],
        summary_sha256=unit_record['summary_sha256'],seal_sha256=unit_record['seal_sha256'],mesh_sha256=spec['mesh_sha256'],geometry_audit_sha256=audit_sha,
        teacher_checkpoint_sha256=teacher_checkpoint_sha,teacher_training_sha256=teacher_training_sha,teacher_descriptor_sha256=teacher_descriptor_sha,
        selection=configuration,environment=environment,source_sha256=sources)
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest(),record


def validate_record_arrays(arrays):
    if set(arrays)!=ARRAY_KEYS:raise ValueError('Canonical bank array schema differs')
    n=len(arrays['patches'])
    if n<1:raise ValueError('Nonempty canonical teacher samples required')
    expected={'patches':(n,2,33,33),'conditions':(n,2,9),'target_suns':(n,2,24,3),'teacher':(n,2,24,32),'available':(n,2,24),'observed_lights':(n,2)}
    for name,shape in expected.items():
        if arrays[name].shape!=shape:raise ValueError('Bank shape differs:'+name)
    if arrays['patches'].dtype!=np.uint8 or arrays['available'].dtype!=bool or not np.issubdtype(arrays['observed_lights'].dtype,np.integer)or any(arrays[k].dtype!=np.float32 for k in('conditions','target_suns','teacher'))or not np.isin(arrays['observed_lights'],EVEN).all():raise ValueError('Even observed lights and explicit native patch/availability dtypes required')
    if not np.isfinite(arrays['conditions']).all()or not np.isfinite(arrays['target_suns']).all():raise ValueError('Finite physical conditioning required')
    if not np.allclose(arrays['target_suns'][:,0],arrays['target_suns'][:,1],rtol=0,atol=1e-6):raise ValueError('Both canonical observations require same target Sun frame')
    for start in(0,3,6):
        if not np.allclose(np.linalg.norm(arrays['conditions'][:,:,start:start+3],axis=-1),1,rtol=0,atol=1e-5):raise ValueError('Unit Sun, normal and viewing directions required')
    if not np.allclose(np.linalg.norm(arrays['target_suns'],axis=-1),1,rtol=0,atol=1e-5):raise ValueError('Unit target Sun vectors required')
    a=arrays['available'];t=arrays['teacher']
    if a[:,:,1::2].any()or np.isfinite(t[:,:,1::2]).any()or np.isfinite(t[~a]).any()or not np.isfinite(t[a]).all():raise ValueError('Unavailable or odd teacher targets must remain absentNaN')
    if not a.any(-1).all()or not(a[:,0]&a[:,1]).any(-1).all():raise ValueError('Paired common counterfactual targets required')
    return n


def completed_cache(root,key,expected_metadata):
    folder=Path(root)/'records';seal=folder/(key+'.COMPLETE.json')
    if not seal.exists():return None
    sealed=read(seal);payload=folder/(key+'.npz');meta_path=folder/(key+'.json')
    if sha(payload)!=sealed['npz_sha256']or sha(meta_path)!=sealed['metadata_sha256']:raise ValueError('Completed teacher cache changed')
    meta=read(meta_path)
    if meta['input_key']!=key or meta['input']!=expected_metadata:raise ValueError('Checkpoint/source-sensitive cache provenance differs')
    with np.load(payload,allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
    count=validate_record_arrays(arrays)
    if meta['samples']!=count or count<8:raise ValueError('Complete cache sample count differs or is inadequate for contrastive sampling')
    return payload,meta_path,seal,meta


def save_record(root,key,arrays,metadata):
    folder=Path(root)/'records';folder.mkdir(exist_ok=True);payload=folder/(key+'.npz');meta=folder/(key+'.json');seal=folder/(key+'.COMPLETE.json')
    if any(p.exists()for p in(payload,meta,seal)):raise FileExistsError('Existing complete/partial teacher record is preserved')
    validate_record_arrays(arrays)
    with payload.open('xb')as stream:np.savez_compressed(stream,**arrays)
    write_new(meta,metadata);write_new(seal,dict(npz_sha256=sha(payload),metadata_sha256=sha(meta)))
    return payload,meta,seal


def run(protocol_path,publication_commit):
    import cv2,torch
    from compact_lunar import load_checkpoint
    from solar_compute_budget_v67 import reserve
    protocol_path=Path(protocol_path);p=read(protocol_path);root=Path(p['output']);tick=time.monotonic();started=datetime.now(timezone.utc)
    if p.get('prepared_only',False):raise ValueError('Protocol template cannot launch teacher extraction')
    if p['phase']not in('cost_pilot','full_bank')or p['cpu_threads']!=2 or type(p['max_calendar_seconds'])is not int or not 0<p['max_calendar_seconds']<=3600:raise ValueError('Bounded two-thread bank extraction protocol required')
    deadline=datetime.fromisoformat(p['absolute_deadline_utc'])
    if deadline<=started:raise TimeoutError('Original published bank extraction deadline expired; never reset')
    def clock():
        if datetime.now(timezone.utc)>=deadline or time.monotonic()-tick>=p['max_calendar_seconds']:raise TimeoutError('Original bank extraction deadline exhausted')
    for name in('original_master','dataset_summary','geometry_audit'):
        if sha(p[name])!=p[name+'_sha256']:raise ValueError('Frozen bank input changed:'+name)
    # The full data control precedes teacher construction and any image decoder.
    gate=verify_geometry_gate(p['original_master'],p['dataset_summary'],p['geometry_audit'])
    if gate.audit['audit_source_sha256']!=p['source_sha256']['verify_solar_development_v68.py']:raise ValueError('Independent geometry audit source changed')
    if sha(p['lighting_schedule'])!=p['lighting_schedule_sha256']or p['lighting_schedule_sha256']!=gate.master['source_schedule_sha256']:raise ValueError('Original rendered lighting schedule changed')
    frozen_lighting=read(p['lighting_schedule'])
    verify_published(protocol_path,p,publication_commit)
    if root.exists()or Path(p['manifest']).exists():raise FileExistsError('Preserve prior bank attempts/manifests; use versioned output and immutable complete caches')
    teacher_dir=Path(p['teacher_directory'])
    if sha(teacher_dir/'selected.pt')!=p['teacher_checkpoint_sha256']or sha(teacher_dir/'training.json')!=p['teacher_training_sha256']:raise ValueError('Original frozen teacher checkpoint changed')
    if p['seed']!=20261101 or type(p['maximum_points_per_unit'])is not int or not 8<=p['maximum_points_per_unit']<=256:raise ValueError('Fixed data selection seed/native anchor budget required')
    if p['phase']=='cost_pilot'and p.get('cache_roots'):raise ValueError('Extraction cost pilot must run actual fresh teacher forwards')
    environment=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,torch=torch.__version__,platform=platform.platform(),processor=platform.processor(),torch_num_threads=2)
    if p['phase']=='full_bank':
        pilot=read(p['cost_pilot'])
        if sha(p['cost_pilot'])!=p['cost_pilot_sha256']or not pilot['complete']or pilot['phase']!='cost_pilot'or pilot['heldout_image_reads']!=0 or pilot['teacher_checkpoint_sha256']!=p['teacher_checkpoint_sha256']:raise ValueError('Actual immutable even-light extraction pilot required')
        if pilot['fresh_teacher_records']!=4 or pilot['reused_complete_records']or len(pilot['records'])!=4 or pilot['geometry_gate_sha256']!=gate.geometry_audit_sha256 or pilot['source_sha256']!={name:p['source_sha256'][name]for name in SOURCES}or pilot['environment']!=environment or pilot['teacher_batch_size']!=p['teacher_batch_size']or pilot['trial_pose_jitter']!=p.get('trial_pose_jitter',{}):raise ValueError('Measured pilot must match source/data/hardware/teacher settings and use fresh forwards')
        minimum_pilot_points=min(r['selected_physical_points']for r in pilot['records'])
        measured_projection=2*pilot['calendar_seconds']*len(gate.eligible_training_indices)/4*p['maximum_points_per_unit']/minimum_pilot_points
        if not np.isclose(p['factor_two_cost_projection_seconds'],measured_projection,rtol=0,atol=1e-6)or measured_projection>p['max_calendar_seconds']:raise ValueError('Measured full-bank extraction does not fit original cap')
    reservation=reserve(p['compute_ledger'],p['compute_ledger_sha256'],p['reservation_journal'],sha(protocol_path)+'/teacher_bank','training',p['max_calendar_seconds'])
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    teacher,training_metadata=load_checkpoint(teacher_dir);teacher.eval().requires_grad_(False)
    if teacher.reliability:raise ValueError('Original expanded32Dteacher without reliability head required')
    descriptor_sha=state_digest(teacher)
    if descriptor_sha!=p['expected_frozen_descriptor_sha256']:raise ValueError('Loaded teacher weights differ from expected descriptor state')
    root.mkdir(parents=True);write_new(root/'execution.json',dict(protocol_sha256=sha(protocol_path),publication_commit=publication_commit,started_utc=started.isoformat(),absolute_deadline_utc=deadline.isoformat(),compute_reservation=reservation,teacher_descriptor_sha256=descriptor_sha,environment=environment))
    records=[];source_manifest={r['unit_index']:r for r in gate.dataset['units']};all_indices=list(gate.eligible_training_indices)
    if p['phase']=='cost_pilot':
        # Four predetermined training groups, never the fastest measured subset.
        indices=[min(i for i in all_indices if gate.master['units'][i]['site']['component']==c)for c in gate.training_groups[:4]]
    else:indices=all_indices
    mesh_cache={};fresh_encoder_records=0;cached_encoder_records=0
    try:
        for index in indices:
            clock();spec=gate.master['units'][index];record=source_manifest[index];folder=Path(record['output']);component=spec['site']['component']
            unit_seed=int(hashlib.sha256(f'solar-bank-v68/{p["seed"]}/{index}'.encode()).hexdigest()[:8],16)
            configuration=dict(maximum_points_per_unit=p['maximum_points_per_unit'],seed=unit_seed,teacher_batch_size=p['teacher_batch_size'],trial_pose_jitter=p.get('trial_pose_jitter',{}),source_lights=list(EVEN),lighting_schedule_sha256=p['lighting_schedule_sha256'])
            key,input_metadata=cache_key(spec,record,gate.geometry_audit_sha256,p['teacher_checkpoint_sha256'],p['teacher_training_sha256'],descriptor_sha,configuration,environment,{name:p['source_sha256'][name]for name in SOURCES})
            hit=None
            for cache_root in p.get('cache_roots',[]):
                hit=completed_cache(cache_root,key,input_metadata)
                if hit is not None:break
            if hit is not None:
                payload,meta_path,seal,metadata=hit;target=root/'records';target.mkdir(exist_ok=True)
                for path in(payload,meta_path,seal):
                    destination=target/path.name
                    if destination.exists():raise FileExistsError('Preserve existing target cache')
                    shutil.copy2(path,destination)
                payload,meta_path,seal=(target/path.name for path in(payload,meta_path,seal));cached_encoder_records+=1
            else:
                summary=read(folder/'summary.json');geometry,labels,label_sha=unit_geometries(folder,summary,spec)
                mesh_path=Path(spec['mesh'])
                if sha(mesh_path)!=spec['mesh_sha256']:raise ValueError('Original terrain mesh changed')
                if component not in mesh_cache:
                    with np.load(mesh_path,allow_pickle=False)as z:mesh_cache={component:{k:z[k].copy()for k in('vertices','faces','body_origin','enu_rows')}}
                mesh=mesh_cache[component]
                if any(not np.array_equal(geometry[role]['body_origin'],mesh['body_origin'])or not np.array_equal(geometry[role]['enu_rows'],mesh['enu_rows'])for role in('reference','query')):raise ValueError('Reference/query training geometry and terrain body frames differ')
                from north_render_plan_v31 import plan
                offsets=np.asarray(plan()['mesh_native_offset_indices'])
                points,geometry_counts=sample_geometry(geometry,labels,p['maximum_points_per_unit'],lambda x:exact_surface_normals(x,mesh,spec['site'],offsets))
                reader=EvenImageReader(folder,summary['images'])
                arrays,details=build_unit_arrays(geometry,points,reader,spec['site']['centre_latitude_deg'],teacher,unit_seed,p['teacher_batch_size'],p.get('trial_pose_jitter'),clock,lighting_records=frozen_lighting['lighting'][component])
                samples=validate_record_arrays(arrays)
                if samples<8:raise ValueError('Fewer than8distinct paired physical samples; do not hide training-data shortfall')
                kept=[points[i]for i in details['selected_indices']]
                physical=[dict(label_index=int(row['label_index']),reference_xy=row['reference_xy'].tolist(),query_xy=row['query_xy'].tolist(),world=row['world'].tolist(),normal=row['normal'].tolist(),normal_provenance=row['normal_provenance'])for row in kept]
                metadata=dict(input_key=key,input=input_metadata,unit_index=index,component=component,split='training',family=spec['camera_pairs'][0]['family'],pair_index=spec['camera_pairs'][0]['pair_index'],samples=samples,source_lights=list(EVEN),label_sha256=label_sha,physical_points=physical,geometry_counts=geometry_counts,teacher_extraction=details,canonicalization_identical_to_inference=True,reference_cam2world=geometry['reference']['cam2world'].tolist(),query_training_truth_cam2world=geometry['query']['cam2world'].tolist(),query_truth_is_training_label_only=True)
                payload,meta_path,seal=save_record(root,key,arrays,metadata);fresh_encoder_records+=1
            records.append(dict(unit_index=index,component=component,split='training',family=metadata['family'],pair_index=metadata['pair_index'],path=str(payload),sha256=sha(payload),metadata_path=str(meta_path),metadata_sha256=sha(meta_path),record_seal_path=str(seal),record_seal_sha256=sha(seal),source_lights=list(EVEN),samples=metadata['samples'],selected_physical_points=metadata['geometry_counts']['selected_physical_points'],cache_key=key,reused_complete_cache=hit is not None))
            print('RESPONSE TEACHER BANK UNIT',index,'points',metadata['samples'],'cached',hit is not None,flush=True)
        clock();groups=sorted({r['component']for r in records});counts={g:sum(r['component']==g for r in records)for g in groups}
        summary=dict(complete=True,phase=p['phase'],records=records,groups=groups,eligible_pairs_per_group=counts,fresh_teacher_records=fresh_encoder_records,reused_complete_records=cached_encoder_records,calendar_seconds=time.monotonic()-tick,protocol_sha256=sha(protocol_path),teacher_checkpoint_sha256=p['teacher_checkpoint_sha256'],teacher_descriptor_sha256=descriptor_sha,heldout_image_reads=0,geometry_gate_sha256=gate.geometry_audit_sha256,maximum_points_per_unit=p['maximum_points_per_unit'],teacher_batch_size=p['teacher_batch_size'],trial_pose_jitter=p.get('trial_pose_jitter',{}),environment=environment,source_sha256={name:p['source_sha256'][name]for name in SOURCES},lighting_schedule_sha256=p['lighting_schedule_sha256'],calibration_or_final_pixels_loaded=0,model_selection_pixels_loaded=0,scope='Training-only exact3Dcanonical patch bank and frozen teacher targets. Not a trained response model, matcher result or additional independent terrain.')
        if p['phase']=='full_bank':
            if set(groups)!=set(gate.training_groups)or {r['unit_index']for r in records}!=set(gate.eligible_training_indices):raise ValueError('Every eligible pair in all16training groups must contribute')
            manifest=dict(schema_version=63,split='training',lighting_indices=list(EVEN),heldout_image_reads=0,teacher_descriptor_sha256=descriptor_sha,expected_frozen_descriptor_sha256=p['expected_frozen_descriptor_sha256'],canonicalization_identical_to_inference=True,terrain_groups=list(gate.training_groups),geometry_gate_passed=True,geometry_audit_path=p['geometry_audit'],geometry_audit_sha256=gate.geometry_audit_sha256,original_master_sha256=gate.original_master_sha256,dataset_summary_sha256=gate.dataset_summary_sha256,records=records,producer_protocol_sha256=sha(protocol_path),producer_completion_path=str(root/'COMPLETE.json'),producer_source_sha256=p['source_sha256'],teacher_checkpoint_sha256=p['teacher_checkpoint_sha256'],sampling='Uniform originalterrain, then uniform eligiblecamera pair, then uniquephysicalsamples;4generatedcamera pairs/group, geometriceligibility counts explicit.',scope=summary['scope'])
            write_new(p['manifest'],manifest)
            from solar_training_v63 import TrainingBank
            TrainingBank(p['manifest'])
            summary['manifest_sha256']=sha(p['manifest'])
        write_new(root/'summary.json',summary);write_new(root/'COMPLETE.json',dict(complete=True,protocol_sha256=sha(protocol_path),manifest_sha256=summary.get('manifest_sha256'),teacher_descriptor_sha256=descriptor_sha,summary_sha256=sha(root/'summary.json'),record_seals={r['cache_key']:r['record_seal_sha256']for r in records}));return summary
    except BaseException as exc:
        write_new(root/'FAILED.json',dict(error_type=type(exc).__name__,error=str(exc),complete_records=len(records),calendar_seconds=time.monotonic()-tick,partial_outputs_preserved=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--publication-commit',required=True);a=p.parse_args();print(json.dumps(run(a.protocol,a.publication_commit),indent=2))
