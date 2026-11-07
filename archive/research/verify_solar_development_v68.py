"""Full96unit development geometry gate, independently replayed after completion.

Every geometrically eligible camera direction remains in the pose denominator.
An oracle failure never becomes an eligibility exclusion. Query truth is audit
scoring only; no learned matcher, teacher target or reserve raster is accessed.
"""
import argparse,json,hashlib,time,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import Counter
import numpy as np
from PIL import Image
from verify_north_camera_qualified_v64 import project,qualify_projected_pixels,pose_errors,check_labels
from north_camera_visibility_v59 import first_mesh_hit,visibility_check
from native_baseline_pose_v7 import propose_pose
from lunar_geometry_v5 import median_reference_range_z
from solar_completion_v68 import verify_seal,verify_original

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
AUDIT_SOURCES=['verify_solar_development_v68.py','solar_completion_v68.py','verify_north_camera_qualified_v64.py',
 'native_baseline_pose_v7.py','localization_robust_pose_v4.py','localization_diagnostics_v3.py',
 'lunar_geometry_v5.py','lunar_geometry_v2.py','north_camera_visibility_v59.py','verify_north_development_v35.py']


def audit_unit(index,record,spec,mesh,clock,mesh_frame=None):
    root=Path(record['output']);s=verify_seal(root);vertices,faces=mesh
    if sha(root/'summary.json')!=record['summary_sha256']or sha(root/'COMPLETE.json')!=record['seal_sha256']:raise ValueError('Completion manifest summary/seal changed')
    if s['matcher_forwards']or s['optimizer_updates']or s['training_allowed']or s['calibration_or_final_access']:raise ValueError('Data-only complete unit required')
    if spec['site']['split']not in('training','model_selection'):raise ValueError('Reserve geometry is outside this audit')
    images=set()
    for row in s['images']:
        key=(row['role'],row['light'])
        if key in images:raise ValueError('Duplicate native view/light image')
        images.add(key);path=root/row['path']
        if sha(path)!=row['sha256']:raise ValueError('Rendered image changed')
        with Image.open(path)as image:
            if image.size!=(512,512):raise ValueError('Native image size differs')
    if images!={(role,l)for role in('reference','query')for l in range(24)}:raise ValueError('Both views/all24lights required')
    geometries={};views=[];pair=spec['camera_pairs'][0];reasons=[]
    for role in('reference','query'):
        clock();view=s['views'][role];path=root/view['path']
        if sha(path)!=view['sha256']:raise ValueError('Saved native geometry changed')
        with np.load(path,allow_pickle=False)as z:g={n:z[n].copy()for n in z.files}
        if mesh_frame is not None:
            for key,expected in zip(('body_origin','enu_rows'),mesh_frame):
                if key not in g or not np.array_equal(g[key],expected):raise ValueError('Reference body/local terrain frame metadata differs:'+key)
            rows=np.asarray(g['enu_rows']);origin=np.asarray(g['body_origin'])
            if rows.shape!=(3,3)or origin.shape!=(3,)or not np.isfinite(origin).all()or not np.allclose(rows@rows.T,np.eye(3),rtol=0,atol=1e-12)or not np.isclose(np.linalg.det(rows),1.,rtol=0,atol=1e-12):raise ValueError('Finite right-handed orthonormal ENU frame required')
        if g['valid'].shape!=(512,512)or g['valid'].dtype!=bool or g['position'].shape!=(512,512,3)or g['axial_depth'].shape!=(512,512):raise ValueError('Native geometry schema differs')
        if not np.array_equal(g['cam2world'],np.asarray(pair[role]['cam2world']))or not np.array_equal(g['intrinsics'],np.asarray(pair[role]['intrinsics'])):raise ValueError('Stored original camera/field of view changed')
        complete=bool(g['valid'].all());qualification=None;mesh_error=None
        if not complete:reasons.append(role+':incomplete_finite_surface')
        else:
            if not np.isfinite(g['position']).all():raise ValueError('Complete surface contains nonfinite coordinates')
            yy,xx=np.mgrid[:512,:512];q,z=project(g['position'].reshape(-1,3),g['intrinsics'],g['cam2world']);qualification=qualify_projected_pixels(q,np.column_stack([xx.ravel(),yy.ravel()]),z)
            if not qualification['qualified']:reasons.append(role+':native_pixel_centre_qualification_failed')
            if not np.allclose(z.reshape(512,512),g['axial_depth'],rtol=1e-10,atol=1e-6):raise ValueError('Independent axial-depth closure differs')
            errors=[]
            for point in g['position'][32:-32:128,32:-32:128].reshape(-1,3):
                clock();ray=point-g['cam2world'][:3,3];distance=float(np.linalg.norm(ray));first=first_mesh_hit(g['cam2world'][:3,3],ray/distance,vertices,faces);errors.append(abs(first-distance))
            mesh_error=float(max(errors))
            if not np.isfinite(mesh_error)or mesh_error>.05:raise ValueError('Independent native centre-ray/full-mesh intersection differs')
        clearance=first_mesh_hit(g['cam2world'][:3,3],[0,0,-1],vertices,faces);free=bool(np.isfinite(clearance)and clearance>10)
        if not free:reasons.append(role+':camera_not_certified_free_space')
        views.append(dict(role=role,body_local_frame_verified=mesh_frame is not None,complete_surface=complete,pixel_qualification=qualification,maximum_independent_mesh_error_m=mesh_error,camera_clearance_m=None if not np.isfinite(clearance)else float(clearance),free_space=free))
        geometries[role]=g
    label_records={r['source']:r for r in s['labels']}
    if len(s['labels'])!=2 or set(label_records)!={'reference','query'}:raise ValueError('Both exact correspondence directions required')
    visible_fraction=[];labels_by_role={};independent_visibility=[];occlusion=[]
    for role in('reference','query'):
        clock();other='query'if role=='reference'else'reference';a,b=geometries[role],geometries[other];record_label=label_records[role];path=root/record_label['path']
        if sha(path)!=record_label['sha256']:raise ValueError('Exact3D labels changed')
        with np.load(path,allow_pickle=False)as z:labels={n:z[n].copy()for n in z.files}
        visible_fraction.append(check_labels(labels,a['cam2world'],b['cam2world'],a['intrinsics'],b['intrinsics']));labels_by_role[role]=labels
        if int(labels['visible'].sum())!=record_label['visible']or len(labels['visible'])!=676:raise ValueError('Saved exact label count differs')
        for kind in('visible','geometrically_occluded','visibility_uncertain'):
            ids=np.flatnonzero(labels[kind])
            if len(ids):
                clock();i=int(ids[0]);check=visibility_check(b['cam2world'][:3,3],labels['world'][i],vertices,faces)
                if not check[kind]:raise ValueError('Independent full-mesh query visibility differs')
                independent_visibility.append(dict(source=role,kind=kind,index=i,**check))
        occlusion.append(dict(source=role,geometrically_occluded=int(labels['geometrically_occluded'].sum()),visibility_uncertain=int(labels['visibility_uncertain'].sum())))
    if min(visible_fraction)<.2:reasons.append('insufficient_bidirectional_visible_overlap')
    eligible=not reasons;outcomes=[]
    if eligible:
        for role in('reference','query'):
            clock();other='query'if role=='reference'else'reference';a,b=geometries[role],geometries[other];labels=labels_by_role[role];visible=labels['visible']
            pose,info=propose_pose(labels['source_points'][visible],labels['target_points'][visible],a['axial_depth'],a['intrinsics'],a['cam2world'],b['intrinsics'])
            error=pose_errors(pose,b['cam2world']);range_m=median_reference_range_z(a['axial_depth'],a['intrinsics'])
            outcomes.append(dict(source=role,pose_error=error,correct=error is not None and error['rotation_deg']<=2 and error['translation_m']<=.01*range_m,reference_range_m=range_m,pose_returned=pose is not None))
    approved=None
    if eligible:
        ref=s['views']['reference'];approved=dict(unit_index=index,component=spec['site']['component'],split=spec['site']['split'],reference_geometry_path=str(root/ref['path']),reference_geometry_sha256=ref['sha256'],scene_mesh_path=spec['mesh'],scene_mesh_sha256=spec['mesh_sha256'],scene_maximum_height_m=float(vertices[:,2].max()),scene_coverage='Original complete finite renderer terrain mesh; retain this ceiling when map coverage is cropped, not max(croppedmap).')
    return dict(approved_reference_input=approved,unit_index=index,component=spec['site']['component'],split=spec['site']['split'],family=pair['family'],pair_index=pair['pair_index'],eligible=eligible,qualification_reasons=reasons,source_output=str(root),source_summary_sha256=sha(root/'summary.json'),source_seal_sha256=sha(root/'COMPLETE.json'),views=views,minimum_bidirectional_visible_fraction=min(visible_fraction),independent_camera_visibility=independent_visibility,geometric_occlusion=occlusion,oracle_outcomes=outcomes)


def audit(dataset_summary,completion_contract,output,publication_commit,max_seconds=3600):
    import cv2
    cv2.setNumThreads(1);tick=time.monotonic();started=datetime.now(timezone.utc);deadline=started+timedelta(seconds=max_seconds)
    if max_seconds!=3600:raise ValueError('Predeclared one-hour full-dataset geometry audit allowance required')
    if Path(output).exists():raise FileExistsError('Preserve previous audits')
    for n in AUDIT_SOURCES:
        if subprocess.check_output(['git','show',publication_commit+':'+n])!=Path(n).read_bytes():raise ValueError('Executed audit source not source-published:'+n)
    contract=read(completion_contract);master,snapshot=verify_original(contract);data=read(dataset_summary)
    if not data['complete']or data['rendered_images']!=4608 or len(data['units'])!=96 or data['fitting_allowed']or not data['geometry_audit_pending']:raise ValueError('Complete raw96unit data-only completion required')
    if data['completion_contract_sha256']!=sha(completion_contract)or data['original_master_sha256']!=sha(contract['original_master']):raise ValueError('Completion/source schedule provenance differs')
    if {r['unit_index']for r in data['units']}!=set(range(96)):raise ValueError('Every original unit must occur once')
    def clock():
        if datetime.now(timezone.utc)>=deadline or time.monotonic()-tick>=max_seconds:raise TimeoutError('Original full-dataset geometry audit allowance exhausted')
    rows=[];component=None;mesh=None
    try:
        for record in sorted(data['units'],key=lambda x:x['unit_index']):
            clock();index=record['unit_index'];spec=master['units'][index];name=spec['site']['component']
            if name!=component:
                if sha(spec['mesh'])!=spec['mesh_sha256']:raise ValueError('Declared terrain mesh changed')
                with np.load(spec['mesh'],allow_pickle=False)as z:mesh=(z['vertices'].copy(),z['faces'].copy());mesh_frame=(z['body_origin'].copy(),z['enu_rows'].copy())
                component=name
            row=audit_unit(index,record,spec,mesh,clock,mesh_frame);rows.append(row)
            progress=dict(complete_dataset=False,completed_units=len(rows),expected_units=96,units=rows,passed=False,original_master_sha256=sha(contract['original_master']),completion_summary_sha256=sha(dataset_summary),audit_source_sha256=sha(__file__))
            Path(str(output)+'.progress.json').write_text(json.dumps(progress,indent=2,allow_nan=False)+'\n')
            print('GEOMETRY AUDIT',len(rows),'/96','eligible',row['eligible'],'oracle',[x['correct']for x in row['oracle_outcomes']],flush=True)
        outcomes=[x for r in rows for x in r['oracle_outcomes']];correct=sum(x['correct']for x in outcomes);rate=correct/len(outcomes)if outcomes else 0.
        if len(outcomes)!=2*sum(r['eligible']for r in rows):raise ValueError('Every eligible camera direction must remain in oracle denominator')
        groups={r['component']:r['split']for r in rows};counts=dict(Counter(groups.values()));eligible_groups={r['component']for r in rows if r['eligible']};terrain_complete=eligible_groups==set(groups)
        if counts!={'training':16,'model_selection':8}:raise ValueError('Original sixteen/eight terrain groups required')
        result=dict(passed=bool(terrain_complete and rate>=.95),complete_dataset=True,expected_units=96,completed_units=96,terrain_counts=counts,terrain_coverage_complete=terrain_complete,eligible_unit_indices=[r['unit_index']for r in rows if r['eligible']],qualified_pair_ids=[r['component']+'/'+r['family']+'-'+str(r['pair_index'])for r in rows if r['eligible']],approved_reference_inputs={r['component']+'/'+r['family']+'-'+str(r['pair_index']):r['approved_reference_input']for r in rows if r['eligible']},eligible_units=sum(r['eligible']for r in rows),oracle_success=rate,oracle_pose_directions=len(outcomes),correct_oracle_poses=correct,units=rows,original_master_sha256=sha(contract['original_master']),master_protocol_sha256=sha(contract['original_master']),completion_summary_path=str(dataset_summary),completion_summary_sha256=sha(dataset_summary),completion_contract_sha256=sha(completion_contract),audit_source_sha256=sha(__file__),source_sha256={n:sha(n)for n in AUDIT_SOURCES},publication_commit=publication_commit,maximum_native_pixel_centre_error_px=.02,minimum_bidirectional_visible_fraction=.2,minimum_camera_clearance_m=10.,maximum_independent_mesh_error_m=.05,minimum_oracle_success=.95,calendar_seconds=time.monotonic()-tick,photometric_pixels_decoded=0,matcher_forwards=0,optimizer_updates=0,calibration_or_final_access=False,pose_failure_never_excludes_eligible_unit=True,scope='Full original96camera pairs/24developmentgroups under24physical lighting conditions. Exact-label/common-reference-depth pose oracle and independent mesh/nativecentre/visibility checks. Extra conditions are not independent terrain; no matcher accuracy or model improvement established.')
        with Path(output).open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        return result
    except BaseException as exc:
        with Path(str(output)+'.FAILED.json').open('x')as f:json.dump(dict(error_type=type(exc).__name__,error=str(exc),completed_units=len(rows),expected_units=96,passed=False,calendar_seconds=time.monotonic()-tick),f,indent=2)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-summary',required=True);p.add_argument('--completion-contract',required=True);p.add_argument('--output',required=True);p.add_argument('--publication-commit',required=True);a=p.parse_args();r=audit(a.dataset_summary,a.completion_contract,a.output,a.publication_commit);print(json.dumps({k:v for k,v in r.items()if k not in('units','source_sha256')},indent=2))
