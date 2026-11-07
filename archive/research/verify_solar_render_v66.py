"""Independent geometry and common-pose audit for the24-light cost pilot.

Scoring consumes query truth, which never becomes primary matcher input.
Original image/position files, failed attempts and deadlines remain unchanged.
"""
import argparse,json,hashlib
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
from verify_north_camera_qualified_v64 import project,qualify_projected_pixels,pose_errors,check_labels
from north_camera_visibility_v59 import first_mesh_hit,visibility_check
from native_baseline_pose_v7 import propose_pose
from lunar_geometry_v5 import median_reference_range_z

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(root,protocol_path,output):
    root=Path(root);p=json.loads(Path(protocol_path).read_text());s=json.loads((root/'summary.json').read_text());tick=perf_counter()
    if Path(output).exists():raise FileExistsError('Preserve previous audits')
    if not s['complete']or s['rendered_images']!=48 or len(s['images'])!=48 or s['matcher_forwards']or s['optimizer_updates']or s['training_allowed']:raise ValueError('Complete48frame inference-free cost pilot required')
    if s['protocol_sha256']!=sha(protocol_path)or p['phase']!='render_cost_pilot'or p['max_calendar_seconds']!=900:raise ValueError('Original published pilot contract changed')
    if sha(p['mesh'])!=p['mesh_sha256']:raise ValueError('Pinned terrain changed')
    with np.load(p['mesh'],allow_pickle=False)as z:vertices,faces=z['vertices'].copy(),z['faces'].copy()
    images=set()
    for row in s['images']:
        key=(row['role'],row['light']);images.add(key);path=root/row['path']
        if sha(path)!=row['sha256']:raise ValueError('Rendered image changed')
        with Image.open(path)as image:
            if image.size!=(512,512):raise ValueError('Native field of view changed')
    if images!={(role,l)for role in('reference','query')for l in range(24)}:raise ValueError('Unique complete24light two-camera inventory required')
    geometries={};views=[];pair=p['camera_pairs'][0]
    for role in('reference','query'):
        row=s['views'][role];path=root/row['path']
        if sha(path)!=row['sha256']:raise ValueError('Saved centre-ray geometry changed')
        with np.load(path,allow_pickle=False)as z:g={n:z[n].copy()for n in z.files}
        if not g['valid'].all()or g['valid'].shape!=(512,512):raise ValueError('Complete finite native surface required')
        if not np.array_equal(g['cam2world'],np.asarray(pair[role]['cam2world']))or not np.array_equal(g['intrinsics'],np.asarray(pair[role]['intrinsics'])):raise ValueError('Camera generation calibration changed')
        yy,xx=np.mgrid[:512,:512];q,z=project(g['position'].reshape(-1,3),g['intrinsics'],g['cam2world']);qualification=qualify_projected_pixels(q,np.column_stack([xx.ravel(),yy.ravel()]),z)
        if not np.allclose(z.reshape(512,512),g['axial_depth'],rtol=1e-10,atol=1e-6):raise ValueError('Independent axial-depth closure failed')
        errors=[]
        for point in g['position'][32:-32:128,32:-32:128].reshape(-1,3):
            ray=point-g['cam2world'][:3,3];distance=float(np.linalg.norm(ray));first=first_mesh_hit(g['cam2world'][:3,3],ray/distance,vertices,faces);errors.append(abs(first-distance))
        clearance=first_mesh_hit(g['cam2world'][:3,3],[0,0,-1],vertices,faces)
        views.append(dict(role=role,pixel_qualification=qualification,maximum_independent_mesh_error_m=float(max(errors)),camera_clearance_m=None if not np.isfinite(clearance)else float(clearance),passed=bool(qualification['qualified']and max(errors)<=.05 and np.isfinite(clearance)and clearance>10)))
        geometries[role]=g
    outcomes=[];visible_fraction=[];independent_visibility=[]
    for record in s['labels']:
        role=record['source'];other='query'if role=='reference'else'reference';path=root/record['path'];a,b=geometries[role],geometries[other]
        if sha(path)!=record['sha256']:raise ValueError('Exact3D source labels changed')
        with np.load(path,allow_pickle=False)as z:labels={n:z[n].copy()for n in z.files}
        visible_fraction.append(check_labels(labels,a['cam2world'],b['cam2world'],a['intrinsics'],b['intrinsics']))
        for kind in('visible','geometrically_occluded','visibility_uncertain'):
            indices=np.flatnonzero(labels[kind])
            if len(indices):
                i=int(indices[0]);check=visibility_check(b['cam2world'][:3,3],labels['world'][i],vertices,faces)
                if not check[kind]:raise ValueError('Independent full-mesh camera visibility differs')
                independent_visibility.append(dict(source=role,kind=kind,index=i,**check))
        visible=labels['visible'];pose,info=propose_pose(labels['source_points'][visible],labels['target_points'][visible],a['axial_depth'],a['intrinsics'],a['cam2world'],b['intrinsics'])
        error=pose_errors(pose,b['cam2world']);range_m=median_reference_range_z(a['axial_depth'],a['intrinsics'])
        outcomes.append(dict(source=role,pose_error=error,correct=error is not None and error['rotation_deg']<=2 and error['translation_m']<=.01*range_m,reference_range_m=range_m))
    rate=sum(x['correct']for x in outcomes)/len(outcomes);passed=all(x['passed']for x in views)and min(visible_fraction)>=.2 and rate>=.95
    result=dict(passed=passed,oracle_success=rate,oracle_pose_directions=len(outcomes),correct_oracle_poses=sum(x['correct']for x in outcomes),views=views,outcomes=outcomes,minimum_bidirectional_visible_fraction=min(visible_fraction),independent_camera_visibility=independent_visibility,source_sha256=sha(__file__),raw_summary_sha256=sha(root/'summary.json'),calendar_seconds=perf_counter()-tick,scope='ONE old excluded pilot terrain with fresh camera seed and24physical lights. Exact-correspondence oracle/geometry only, no matcher accuracy, learned gain, new geography or training permission.')
    with Path(output).open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);a=p.parse_args();audit(a.root,a.protocol,a.output)
