"""Even-light 3D patch-bank construction for compact lighting-response learning.

Query pose and visibility are training labels. Canonical image resampling is
shared with inference. Neither brightness nor teacher feature validity creates
camera/solar visibility labels. Odd training PNGs are never opened or hashed.
"""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from lunar_geometry_v2 import camera_check
from lunar_geometry_v5 import project_z
from solar_native_v63 import canonical_patch, state_digest
from solar_physics_v63 import lighting_schedule
from north_render_plan_v31 import RADIUS,plan
from north_terrain_pilot_v32 import grid_faces

EVEN=tuple(range(0,24,2))
ROLES=('reference','query')
ARRAY_KEYS={'patches','conditions','target_suns','teacher','available','observed_lights'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


@dataclass(frozen=True)
class GeometryGate:
    master:dict
    dataset:dict
    audit:dict
    eligible_training_indices:tuple
    training_groups:tuple
    original_master_sha256:str
    dataset_summary_sha256:str
    geometry_audit_sha256:str


def verify_geometry_gate(master_path,dataset_summary_path,audit_path):
    """Full-dataset geometry gate is checked before any model/image decoding."""
    master,dataset,audit=map(read,(master_path,dataset_summary_path,audit_path))
    master_digest=sha(master_path);dataset_digest=sha(dataset_summary_path)
    if not dataset.get('complete')or len(dataset.get('units',[]))!=96:raise ValueError('All96 rendered units required before teacher extraction')
    if dataset.get('original_master_sha256',dataset.get('master_protocol_sha256'))!=master_digest:raise ValueError('Completed data must retain the frozen original master')
    if not audit.get('passed')or not audit.get('complete_dataset')or audit.get('expected_units')!=96 or audit.get('completed_units')!=96 or audit.get('terrain_counts')!={'training':16,'model_selection':8}:raise ValueError('Independent full96unit geometry audit required')
    if audit.get('original_master_sha256')!=master_digest or audit.get('completion_summary_sha256')!=dataset_digest:raise ValueError('Geometry audit/data lineage changed')
    if not .95<=audit.get('oracle_success',0)<=1:raise ValueError('At least95%exact-correspondence pose recovery required')
    if len(master.get('units',[]))!=96 or master.get('calibration_or_final_access')or dataset.get('calibration_or_final_access',False):raise ValueError('Original development-only master required')
    expected_sites={s['component']:s for s in plan()['sites']if s['split']in('training','model_selection')}
    training_groups=tuple(sorted(c for c,s in expected_sites.items()if s['split']=='training'))
    ds={r['unit_index']:r for r in dataset['units']};ar={r['unit_index']:r for r in audit.get('units',[])}
    if len(ds)!=96 or len(ar)!=96 or set(ds)!=set(range(96))or set(ar)!=set(ds):raise ValueError('No duplicated, dropped or substituted camera units')
    counts={};eligible=[]
    for i,spec in enumerate(master['units']):
        site=spec['site'];component=site['component'];split=site['split']
        if component not in expected_sites or split!=expected_sites[component]['split']:raise ValueError('Reserve or substituted terrain in training input metadata')
        counts[component]=counts.get(component,0)+1
        record,checked=ds[i],ar[i];folder=Path(record['output'])
        if sha(folder/'summary.json')!=record['summary_sha256']or sha(folder/'COMPLETE.json')!=record['seal_sha256']:raise ValueError('Immutable complete-unit metadata changed')
        if checked['source_summary_sha256']!=record['summary_sha256']or checked['source_seal_sha256']!=record['seal_sha256']or checked['component']!=component or checked['split']!=split:raise ValueError('Audit unit provenance differs')
        summary=read(folder/'summary.json')
        if not summary.get('complete')or summary.get('rendered_images')!=48 or len(summary.get('images',[]))!=48 or summary.get('matcher_forwards')or summary.get('optimizer_updates'):raise ValueError('Complete inference-free48image units required')
        inventory={(r['role'],r['light'])for r in summary['images']}
        if inventory!={(role,l)for role in ROLES for l in range(24)}:raise ValueError('Complete unique physical24light inventory required')
        if type(checked['eligible'])is not bool:raise ValueError('Explicit independent geometric eligibility required')
        if split=='training'and checked['eligible']:eligible.append(i)
    if set(counts)!=set(expected_sites)or set(counts.values())!={4}:raise ValueError('Four generated camera pairs per original24groups required')
    eligible_directions=2*sum(r['eligible']for r in ar.values())
    if audit.get('oracle_pose_directions')!=eligible_directions or not np.isclose(audit.get('correct_oracle_poses',-1)/max(eligible_directions,1),audit['oracle_success'],rtol=0,atol=1e-12):raise ValueError('Every eligible positive pose direction must remain in oracle denominator')
    if {master['units'][i]['site']['component']for i in eligible}!=set(training_groups):raise ValueError('At least one geometrically eligible pair in every training group required')
    return GeometryGate(master,dataset,audit,tuple(eligible),training_groups,master_digest,dataset_digest,sha(audit_path))


class EvenImageReader:
    """The only training PNG entry point; parity is checked BEFORE file access."""
    def __init__(self,folder,image_records,read_bytes=None):
        self.folder=Path(folder).resolve();self.records={(r['role'],r['light']):r for r in image_records};self._read_bytes=read_bytes or(lambda p:p.read_bytes());self.cache={};self.decoded=[]
        if len(self.records)!=len(image_records):raise ValueError('Duplicate image metadata')
    def get(self,role,light):
        if role not in ROLES or type(light)is not int or light not in EVEN:raise ValueError('Heldout/odd training illumination access forbidden')
        key=(role,light)
        if key not in self.cache:
            record=self.records[key];path=(self.folder/record['path']).resolve()
            if not path.is_relative_to(self.folder):raise ValueError('Image path escapes frozen unit')
            payload=self._read_bytes(path)
            if hashlib.sha256(payload).hexdigest()!=record['sha256']:raise ValueError('Even training image changed after audit')
            with Image.open(io.BytesIO(payload))as image:
                if image.size!=(512,512):raise ValueError('Native512px field of view required')
                a=np.asarray(image.convert('L')).copy()
            if a.dtype!=np.uint8 or a.shape!=(512,512):raise ValueError('Native uint8 grayscale training image required')
            self.cache[key]=a;self.decoded.append(dict(role=role,light=light,path=record['path'],sha256=record['sha256']))
        return self.cache[key]


def local_facets(points,vertices_grid,rows,columns,tolerance_m=.05):
    """Normals of supporting nearby facets; check plane and barycentric support."""
    p=np.asarray(points,float);g=np.asarray(vertices_grid,float);rows=np.asarray(rows,int);columns=np.asarray(columns,int)
    if p.ndim!=2 or p.shape[1]!=3 or g.ndim!=3 or g.shape[2]!=3 or g.shape[0]!=g.shape[1]or rows.shape!=(len(p),)or columns.shape!=(len(p),):raise ValueError('Structured terrain, point and cell arrays required')
    result=[];records=[];n=len(g)
    for point,row,col in zip(p,rows,columns):
        candidates=[]
        for r in range(max(0,row-1),min(n-1,row+2)):
            for c in range(max(0,col-1),min(n-1,col+2)):
                a,b,d,e=g[r,c],g[r,c+1],g[r+1,c],g[r+1,c+1]
                for triangle_index,triangle in enumerate(((a,b,d),(b,e,d))):
                    a,b,c0=triangle;u=b-a;v=c0-a;normal=np.cross(u,v);length=np.linalg.norm(normal)
                    if not np.isfinite(length)or length<1e-10:continue
                    normal/=length;distance=float(np.dot(point-a,normal));foot=point-distance*normal
                    dot00,dot01,dot11=np.dot(u,u),np.dot(u,v),np.dot(v,v);relative=foot-a;den=dot00*dot11-dot01*dot01
                    if den<=1e-15:continue
                    s=(dot11*np.dot(relative,u)-dot01*np.dot(relative,v))/den;t=(dot00*np.dot(relative,v)-dot01*np.dot(relative,u))/den;bary=np.array([1-s-t,s,t])
                    if np.min(bary)>=-1e-4 and abs(distance)<=tolerance_m:
                        if normal[2]<0:normal=-normal
                        candidates.append((abs(distance),r,c,triangle_index,normal,bary))
        if not candidates:raise ValueError('Exact ray-labelled point has no supporting terrain facet within.05m')
        distance,r,c,tri,normal,bary=min(candidates,key=lambda x:x[:4]);result.append(normal);records.append(dict(row=int(r),column=int(c),triangle=int(tri),plane_error_m=float(distance),minimum_barycentric=float(bary.min())))
    return np.asarray(result).reshape(-1,3),records


def exact_surface_normals(points,mesh,site,offsets):
    offsets=np.asarray(offsets,float);vertices=np.asarray(mesh['vertices']);faces=np.asarray(mesh['faces']);n=len(offsets)
    if vertices.shape!=(n*n,3)or faces.shape!=(2*(n-1)**2,3):raise ValueError('Original complete structured terrain required')
    expected=grid_faces(n)
    if not np.array_equal(np.sort(faces,axis=1),np.sort(expected,axis=1)):raise ValueError('Missing/substituted terrain facets')
    points=np.asarray(points,float);body=points@np.asarray(mesh['enu_rows'])+np.asarray(mesh['body_origin']);radii=np.linalg.norm(body,axis=1)
    longitude=np.arctan2(body[:,1],body[:,0]);latitude=np.arcsin(np.clip(body[:,2]/radii,-1,1));rho=2*RADIUS*np.tan((np.pi/2-latitude)/2)
    x=rho*np.sin(longitude);y=-rho*np.cos(longitude)
    columns=np.searchsorted(offsets,(x-site['x'])/10,side='right')-1;rows=np.searchsorted(offsets,(site['y']-y)/10,side='right')-1
    return local_facets(points,vertices.reshape(n,n,3),rows,columns)


def unit_geometries(folder,summary,spec):
    result={}
    for role in ROLES:
        meta=summary['views'][role];path=Path(folder)/meta['path']
        if sha(path)!=meta['sha256']:raise ValueError('Audited reference/query training geometry changed')
        with np.load(path,allow_pickle=False)as z:g={n:z[n].copy()for n in('axial_depth','intrinsics','cam2world','body_origin','enu_rows','valid')}
        g['intrinsics'],g['cam2world']=camera_check(g['intrinsics'],g['cam2world'])
        if g['axial_depth'].shape!=(512,512)or g['valid'].shape!=(512,512)or g['valid'].dtype!=bool or not g['valid'].all()or not np.isfinite(g['axial_depth']).all()or (g['axial_depth']<=0).any():raise ValueError('Full finite native axial-depth geometry required')
        if not np.array_equal(g['cam2world'],np.asarray(spec['camera_pairs'][0][role]['cam2world']))or not np.array_equal(g['intrinsics'],np.asarray(spec['camera_pairs'][0][role]['intrinsics'])):raise ValueError('Frozen camera schedule differs')
        result[role]=g
    label_record=next(r for r in summary['labels']if r['source']=='reference');path=Path(folder)/label_record['path']
    if sha(path)!=label_record['sha256']:raise ValueError('Exact3D visibility labels changed')
    with np.load(path,allow_pickle=False)as z:labels={k:z[k].copy()for k in z.files}
    return result,labels,label_record['sha256']


def spatial_order(points):
    """Round-robin4x4spatial cells, deterministic within each cell."""
    p=np.asarray(points);codes=np.floor(p[:,1]/128).astype(int)*4+np.floor(p[:,0]/128).astype(int);groups=[np.flatnonzero(codes==c).tolist()for c in sorted(set(codes))]
    return [group[j]for j in range(max(map(len,groups),default=0))for group in groups if j<len(group)]


def sample_geometry(geometry,labels,maximum_points,normal_provider):
    from solar_frontend_v63 import reference_anchors
    if type(maximum_points)is not int or not 1<=maximum_points<=256:raise ValueError('Published one-to256native anchor budget required')
    ref,query=geometry['reference'],geometry['query'];p=np.asarray(labels['source_points'],float);q=np.asarray(labels['target_points'],float);world=np.asarray(labels['world'],float)
    if p.shape!=(676,2)or q.shape!=p.shape or world.shape!=(676,3):raise ValueError('Exact676point source ray labels required')
    valid=np.asarray(labels['source_valid']);inside=np.asarray(labels['in_frame']);visible=np.asarray(labels['visible']);occluded=np.asarray(labels['geometrically_occluded']);uncertain=np.asarray(labels['visibility_uncertain'])
    if any(v.shape!=(676,)or v.dtype!=bool for v in(valid,inside,visible,occluded,uncertain)):raise ValueError('Explicit independent geometric visibility masks required')
    if (inside&~valid).any()or (visible.astype(int)+occluded.astype(int)+uncertain.astype(int)!=inside.astype(int)).any():raise ValueError('Camera visibility/occlusion/unknown must partition geometric in-frame support')
    projected,z,front=project_z(world[valid],ref['intrinsics'],ref['cam2world'])
    if not front.all()or np.max(np.linalg.norm(projected-p[valid],axis=1))>.02:raise ValueError('Exact source world/pixel convention differs')
    projected,z,front=project_z(world[visible],query['intrinsics'],query['cam2world'])
    if not front.all()or np.max(np.linalg.norm(projected-q[visible],axis=1))>.02:raise ValueError('Exact query world/pixel convention differs')
    allowed={tuple(a):i for i,a in enumerate(p)};anchor_ids=[allowed[tuple(a)]for a in reference_anchors()]
    candidates=[i for i in anchor_ids if valid[i]and visible[i]and not occluded[i]and not uncertain[i]]
    zeros=np.zeros((512,512),np.uint8);accepted=[];support_failures=[]
    for index in spatial_order(p[candidates]):
        i=candidates[index]
        patches,meta=canonical_patch(zeros,zeros,p[i],ref['axial_depth'],ref['intrinsics'],ref['cam2world'],query['intrinsics'],query['cam2world'],query_center=q[i])
        if patches is None:support_failures.append(i);continue
        accepted.append(dict(label_index=i,reference_xy=p[i],query_xy=q[i],world=world[i],reference_map=meta['reference_xy'],query_map=meta['query_xy']))
        if len(accepted)==maximum_points:break
    if not accepted:raise ValueError('No co-visible physical anchor with complete canonical support')
    normals,normal_records=normal_provider(np.stack([r['world']for r in accepted]))
    normals=np.asarray(normals,float)
    if len(normal_records)!=len(accepted)or normals.shape!=(len(accepted),3)or not np.isfinite(normals).all()or not np.allclose(np.linalg.norm(normals,axis=1),1,atol=1e-6):raise ValueError('Exact unit reference-surface normals required')
    for row,normal,record in zip(accepted,normals,normal_records):row.update(normal=normal,normal_provenance=record)
    counts=dict(exact_source_points=676,proposal_anchors=256,co_visible_anchor_candidates=len(candidates),selected_physical_points=len(accepted),canonical_support_failures=support_failures,camera_occluded_points=int(occluded.sum()),camera_visibility_uncertain_points=int(uncertain.sum()),unknown_patch_interior_visibility=True)
    return accepted,counts


def training_trial_pose(query_pose,rng,translation_sigma_m=0.,rotation_sigma_deg=0.):
    """Optional predeclared geometry-error augmentation, never a lighting label."""
    import cv2
    if min(translation_sigma_m,rotation_sigma_deg)<0 or not np.isfinite([translation_sigma_m,rotation_sigma_deg]).all():raise ValueError('Nonnegative finite trial-pose perturbations required')
    out=np.asarray(query_pose,float).copy();out[:3,3]+=rng.normal(0,translation_sigma_m,3)
    rot=rng.normal(0,np.radians(rotation_sigma_deg),3);out[:3,:3]=out[:3,:3]@cv2.Rodrigues(rot)[0]
    return out


def remap_patch(image,xy):
    import cv2
    return cv2.remap(image,np.asarray(xy[:,:,0],np.float32),np.asarray(xy[:,:,1],np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)


def encode_teacher(encoder,patches,batch_size=64,clock=lambda:None):
    import torch
    if type(batch_size)is not int or not 1<=batch_size<=128:raise ValueError('Bounded teacher batch required')
    a=np.asarray(patches)
    if a.ndim!=3 or a.shape[1:]!=(33,33)or a.dtype!=np.uint8:raise ValueError('Canonical native33pxuint8teacher patches required')
    features=[]
    for start in range(0,len(a),batch_size):
        clock()
        with torch.inference_mode():
            dense=encoder(torch.from_numpy(a[start:start+batch_size].copy()).float()[:,None]/255)
            dense=dense[0]if isinstance(dense,tuple)else dense
            if dense.shape!=(len(a[start:start+batch_size]),32,33,33):raise ValueError('Frozen32Dnative descriptor teacher required')
            value=dense[:,:,16,16].cpu().numpy().copy()
        if not np.isfinite(value).all():raise ValueError('Nonfinite frozen teacher outputs')
        features.append(value)
    return np.concatenate(features)if features else np.empty((0,32),np.float32)


def build_unit_arrays(geometry,points,reader,latitude_deg,encoder,seed,batch_size=64,trial_jitter=None,clock=lambda:None,lighting_records=None):
    """All actual teacher extraction goes through this even-image reader."""
    ref,query=geometry['reference'],geometry['query'];r=ref['cam2world'][:3,:3];n=len(points)
    expected_lights=lighting_schedule(latitude_deg)
    lights=expected_lights if lighting_records is None else lighting_records
    if len(lights)!=24 or [a['index']for a in lights]!=list(range(24)):raise ValueError('Frozen physical24light index schedule required')
    world_suns=np.asarray([a['sun_enu']for a in lights],float)
    if not np.isfinite(world_suns).all()or not np.allclose(world_suns,np.asarray([a['sun_enu']for a in expected_lights]),rtol=0,atol=1e-10):raise ValueError('Rendered physical lighting metadata differs from lunar geometry')
    suns=world_suns@r
    rng=np.random.default_rng(seed);jitter=trial_jitter or{};trial=training_trial_pose(query['cam2world'],rng,**jitter)
    world=np.stack([p['world']for p in points]);normal=np.stack([p['normal']for p in points])@r
    ref_view=ref['cam2world'][:3,3]-world;ref_view/=np.linalg.norm(ref_view,axis=1,keepdims=True)
    query_view=trial[:3,3]-world;query_view/=np.linalg.norm(query_view,axis=1,keepdims=True)
    views=np.stack([ref_view@r,query_view@r],1)
    teacher=np.full((n,2,24,32),np.nan,np.float32);available=np.zeros((n,2,24),bool)
    patches=np.empty((n,2,33,33),np.uint8);conditions=np.empty((n,2,9),np.float32)
    indices=(np.arange(n)+int(seed)%12)%12;observed=np.column_stack([np.asarray(EVEN)[indices],np.asarray(EVEN)[(indices+5+np.arange(n)//12)%12]])
    teacher_patches=[];teacher_keys=[];zero_norm=[]
    for light in EVEN:
        clock();images=[reader.get(role,light)for role in ROLES]
        for i,point in enumerate(points):
            for role,image,key in zip(ROLES,images,('reference_map','query_map')):
                teacher_patches.append(remap_patch(image,point[key]));teacher_keys.append((i,ROLES.index(role),light))
    encoded=encode_teacher(encoder,np.stack(teacher_patches),batch_size,clock)
    for key,feature in zip(teacher_keys,encoded):
        norm=float(np.linalg.norm(feature))
        if norm<=1e-8:zero_norm.append(list(key));continue
        if not np.isclose(norm,1.,rtol=0,atol=2e-4):raise ValueError('Nonzero teacher descriptors must be unit normalized')
        teacher[key]=feature;available[key]=True
    for i,point in enumerate(points):
        clock();ref_image=reader.get('reference',int(observed[i,0]));query_image=reader.get('query',int(observed[i,1]))
        pair,meta=canonical_patch(ref_image,query_image,point['reference_xy'],ref['axial_depth'],ref['intrinsics'],ref['cam2world'],query['intrinsics'],trial,query_center=point['query_xy'])
        if pair is None:raise ValueError('Declared trial-pose augmentation loses canonical image support')
        patches[i]=pair
        for role in range(2):conditions[i,role]=np.concatenate([suns[observed[i,role]],normal[i],views[i,role]])
    # Inputs whose observed descriptors/paired common counterfactual targets
    # are unavailable are omitted as unusable descriptor samples, NEVER relabelled
    # as camera occlusions or solar shadows.
    use=available.any(-1).all(-1)&(available[:,0]&available[:,1]).any(-1)
    for role in range(2):use&=available[np.arange(n),role,observed[:,role]]
    if not use.any():raise ValueError('No paired sample with usable frozen teacher targets')
    result=dict(patches=patches[use],conditions=conditions[use],target_suns=np.broadcast_to(suns,(n,2,24,3)).astype(np.float32).copy()[use],teacher=teacher[use],available=available[use],observed_lights=observed[use].astype(np.int64))
    if np.isfinite(result['teacher'][:,:,1::2]).any()or result['available'][:,:,1::2].any()or any(r['light']not in EVEN for r in reader.decoded):raise ValueError('Heldout light leakage detected')
    details=dict(selected_indices=np.flatnonzero(use).tolist(),omitted_unusable_teacher_samples=int((~use).sum()),zero_norm_teacher_targets=zero_norm,even_images_decoded=reader.decoded,heldout_image_reads=0,query_training_trial_pose=trial.tolist(),trial_pose_jitter=jitter,teacher_patch_count=len(teacher_patches),geometry_labels_independent_of_brightness=True,teacher_availability_is_not_geometric_visibility=True)
    return result,details
