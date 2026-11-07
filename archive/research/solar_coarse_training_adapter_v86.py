"""UNQUALIFIED coarse128 teacher adapter; source/toy controls, never a job.

Targets are native-view FULL128 frozen-descriptor samples, not relabelled v79
canonical33 teachers. Query truth is accepted only by this TRAINING module.
No Torch import/model construction occurs at import. Actual teacher extraction
requires a future independently published, source-pinned and budgeted job.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


EVEN=tuple(range(0,24,2))
ROLES=('reference','query')
FORMAT='distributional-coarse128-teacher-v86'
NORMAL_METHOD='reference-axial-depth-bilinear-coarse-centres-difference-estimate-v86'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_to_coarse(points):
    p=np.asarray(points,np.float64)
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():
        raise ValueError('Finite native Nx2 xy required')
    return ((p+.5)/4-.5).astype(np.float32)


def _unit(v):
    v=np.asarray(v,np.float64);norm=np.linalg.norm(v,axis=-1,keepdims=True)
    if not np.isfinite(v).all()or (norm<1e-10).any():raise ValueError('Finite nonzero direction required')
    return v/norm


def _camera(geometry):
    from lunar_geometry_v2 import camera_check
    k,c=camera_check(geometry['intrinsics'],geometry['cam2world'])
    d=np.asarray(geometry['axial_depth'],np.float64);valid=np.asarray(geometry['valid'])
    if d.shape!=(512,512)or valid.shape!=d.shape or valid.dtype!=bool:
        raise ValueError('Native512 axial depth and explicit bool reference support required')
    return k,c,d,valid&np.isfinite(d)&(d>0)


def coarse_support(native_valid,points128,geometry_valid=None,radius=12):
    """All area-resize pixels, radius12 and four descriptor neighbours valid.

    This is image/map support, NOT camera visibility or a sunlight label.
    """
    m=np.asarray(native_valid)
    if m.shape!=(512,512)or m.dtype!=bool:raise ValueError('Explicit native512 bool support required')
    if radius!=12:raise ValueError('Frozen descriptor has declared radius12')
    p=np.asarray(points128,np.float64)
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('Finite coarse Nx2 required')
    low=m.reshape(128,4,128,4).all(axis=(1,3))
    if geometry_valid is not None:
        g=np.asarray(geometry_valid)
        if g.shape!=(128,128)or g.dtype!=bool:raise ValueError('Coarse bool geometry support required')
        low&=g
    supported=cv2.erode(low.astype(np.uint8),np.ones((25,25),np.uint8),
                        borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    lo=np.floor(p).astype(np.int64);hi=np.ceil(p).astype(np.int64)
    inside=((lo>=0)&(hi<128)).all(1)
    result=np.zeros(len(p),bool)
    ids=np.flatnonzero(inside)
    if len(ids):
        a,b=lo[ids],hi[ids]
        result[ids]=supported[a[:,1],a[:,0]]&supported[a[:,1],b[:,0]]&supported[b[:,1],a[:,0]]&supported[b[:,1],b[:,0]]
    return result


def reference_normal_field(geometry):
    """Known-reference-only ESTIMATED normals, expressed in reference camera.

    Bilinear axial depth at native1.5+4*i, central world differences internally
    and one-sided edge differences. Invalid depth/stencil cells are masked.
    These are not exact triangle normals; exact point normals remain separate.
    Identical declared procedure is required by any future inference adapter.
    """
    k,c,d,valid=_camera(geometry)
    yy,xx=np.mgrid[:128,:128];p=np.stack([1.5+4*xx,1.5+4*yy],-1)
    lo=np.floor(p).astype(int);x,y=lo[...,0],lo[...,1]
    good=valid[y,x]&valid[y,x+1]&valid[y+1,x]&valid[y+1,x+1]
    safe=np.where(valid,d,0.)
    z=(safe[y,x]+safe[y,x+1]+safe[y+1,x]+safe[y+1,x+1])*.25
    rays=np.concatenate([p,np.ones((128,128,1))],-1)@np.linalg.inv(k).T
    world=(rays*z[...,None])@c[:3,:3].T+c[:3,3]
    dx=np.gradient(world,axis=1);dy=np.gradient(world,axis=0)
    support=good.copy()
    support[:,1:-1]&=good[:,:-2]&good[:,2:]
    support[1:-1]&=good[:-2]&good[2:]
    support[:,0]&=good[:,1];support[:,-1]&=good[:,-2]
    support[0]&=good[1];support[-1]&=good[-2]
    normal=np.cross(dx,dy);length=np.linalg.norm(normal,axis=-1)
    support&=np.isfinite(normal).all(-1)&(length>1e-10)
    normal/=np.where(length>1e-10,length,1.)[...,None]
    normal=np.where((normal[...,2]<0)[...,None],-normal,normal)
    camera_normal=normal@c[:3,:3]
    camera_normal[~support]=0.
    return dict(normal_xy=np.moveaxis(camera_normal[...,:2],-1,0).astype(np.float32),
                valid=support,method=NORMAL_METHOD,exact_facet_normals=False,
                uses_query_geometry=False)


def training_view_basis(world,reference_pose,target_pose):
    """True target view labels; never callable as primary inference inputs."""
    from lunar_geometry_v2 import camera_check
    _,ref=camera_check(np.eye(3),reference_pose)
    _,target=camera_check(np.eye(3),target_pose)
    x=np.asarray(world,np.float64)
    if x.ndim!=2 or x.shape[1]!=3:raise ValueError('Exact scene-frame Nx3 world labels required')
    los=_unit(target[:3,3]-x)@ref[:3,:3]
    # Camera convention is right/DOWN/forward, so up is NEGATIVE column1.
    up=np.broadcast_to(-target[:3,1]@ref[:3,:3],los.shape)
    return np.column_stack([np.ones(len(x)),los,up]).astype(np.float32)


def validate_metadata_inventory(bank,metadata,eligible_unit_indices,training_groups):
    """Validate complete retained point inventory without decoding its arrays."""
    if bank.get('split')!='training'or bank.get('lighting_indices')!=list(EVEN)or bank.get('heldout_image_reads')!=0:
        raise ValueError('Training-only even-light bank required')
    records=bank.get('records',[])
    if len(metadata)!=len(records)or len(records)!=len(eligible_unit_indices):raise ValueError('All eligible original camera units required')
    if set(r['unit_index']for r in records)!=set(eligible_unit_indices):raise ValueError('Missing, duplicate or foreign camera unit')
    allowed=set(training_groups);seen=set()
    for r,m in zip(records,metadata):
        if (m.get('split')!='training'or r.get('split')!='training'or m.get('component')not in allowed or r.get('component')!=m['component']
                or m.get('unit_index')!=r['unit_index']or m.get('source_lights')!=list(EVEN)
                or m.get('samples')!=len(m.get('physical_points',[]))):raise ValueError('Camera/point/split metadata differs')
        t=m['teacher_extraction']
        if t.get('heldout_image_reads')!=0 or len(t['even_images_decoded'])!=24 or {(a['role'],a['light'])for a in t['even_images_decoded']}!={(role,i)for role in ROLES for i in EVEN}:
            raise ValueError('Both roles and every even illumination required; odd access forbidden')
        seen.add(m['component'])
    if seen!=allowed:raise ValueError('Every original training terrain group must contribute')
    return True


def load_training_metadata_index(root,bank_manifest,master,dataset_summary,audit):
    """Future real adapter entry: existing independent full96 gate FIRST.

    Gate's original relative paths require execution from repository root.
    This function reads JSON only, not bank NPZs, RGB or model checkpoints.
    """
    root=Path(root).resolve()
    if Path.cwd().resolve()!=root:raise ValueError('Run the unchanged geometry gate from repository root')
    from solar_training_bank_v68 import verify_geometry_gate
    gate=verify_geometry_gate(root/master,root/dataset_summary,root/audit)
    bank=json.loads((root/bank_manifest).read_text())
    if (bank.get('geometry_audit_sha256')!=gate.geometry_audit_sha256
            or bank.get('original_master_sha256')!=gate.original_master_sha256
            or bank.get('dataset_summary_sha256')!=gate.dataset_summary_sha256):raise ValueError('Bank/full geometry lineage differs')
    metadata=[]
    for record in bank['records']:
        path=(root/record['metadata_path']).resolve()
        if not path.is_relative_to(root)or sha(path)!=record['metadata_sha256']:raise ValueError('Pinned training metadata changed')
        metadata.append(json.loads(path.read_text()))
    validate_metadata_inventory(bank,metadata,gate.eligible_training_indices,gate.training_groups)
    return bank,metadata,gate


def frozen_coarse_teacher(encoder,image128,points128,*,expected_descriptor_sha256):
    """Future actual forward: FULL128 original encoder, same padding as fine.

    No patch crop, normal channels, rotation, oracle pose, or new learned model.
    Called only after independent scientific admission by a future job runner.
    """
    a=np.asarray(image128);p=np.asarray(points128,np.float32)
    if a.shape!=(128,128)or a.dtype!=np.uint8:raise ValueError('Full128 uint8 image required')
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('Finite coarse target coordinates required')
    if type(expected_descriptor_sha256)is not str or len(expected_descriptor_sha256)!=64 or any(c not in '0123456789abcdef'for c in expected_descriptor_sha256):raise ValueError('Pinned original descriptor SHA256 required')
    import torch
    from contrastive_terrain import sample_descriptors
    from solar_native_v63 import state_digest
    if (getattr(encoder,'reliability',True)or sum(v.numel()for v in encoder.parameters())!=28272
            or encoder.training or any(v.requires_grad or v.device.type!='cpu'for v in encoder.parameters())
            or torch.get_num_threads()>2):raise ValueError('Frozen original28272-parameter CPU teacher and at most2threads required')
    if state_digest(encoder)!=expected_descriptor_sha256:raise ValueError('Original teacher state changed or wrong descriptor supplied')
    with torch.inference_mode():
        dense,_=encoder(torch.from_numpy(a.copy()).float()[None,None]/255)
        if dense.shape!=(1,32,128,128):raise ValueError('Full same-padding32D teacher grid required')
        return sample_descriptors(dense,torch.from_numpy(p.copy())[None])[0].cpu().numpy().copy()


def build_coarse_training_unit(metadata,geometry,even_reader,lighting_records,teacher_function,*,reference_log_scale=0.):
    """Source-only future unit builder, injectable teacher for toy controls.

    Retains every legacy point and camera, even if no new target is usable.
    New support/zero descriptors get NaN targets+explicit omissions, NOT wrong
    correspondence, camera-occlusion, brightness or solar-shadow labels.
    """
    from lunar_geometry_v5 import project_z
    m=metadata
    if m.get('split')!='training'or m.get('source_lights')!=list(EVEN):raise ValueError('Training-only unit and exact12even lights required')
    if not np.isfinite(reference_log_scale):raise ValueError('Explicit finite known reference scale required')
    if len(lighting_records)!=24 or [a['index']for a in lighting_records]!=list(range(24)):raise ValueError('Frozen24 physical light records required')
    suns_world=np.asarray([a['sun_enu']for a in lighting_records],np.float64)
    if suns_world.shape!=(24,3)or not np.allclose(np.linalg.norm(suns_world,axis=1),1.,atol=1e-6,rtol=0):raise ValueError('Unit physical Sun vectors required')
    rows=m['physical_points'];n=len(rows)
    if n<1:raise ValueError('Retained historical point inventory required')
    world=np.asarray([a['world']for a in rows],np.float64)
    normal=np.asarray([a['normal']for a in rows],np.float64)
    if world.shape!=(n,3)or normal.shape!=(n,3)or not np.isfinite(world).all()or not np.isfinite(normal).all()or not np.allclose(np.linalg.norm(normal,axis=1),1.,atol=1e-6,rtol=0):raise ValueError('Finite exact world labels and unit source facet normals required')
    native=np.stack([np.asarray([a[role+'_xy']for a in rows],np.float64)for role in ROLES],1)
    coarse=np.stack([native_to_coarse(native[:,i])for i in range(2)],1)
    cameras={role:_camera(geometry[role])for role in ROLES}
    ref=cameras['reference'][1]
    if not np.array_equal(ref,np.asarray(m['reference_cam2world']))or not np.array_equal(cameras['query'][1],np.asarray(m['query_training_truth_cam2world'])):raise ValueError('True training camera matrices differ from point metadata')
    for role,i in zip(ROLES,range(2)):
        k,c,_,_=cameras[role];q,_,front=project_z(world,k,c)
        if not front.all()or np.max(np.linalg.norm(q-native[:,i],axis=1))>.02:raise ValueError('Exact native3D point projection differs')
    normal_field=reference_normal_field(geometry['reference'])
    support=np.stack([coarse_support(cameras[role][3],coarse[:,i],normal_field['valid']if i==0 else None)for i,role in enumerate(ROLES)],1)
    views=np.stack([training_view_basis(world,ref,cameras[role][1])for role in ROLES],1)
    suns=(suns_world@ref[:3,:3]).astype(np.float32)
    condition=np.empty((n,12,10),np.float32)
    condition[:,:,:3]=suns[list(EVEN)][None]
    condition[:,:,3:6]=(normal@ref[:3,:3])[:,None]
    condition[:,:,6:9]=views[:,0,None,1:4]
    condition[:,:,9]=reference_log_scale
    target=np.full((n,2,24,32),np.nan,np.float32);available=np.zeros((n,2,24),bool)
    images=np.empty((12,128,128),np.uint8);new_omissions=[];teacher_calls=0
    for i,row in enumerate(rows):
        for role_index,role in enumerate(ROLES):
            if not support[i,role_index]:new_omissions.append(dict(label_index=int(row['label_index']),role=role,reason='coarse_receptive_field_or_geometry_support'))
    for slot,light in enumerate(EVEN):
        for role_index,role in enumerate(ROLES):
            image=np.asarray(even_reader.get(role,int(light)))
            if image.shape!=(512,512)or image.dtype!=np.uint8:raise ValueError('Native512 uint8 even image required')
            low=cv2.resize(image,(128,128),interpolation=cv2.INTER_AREA)
            if role_index==0:images[slot]=low
            ids=np.flatnonzero(support[:,role_index])
            if not len(ids):continue
            features=np.asarray(teacher_function(low,coarse[ids,role_index]),np.float32);teacher_calls+=1
            if features.shape!=(len(ids),32)or not np.isfinite(features).all():raise ValueError('Finite same-padding coarse teacher32D outputs required')
            norm=np.linalg.norm(features,axis=1);good=norm>1e-8
            if not np.allclose(norm[good],1.,atol=2e-4,rtol=0):raise ValueError('Nonzero teacher features must be unit normalized')
            target[ids[good],role_index,light]=features[good];available[ids[good],role_index,light]=True
            for row_index in ids[~good]:new_omissions.append(dict(label_index=int(rows[row_index]['label_index']),role=role,light=light,reason='zero_norm_teacher_not_visibility_label'))
    if available[:,:,1::2].any()or np.isfinite(target[:,:,1::2]).any():raise ValueError('Heldout target access')
    legacy=m['teacher_extraction']
    details=dict(format=FORMAT,unit_index=m['unit_index'],component=m['component'],split='training',retained_point_rows=n,
                 teacher_function_full128_calls=teacher_calls,even_lights=list(EVEN),even_image_get_calls=24,
                 reference_normal_method=NORMAL_METHOD,reference_normal_is_estimate=True,
                 legacy_before_trial_support=m['geometry_counts']['selected_physical_points'],
                 legacy_trial_support_omissions=legacy['omitted_trial_support_samples'],
                 legacy_trial_support_failure_records=deepcopy(legacy['canonical_trial_support_failures']),
                 legacy_selected_indices=deepcopy(legacy['selected_indices']),new_coverage_or_teacher_omissions=new_omissions,
                 all_legacy_rows_and_camera_retained=True,query_truth_is_training_label_only=True,
                 geometric_visibility_from_brightness=False,feature_context_covisibility='unknown_not_certified_by_centre_label',
                 reference_log_scale_convention='log_sampling_factor_relative_to_coarse128_baseline',
                 target_teacher='original_frozen_full128_native_view_INTER_AREA_sample_descriptors')
    return dict(reference_images=images,reference_normal_xy=normal_field['normal_xy'],reference_geometry_valid=normal_field['valid'],
                native_xy=native,coarse_xy=coarse,world=world,point_label_ids=np.asarray([a['label_index']for a in rows],np.int64),
                observed_conditions=condition,target_suns=suns,target_view_basis=views,teacher=target,available=available,
                correspondence_visible=np.ones((n,2),bool),coarse_support=support,reference_input_available=support[:,0].copy(),details=details)


def fit_balanced_training_view_prior(units,*,allowed_training_groups,eligible_unit_indices,source_sha256):
    """Training QUERY view prior, equal group→camera→retained-point weights.

    Self-reference targets help lighting training but must not multiply the
    unknown-query prior. Neither twelve light repeats nor target availability
    reweight known geometry. This function consumes no RGB or descriptors.
    """
    from solar_distributional_response_v86 import TrainingViewPrior,PRIOR_FORMAT
    allowed=set(allowed_training_groups);groups={g:[]for g in allowed};seen=set()
    for u in units:
        d=u['details'];key=d['unit_index']
        if d.get('split')!='training'or d.get('component')not in allowed or key in seen:raise ValueError('Unique original training units only')
        seen.add(key);v=np.asarray(u['target_view_basis'],np.float64)[:,1]
        if v.ndim!=2 or v.shape[1]!=7 or len(v)<1 or not np.isfinite(v).all():raise ValueError('True training query view7 labels required')
        if not np.allclose(v[:,0],1.)or not np.allclose(np.linalg.norm(v[:,1:4],axis=1),1.,atol=1e-6,rtol=0)or not np.allclose(np.linalg.norm(v[:,4:7],axis=1),1.,atol=1e-6,rtol=0):raise ValueError('Unit true query LOS/up required')
        groups[d['component']].append(v)
    if seen!=set(eligible_unit_indices):raise ValueError('Every independently eligible original camera required')
    if any(not x for x in groups.values()):raise ValueError('Every original training group required')
    values=[];weights=[]
    for g in sorted(groups):
        for v in groups[g]:values.append(v);weights.append(np.full(len(v),1/(len(groups)*len(groups[g])*len(v))))
    values=np.concatenate(values);weights=np.concatenate(weights);mean=np.einsum('n,nv->v',weights,values)
    centred=values-mean;cov=np.einsum('n,nv,nw->vw',weights,centred,centred)
    provenance=dict(split='training',sample_count=len(values),allowed_training_groups=sorted(allowed),fitted_groups=sorted(groups),
                    source_sha256=dict(source_sha256),weighting='equal_group_then_camera_then_retained_point_query_role_only',
                    training_view_array_sha256=hashlib.sha256(values.tobytes()+weights.tobytes()).hexdigest(),camera_units=len(seen),lighting_duplicates_counted=False)
    payload=dict(format=PRIOR_FORMAT,mean=mean.tolist(),covariance=cov.tolist(),provenance=provenance,
                 moment_sha256=hashlib.sha256(mean.tobytes()+cov.tobytes()).hexdigest())
    return TrainingViewPrior.from_frozen_payload(payload)
