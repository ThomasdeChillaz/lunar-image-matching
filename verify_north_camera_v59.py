"""Draft saved-render geometry/depth/oracle audit; no neural model."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
SOURCE_RAY_TOLERANCE_PX=.02  # Existing independently audited v35 float-ray convention.


def project(world,k,pose):
 world=np.asarray(world,float);k=np.asarray(k,float);pose=np.asarray(pose,float)
 camera=(world-pose[:3,3])@pose[:3,:3];h=camera@k.T
 with np.errstate(divide='ignore',invalid='ignore'):q=h[:,:2]/h[:,2,None]
 return q,camera[:,2]


def pose_errors(estimate,truth):
 if estimate is None:return None
 a,b=np.asarray(estimate),np.asarray(truth)
 return dict(rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1)))),translation_m=float(np.linalg.norm(a[:3,3]-b[:3,3])))


def check_labels(labels,source,target,k,target_intrinsics=None):
 p,q,w=[np.asarray(labels[n])for n in ('source_points','target_points','world')]
 valid=np.asarray(labels['source_valid']);inside=np.asarray(labels['in_frame']);visible=np.asarray(labels['visible'])
 if p.shape!=(676,2)or q.shape!=p.shape or w.shape!=(676,3)or any(x.shape!=(676,)or x.dtype!=bool for x in (valid,inside,visible)):
  raise ValueError('Exact676point BVH label schema required')
 if not all(np.isfinite(x).all()for x in (p,q,w)):raise ValueError('Nonfinite geometric labels')
 yy,xx=np.mgrid[:26,:26];expected_grid=np.column_stack([49.5+16*xx.ravel(),49.5+16*yy.ravel()])
 if not np.array_equal(p,expected_grid):raise ValueError('Declared half-pixel proposal grid differs')
 if np.any(visible&~inside)or np.any(inside&~valid):raise ValueError('Visibility must imply valid in-frame surface')
 occluded=np.asarray(labels['geometrically_occluded']);uncertain=np.asarray(labels['visibility_uncertain']);hits=np.asarray(labels['target_hit_distance_m'])
 if any(x.shape!=(676,)or x.dtype!=bool for x in (occluded,uncertain))or hits.shape!=(676,):raise ValueError('Explicit geometric occlusion/hit-distance labels required')
 distance=np.linalg.norm(w-np.asarray(target)[:3,3],axis=1);finite=np.isfinite(hits)
 expected_visible=inside&finite&(np.abs(hits-distance)<=.05);expected_occluded=inside&finite&(hits<distance-.05)
 if not np.array_equal(visible,expected_visible)or not np.array_equal(occluded,expected_occluded)or not np.array_equal(uncertain,inside&~expected_visible&~expected_occluded):raise ValueError('Target first-hit classification differs')
 projected,z=project(w,k,source);other,depth=project(w,k if target_intrinsics is None else target_intrinsics,target)
 if not valid.any()or np.linalg.norm(projected[valid]-p[valid],axis=1).max()>SOURCE_RAY_TOLERANCE_PX:raise ValueError('Exact source ray projection differs')
 front=valid&(depth>0);expected=front&((other>=0)&(other<=511)).all(1)
 if not np.array_equal(expected,inside):raise ValueError('In-frame label differs from independent projection')
 if front.any()and np.max(np.abs(other[front]-q[front]))>1e-8:raise ValueError('Exact target projection differs')
 return float(visible.mean())


def audit(run,schedule_path,output,data='data/north-render-v31/pilots-v33-local'):
 tick=time.monotonic()
 def clock():
  if time.monotonic()-tick>=1200:raise TimeoutError('Original20minute geometry replay allowance exhausted')
 root=Path(run);out=Path(output)
 if out.exists():raise FileExistsError('Preserve earlier audits')
 p=read(root/'protocol.json');s=read(root/'summary.json');schedule=read(schedule_path)
 if not s['complete']or s['rendered_images']!=576 or len(s['pairs'])!=48 or s['matcher_forwards']or s['optimizer_updates']or s['calibration_or_final_images_rendered']or not 0<=s['calendar_seconds']<=1800:
  raise ValueError('Complete bounded pilot-only render required')
 if s['protocol_sha256']!=sha(root/'protocol.json')or p['schedule_sha256']!=sha(schedule_path):raise ValueError('Protocol/schedule changed')
 if s['raw_depth_convention']not in ('axial','ray'):raise ValueError('Unknown renderer depth convention')
 if p['max_geometry_audit_seconds']!=1200 or p['geometric_camera_clearance_minimum_m']!=10.:raise ValueError('Predeclared audit/clearance contract differs')
 sealed=read(root/'COMPLETE.json')['sha256']
 actual={str(x.relative_to(root))for x in root.rglob('*')if x.is_file()and x!=root/'COMPLETE.json'}
 if set(sealed)!=actual:raise ValueError('Complete artifact inventory differs')
 for n,h in sealed.items():
  if sha(root/n)!=h:raise ValueError('Saved camera artifact changed:'+n)
 for n,h in p['source_sha256'].items():
  if sha(n)!=h or subprocess.check_output(['git','show',p['publication_commit']+':'+n])!=Path(n).read_bytes():raise ValueError('Executed/public source differs')
 if p['software_check_sha256']!=sha('reports/north_camera_software_v59.json'):raise ValueError('Camera QA receipt changed')
 for v,h in p['upstream_fine_audit_sha256'].items():
  path='reports/north_fine_'+v+'_study_audit_v58.json'
  if sha(path)!=h or not read(path)['passed']:raise ValueError('Prior fine-study audit changed')
 from native_baseline_pose_v7 import propose_pose
 from lunar_geometry_v5 import median_reference_range_z
 from lunar_geometry_v2 import camera_check
 from north_render_plan_v31 import plan
 from verify_north_development_v35 import local_surface_check
 from north_camera_visibility_v59 import first_mesh_hit,visibility_check
 from PIL import Image
 import cv2
 cv2.setNumThreads(1)
 expected={(x['component'],x['family']+'-'+str(x['pair_index'])):x for x in schedule['pairs']}
 data=Path(data)
 if sha(data/'summary.json')!=p['data_summary_sha256']:raise ValueError('Acquired mesh metadata changed')
 source_plan=plan();sites={x['component']:x for x in source_plan['sites']if x['split']=='renderer_cost_pilot'};offsets=np.asarray(source_plan['mesh_native_offset_indices'])
 meshes={};surface_rows=[];camera_clearance_rows=[];visibility_rows=[];occlusion_rows=[]
 seen=set();outcomes=[];eligible={f:0 for f in schedule['families']}
 image_keys=set()
 for row in s['images']:
  clock()
  key=(row['component'],row['pair_id'],row['role'],row['light'])
  if key in image_keys:raise ValueError('Duplicate rendered image')
  image_keys.add(key);path=root/row['image']
  if sha(path)!=row['sha256']:raise ValueError('Image hash differs')
  with Image.open(path)as im:
   if im.size!=(512,512):raise ValueError('Native image size differs')
 for row in s['pairs']:
  clock()
  key=(row['component'],row['pair_id'])
  if key not in expected or key in seen:raise ValueError('Camera pair inventory differs')
  seen.add(key);spec=expected[key];geometries={};complete=True
  component=row['component']
  if component not in meshes:
   mesh_path=data/component/'mesh.npz'
   if sha(mesh_path)!=schedule['mesh_sha256'][component]:raise ValueError('Acquired mesh changed')
   with np.load(mesh_path,allow_pickle=False)as z:meshes[component]={n:z[n].copy()for n in ('vertices','faces','body_origin','enu_rows')}
  mesh=meshes[component]
  for role in ('reference','query'):
   clock()
   meta=row['views'][role];path=root/meta['path']
   if sha(path)!=meta['sha256']:raise ValueError('Geometry hash differs')
   with np.load(path,allow_pickle=False)as z:geom={k:z[k].copy()for k in z.files}
   k,pose=camera_check(geom['intrinsics'],geom['cam2world'])
   if not np.array_equal(pose,np.asarray(spec[role]['cam2world'])):raise ValueError('Camera schedule/renderer metadata differ')
   if not np.array_equal(k,np.asarray(spec[role]['intrinsics'])):raise ValueError('Separate source/target camera calibration differs')
   clearance=first_mesh_hit(pose[:3,3],[0.,0.,-1.],mesh['vertices'],mesh['faces'])
   free_space=bool(np.isfinite(clearance)and clearance>10.);saved=meta['camera_downward_clearance_m']
   if free_space!=meta['camera_free_space']or (saved is None)!=bool(not np.isfinite(clearance))or(saved is not None and abs(clearance-saved)>.05):raise ValueError('Independent camera terrain clearance differs')
   complete &= free_space
   camera_clearance_rows.append(dict(component=component,pair_id=row['pair_id'],role=role,free_space=free_space,clearance_m=None if not np.isfinite(clearance)else float(clearance)))
   valid=np.asarray(geom['valid']);complete &= bool(valid.all())
   if bool(valid.all())!=meta['complete_surface']:raise ValueError('Complete-surface declaration differs')
   if valid.shape!=(512,512)or valid.dtype!=bool:raise ValueError('Native geometry mask differs')
   if valid.all():
    world=geom['position'];pixel,z=project(world.reshape(-1,3),k,pose);yy,xx=np.mgrid[:512,:512]
    expected_pixels=np.column_stack([xx.ravel(),yy.ravel()])
    if np.linalg.norm(pixel-expected_pixels,axis=1).max()>.02 or not(z>0).all():raise ValueError('Rendered native pixel-center convention differs')
    if not np.array_equal(geom['body_origin'],mesh['body_origin'])or not np.array_equal(geom['enu_rows'],mesh['enu_rows']):raise ValueError('Acquired/rendered terrain frame differs')
    points=world[32:-32:64,32:-32:64].reshape(-1,3).astype(float)
    errors=local_surface_check(points,pose,mesh,sites[component],offsets)
    if not np.isfinite(errors).all()or errors.max()>.05:raise ValueError('Independent rendered-surface intersections differ')
    surface_rows.append(dict(component=component,pair_id=row['pair_id'],role=role,independent_mesh_rays=len(points),maximum_mesh_error_m=float(errors.max())))
    axial=z.reshape(512,512);ray=np.linalg.norm(world-pose[:3,3],axis=-1)
    raw_expected=axial if s['raw_depth_convention']=='axial'else ray
    for name,truth in (('axial_depth',axial),('ray_depth',ray),('renderer_depth',raw_expected)):
     if not np.allclose(geom[name],truth,rtol=1e-5,atol=1e-3):raise ValueError('Native depth convention differs:'+name)
   geometries[role]=geom
  if complete:
   if len(row['labels'])!=2 or {x['source']for x in row['labels']}!={'reference','query'}:raise ValueError('Both exact correspondence directions required')
   fractions=[];checks=[]
   for label in row['labels']:
    if label['target']!=('query'if label['source']=='reference'else'reference')or label['points']!=676:raise ValueError('Exact correspondence direction/grid differs')
    a=geometries[label['source']];b=geometries[label['target']];path=root/label['path']
    if sha(path)!=label['sha256']:raise ValueError('Exact BVH labels changed')
    with np.load(path,allow_pickle=False)as z:labels={k:z[k].copy()for k in z.files}
    clock();fractions.append(check_labels(labels,a['cam2world'],b['cam2world'],a['intrinsics'],b['intrinsics']))
    for name in ('visible','geometrically_occluded','visibility_uncertain','source_valid'):
     if int(labels[name].sum())!=label[name]:raise ValueError('Saved geometric label count differs:'+name)
    occlusion_rows.append(dict(component=component,pair_id=row['pair_id'],source=label['source'],geometrically_occluded=int(labels['geometrically_occluded'].sum()),visibility_uncertain=int(labels['visibility_uncertain'].sum())))
    for kind in ('visible','geometrically_occluded','visibility_uncertain'):
     indices=np.flatnonzero(labels[kind])
     if not len(indices):continue
     clock();index=int(indices[0]);observed=visibility_check(b['cam2world'][:3,3],labels['world'][index],mesh['vertices'],mesh['faces'])
     if any(observed[n]!=bool(labels[n][index])for n in ('visible','geometrically_occluded','visibility_uncertain')):raise ValueError('Independent full-mesh query visibility differs:'+component+'/'+row['pair_id'])
     visibility_rows.append(dict(component=component,pair_id=row['pair_id'],source=label['source'],point_index=index,kind=kind,**observed))
    samples=labels['world'][np.flatnonzero(labels['source_valid'])[::16]]
    errors=local_surface_check(samples,a['cam2world'],mesh,sites[component],offsets)
    if not np.isfinite(errors).all()or not len(errors)or errors.max()>.05:raise ValueError('Independent exact-label surface intersections differ')
    surface_rows.append(dict(component=component,pair_id=row['pair_id'],role=label['source']+'-exact',independent_mesh_rays=len(samples),maximum_mesh_error_m=float(errors.max())))
    visible=labels['visible'];pose,info=propose_pose(labels['source_points'][visible],labels['target_points'][visible],a['axial_depth'],a['intrinsics'],a['cam2world'],b['intrinsics'])
    error=pose_errors(pose,b['cam2world']);range_m=median_reference_range_z(a['axial_depth'],a['intrinsics'])
    checks.append(dict(source=label['source'],oracle_error=error,reference_range_m=range_m,correct=error is not None and error['rotation_deg']<=2 and error['translation_m']<=.01*range_m))
   is_eligible=min(fractions)>=.20
  else:is_eligible=False;checks=[]
  if is_eligible!=row['eligible']:raise ValueError('Geometric eligibility differs')
  if is_eligible:
   eligible[row['family']]+=1
   outcomes.extend(dict(component=row['component'],pair_id=row['pair_id'],family=row['family'],**x)for x in checks)
 if seen!=set(expected)or image_keys!={(c,i,r,l)for c,i in expected for r in ('reference','query')for l in range(6)}:raise ValueError('Complete camera/image inventory required')
 if eligible!=s['eligible_by_family']or min(eligible.values())<12:raise ValueError('At least12eligible geometries per family required')
 correct=sum(x['correct']for x in outcomes);passed=correct/len(outcomes)>=.95
 gate=schedule['occlusion_training_gate'];occluded=sum(x['geometrically_occluded']for x in occlusion_rows)
 occluded_components={x['component']for x in occlusion_rows if x['geometrically_occluded']};verified_occluded=sum(x['geometrically_occluded']for x in visibility_rows)
 occlusion_supported=occluded>=gate['minimum_geometrically_occluded_points']and len(occluded_components)>=gate['minimum_components_with_occlusion']and verified_occluded>=gate['minimum_independently_verified_occluded_query_rays']
 clock()
 result=dict(passed=passed,eligible_by_family=eligible,oracle_pose_directions=len(outcomes),correct_oracle_poses=correct,
  oracle_success=correct/len(outcomes),outcomes=outcomes,source_sha256=sha(__file__),summary_sha256=sha(root/'summary.json'),
  matcher_forwards=0,model_improvement_established=False,
  independent_surface_records=surface_rows,source_ray_tolerance_px=SOURCE_RAY_TOLERANCE_PX,
  independent_camera_clearance=camera_clearance_rows,independent_full_mesh_query_visibility=visibility_rows,
  geometric_occlusion_records=occlusion_rows,geometrically_occluded_samples=occluded,components_with_occlusion=len(occluded_components),
  independently_verified_occluded_query_rays=verified_occluded,occlusion_training_supported=bool(occlusion_supported),calendar_seconds=time.monotonic()-tick,
  visibility_limit='Full-mesh query first hits independently replayed for one deterministic point per available visibility class/direction; not exhaustive ray replication.',
  scope='Saved renderer/depth/pixel-center and exact-correspondence common-estimator controls on four old pilot terrains. Oracle poses are geometry QA, not image matching accuracy or fresh independent terrain.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='outcomes'},indent=2));return result


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--schedule',required=True);p.add_argument('--output',required=True);p.add_argument('--data',default='data/north-render-v31/pilots-v33-local');a=p.parse_args();audit(a.run,a.schedule,a.output,a.data)
