"""DRAFT pilot-only varied-camera renderer; publish and pass QA before execution."""
import argparse,hashlib,json,re,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timedelta,timezone
import bpy
import numpy as np
from mathutils import Matrix,Vector
sys.path.insert(0,str(Path.cwd()))
from blender_camera_control_v31 import read_exr
from north_camera_variation_v59 import protocol as camera_protocol
from north_camera_scene_v59 import setup_scene
from north_camera_contract_v59 import verify_schedule
from north_render_plan_v31 import plan
from north_development_plan_v35 import proposal_grid_native
from north_camera_correspondences_v59 import mesh_correspondences
from north_solar_geometry_v37 import solar_configuration
from renderer_depth_v34 import control_convention,canonical_depth
from north_coarse_reachability_v53 import verify_ended

SOURCES=['north_camera_variation_v59.py','north_camera_scene_v59.py','render_north_camera_v59.py',
 'north_render_plan_v31.py','north_development_plan_v35.py','north_bvh_supervision_v35.py',
 'north_camera_correspondences_v59.py','north_camera_visibility_v59.py','test_north_camera_visibility_v59.py',
 'north_solar_geometry_v37.py','renderer_depth_v34.py','blender_camera_control_v31.py',
 'north_coarse_reachability_v53.py','verify_north_development_v35.py','north_camera_contract_v59.py','test_north_camera_contract_v59.py','verify_north_camera_v59.py','test_north_camera_variation_v59.py','test_north_camera_audit_v59.py']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def write_new(path,value):
 p=Path(path)
 with p.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def run(schedule_path,data,output,commit):
 # No renderer work while the current fitting/comparison pipeline remains live.
 verify_ended('runs/north-fine-v58-state.json')
 for variant in ('aligned','fractional'):
  a=read('reports/north_fine_'+variant+'_study_audit_v58.json')
  if not a['passed']or a['replayed_rows']!=576:raise ValueError('Completed fine study must be audited before this pilot')
 if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact published source required')
 for n in SOURCES:
  if subprocess.check_output(['git','show',commit+':'+n])!=Path(n).read_bytes():raise ValueError('Published render source differs:'+n)
 qa=read('reports/north_camera_software_v59.json')
 if not qa['passed']or not qa['full_repository_regression']or not qa['fresh_smoke']:raise ValueError('Full current-source software checks required')
 for n,h in qa['source_sha256'].items():
  if sha(n)!=h:raise ValueError('Camera software changed after QA')
 if bpy.app.version[:3]!=(4,5,9)or bpy.app.build_hash.decode()!='8bf95cbd38d1':raise ValueError('Pinned official renderer required')
 root=Path(output).resolve();source=Path(data);schedule=read(schedule_path);acquired=read(source/'summary.json')
 if not acquired['complete']or acquired['calibration_or_test_pixels_decoded']or sha(source/'summary.json')!=schedule['acquired_summary_sha256']:raise ValueError('Unchanged pilot-only acquired metadata required')
 base=plan();sites=[s for s in base['sites']if s['split']=='renderer_cost_pilot'];records={r['component']:r for r in acquired['records']}
 expected=camera_protocol(sites,{k:r['core_max_local_height_m']for k,r in records.items()})
 schedule_replay=verify_schedule(schedule,expected)
 if len(schedule['pairs'])!=48:raise ValueError('Predeclared four-support48pair schedule required')
 if sha('north_render_plan_v31.py')!=schedule['camera_math_source_sha256']:raise ValueError('Camera math changed')
 solar_path='reports/north_solar_amendment_v37.json';solar_contract=read(solar_path)
 if solar_contract['solar_source_sha256']!=sha('north_solar_geometry_v37.py'):raise ValueError('Solar contract changed')
 if root.exists():raise FileExistsError('Preserve all completed/failed attempts')
 root.mkdir(parents=True);start=datetime.now(timezone.utc);tick=time.monotonic();cap=1800
 deadline=start+timedelta(seconds=cap)
 def clock():
  now=datetime.now(timezone.utc)
  if now<start or now>=deadline or time.monotonic()-tick>=cap:raise TimeoutError('Original30minute camera pilot allowance exhausted')
 protocol=dict(created_utc=start.isoformat(),absolute_deadline_utc=deadline.isoformat(),max_calendar_seconds=cap,
  cpu_threads=2,publication_commit=commit,source_sha256={n:sha(n)for n in SOURCES},
  software_check_sha256=sha('reports/north_camera_software_v59.json'),upstream_fine_audit_sha256={v:sha('reports/north_fine_'+v+'_study_audit_v58.json')for v in ('aligned','fractional')},
  schedule_replay=schedule_replay,schedule_sha256=sha(schedule_path),data_summary_sha256=sha(source/'summary.json'),solar_contract_sha256=sha(solar_path),
  geometric_minimum_bidirectional_visible_fraction=.20,eligible_family_minimum=12,max_geometry_audit_seconds=1200,
  geometric_camera_clearance_minimum_m=10.,occlusion_training_gate=schedule['occlusion_training_gate'],
  material='Unchanged v40 constant diffuse; no albedo texture, new photometric method or dated ephemeris claim.',
  scope='Varied-camera renderer-cost pilot only; no neural inference, training, final/calibration supports or verified orbital accuracy.')
 write_new(root/'protocol.json',protocol);results=[];images=[];convention=None
 try:
  clock();convention=control_convention()
  for site in sites:
   clock();component=site['component'];folder=root/component;folder.mkdir();mesh_path=source/component/'mesh.npz'
   if sha(mesh_path)!=schedule['mesh_sha256'][component]or sha(mesh_path)!=records[component]['mesh_sha256']:raise ValueError('Pilot mesh changed')
   with np.load(mesh_path,allow_pickle=False)as z:
    vertices=z['vertices'].copy();faces=z['faces'].copy();origin=z['body_origin'].copy();enu=z['enu_rows'].copy()
   k=np.asarray(base['intrinsics']);scene,camera,sun,nodes,bvh=setup_scene(folder,vertices,faces,k)
   for pair in [p for p in schedule['pairs']if p['component']==component]:
    clock();pair_id=pair['family']+'-'+str(pair['pair_index']);pair_dir=folder/pair_id;pair_dir.mkdir();views={};valid_pair=True
    for role in ('reference','query'):
     k=np.asarray(pair[role]['intrinsics']);camera.data.lens=36*k[0,0]/512
     pose=np.asarray(pair[role]['cam2world']);m=pose.copy();m[:3,:3]=pose[:3,:3]@np.diag([1.,-1.,-1.]);camera.matrix_world=Matrix(m.tolist())
     camera_dir=pair_dir/role;camera_dir.mkdir()
     hit,normal,index,travel=bvh.ray_cast(Vector(pose[:3,3].tolist()),Vector((0.,0.,-1.)),50000.)
     clearance=None if hit is None else float(travel);free_space=clearance is not None and clearance>10.
     valid_pair &= free_space
     for light in range(6):
      clock();light_dir=camera_dir/f'light-{light}';light_dir.mkdir();solar=solar_configuration(site,light)
      if solar!=solar_contract['prospective_all44_solar_configurations'][component][light]:raise ValueError('Pilot solar direction changed')
      direction=enu@np.asarray(solar['solar_body_direction']);np.testing.assert_allclose(direction,solar['solar_enu_direction'],atol=1e-10,rtol=0)
      sun.rotation_euler=(-Vector(direction.tolist())).to_track_quat('-Z','Y').to_euler()
      for node in nodes:node.base_path=str(light_dir)
      scene.render.filepath=str(light_dir/'image.png');before=time.monotonic();bpy.context.view_layer.update();bpy.ops.render.render(write_still=True)
      images.append(dict(component=component,pair_id=pair_id,role=role,light=light,seconds=time.monotonic()-before,image=str((light_dir/'image.png').relative_to(root)),sha256=sha(light_dir/'image.png')))
      if light==0:
       world=read_exr(next(light_dir.glob('position_*.exr')),512);depth=read_exr(next(light_dir.glob('depth_*.exr')),512)[...,0]
       valid=np.isfinite(world).all(2)&np.isfinite(depth)&(depth>0)&(depth<50000)
       geometry=dict(position=world,renderer_depth=depth,valid=valid,cam2world=pose,intrinsics=k,body_origin=origin,enu_rows=enu)
       if valid.all():
        axial,ray,error=canonical_depth(world,pose,depth,convention);geometry.update(axial_depth=axial,ray_depth=ray)
       else:valid_pair=False
       dest=camera_dir/'geometry.npz';np.savez_compressed(dest,**geometry);views[role]=dict(path=str(dest.relative_to(root)),sha256=sha(dest),complete_surface=bool(valid.all()),camera_free_space=bool(free_space),camera_downward_clearance_m=clearance)
    item=dict(component=component,pair_id=pair_id,family=pair['family'],pair_index=pair['pair_index'],views=views,eligible=False,reason='incomplete finite surface support or unsafe camera clearance',labels=[])
    if valid_pair:
     fractions=[]
     for role,other in (('reference','query'),('query','reference')):
      clock();labels=mesh_correspondences(bvh,np.asarray(pair[role]['cam2world']),np.asarray(pair[other]['cam2world']),np.asarray(pair[role]['intrinsics']),np.asarray(pair[other]['intrinsics']),proposal_grid_native())
      dest=pair_dir/(role+'-to-'+other+'.npz');np.savez_compressed(dest,**labels)
      fraction=float(labels['visible'].mean());fractions.append(fraction)
      item['labels'].append(dict(source=role,target=other,path=str(dest.relative_to(root)),sha256=sha(dest),points=len(labels['visible']),visible=int(labels['visible'].sum()),geometrically_occluded=int(labels['geometrically_occluded'].sum()),visibility_uncertain=int(labels['visibility_uncertain'].sum()),source_valid=int(labels['source_valid'].sum())))
     item.update(eligible=min(fractions)>=.20,reason=None if min(fractions)>=.20 else 'insufficient bidirectional geometric overlap')
    write_new(pair_dir/'summary.json',item);results.append(item);print('CAMERA PILOT PAIR',component,pair_id,'eligible',item['eligible'],flush=True)
  clock();eligible={family:sum(r['eligible']and r['family']==family for r in results)for family in schedule['families']}
  result=dict(complete=True,pairs=results,images=images,rendered_images=len(images),eligible_by_family=eligible,
   adequate_geometric_case_count=all(v>=12 for v in eligible.values()),geometry_pose_audit_pending=True,
   raw_depth_convention=convention,calendar_seconds=(datetime.now(timezone.utc)-start).total_seconds(),
   protocol_sha256=sha(root/'protocol.json'),matcher_forwards=0,optimizer_updates=0,calibration_or_final_images_rendered=0)
  write_new(root/'summary.json',result);write_new(root/'COMPLETE.json',dict(sha256={str(p.relative_to(root)):sha(p)for p in sorted(root.rglob('*'))if p.is_file()}));return result
 except BaseException as exc:
  write_new(root/'FAILED.json',dict(error_type=type(exc).__name__,error=str(exc),completed_pairs=len(results),elapsed_seconds=time.monotonic()-tick));raise


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--schedule',required=True);p.add_argument('--data',default='data/north-render-v31/pilots-v33-local');p.add_argument('--output',required=True);p.add_argument('--publication-commit',required=True)
 a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);run(a.schedule,a.data,a.output,a.publication_commit)
