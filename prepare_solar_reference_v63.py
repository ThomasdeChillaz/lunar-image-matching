"""Blender BVH reference-only solar-ray tables, including full-map height shifts.

Publish this source and a hashed input protocol before execution. No rendering,
query camera, query depth, image brightness or neural forward is performed.
"""
import argparse,hashlib,json,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from solar_map_v63 import grid_envelope,reference_solar_table
from solar_physics_v63 import solar_direction
from solar_sun_search_v68 import sun_search_catalogue

SOURCES=['prepare_solar_reference_v63.py','solar_map_v63.py','solar_physics_v63.py','solar_active_v63.py','north_terrain_pilot_v32.py','solar_sun_search_v68.py']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(protocol_path,publication_commit):
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    p=json.loads(Path(protocol_path).read_text());tick=time.monotonic();out=Path(p['output'])
    if out.exists()or Path(str(out)+'.json').exists():raise FileExistsError('Preserve existing ray tables')
    if p['query_truth_used']or p['cpu_threads']!=2 or not 0<p['max_calendar_seconds']<=900:raise ValueError('Reference-only bounded solar lookup protocol required')
    for n in SOURCES:
        if p['source_sha256'][n]!=sha(n)or subprocess.check_output(['git','show',publication_commit+':'+n])!=Path(n).read_bytes():raise ValueError('Solar source/publication differs:'+n)
    if subprocess.check_output(['git','show',publication_commit+':'+protocol_path])!=Path(protocol_path).read_bytes():raise ValueError('Input protocol not source-published')
    for key in('reference_geometry','mesh','scene_mesh','geometry_audit'):
        if sha(p[key])!=p[key+'_sha256']:raise ValueError('Pinned input changed:'+key)
    audit=json.loads(Path(p['geometry_audit']).read_text())
    if not audit['passed']or audit['oracle_success']<.95:raise ValueError('95%exact geometry-pose control required')
    approved=audit.get('approved_reference_inputs')
    if approved is not None:
        key=p['approved_reference_key']
        if key not in approved:raise ValueError('Reference geometry is not globally audit-qualified')
        record=approved[key]
        if record['reference_geometry_sha256']!=p['reference_geometry_sha256']or record['scene_mesh_sha256']!=p['scene_mesh_sha256']:raise ValueError('Reference or ORIGINAL scene mesh differs from geometry audit')
    else:
        # Explicit legacy lineage is still supported, but the original scene
        # mesh must be proved through the immutable camera schedule. A cropped
        # map may be a separate mesh; its maximum cannot replace this bound.
        if p['qualified_pair_id']not in audit['qualified_pair_ids']:raise ValueError('Only an explicitly audit-qualified legacy pair may supply reference geometry')
        summary=json.loads(Path(p['render_summary']).read_text())
        if sha(p['render_summary'])!=audit['summary_sha256']:raise ValueError('Audited raw render summary changed')
        row=next(r for r in summary['pairs']if r['component']+'/'+r['pair_id']==p['qualified_pair_id'])
        if row['views']['reference']['sha256']!=p['reference_geometry_sha256']:raise ValueError('Reference input is not the qualified reference view')
        render_protocol=json.loads(Path(p['render_summary']).with_name('protocol.json').read_text())
        if sha(p['camera_schedule'])!=render_protocol['schedule_sha256']:raise ValueError('Original renderer camera schedule changed')
        camera_schedule=json.loads(Path(p['camera_schedule']).read_text())
        if camera_schedule['mesh_sha256'][row['component']]!=p['scene_mesh_sha256']:raise ValueError('Scene ceiling must come from the ORIGINAL full render mesh')
        record={}
    with np.load(p['reference_geometry'],allow_pickle=False)as z:depth,k,pose=[z[n].copy()for n in('axial_depth','intrinsics','cam2world')]
    with np.load(p['mesh'],allow_pickle=False)as z:vertices,faces=z['vertices'].copy(),z['faces'].copy()
    with np.load(p['scene_mesh'],allow_pickle=False)as z:
        scene_vertices=z['vertices']
        if scene_vertices.ndim!=2 or scene_vertices.shape[1]!=3 or not np.isfinite(scene_vertices).all():raise ValueError('Original finite-scene vertices required')
        original_scene_ceiling=float(scene_vertices[:,2].max())
    if 'scene_maximum_height_m'in record and abs(original_scene_ceiling-record['scene_maximum_height_m'])>1e-6:raise ValueError('Original scene ceiling differs from independent geometry audit')
    envelope=grid_envelope(vertices,faces,scene_maximum_height=original_scene_ceiling)
    bvh=BVHTree.FromPolygons(vertices.tolist(),faces.tolist(),all_triangles=True,epsilon=0.)
    def raycast(origin,direction,limit):
        if time.monotonic()-tick>=p['max_calendar_seconds']:raise TimeoutError('Original solar table deadline exhausted')
        point,normal,face,distance=bvh.ray_cast(Vector(origin),Vector(direction),float(limit))
        return None if point is None else(np.asarray(point,float),np.asarray(normal,float))
    axis=49.5+16*np.rint(np.linspace(0,25,16)).astype(int);yy,xx=np.meshgrid(axis,axis,indexing='ij');anchors=np.column_stack([xx.ravel(),yy.ravel()])
    catalogue=sun_search_catalogue(p['latitude_deg']);suns=np.array([x['sun_enu']for x in catalogue])
    table,reasons=reference_solar_table(anchors,k,pose,suns,envelope,raycast)
    reference_sun=solar_direction(p['latitude_deg'],p['reference_hour_angle_deg'],p.get('reference_subsolar_latitude_deg',1.5))
    np.savez_compressed(out,depth=depth,intrinsics=k,cam2world=pose,reference_sun=reference_sun,sun_vectors=suns,
        sun_parameters=np.array([[x['hour_angle_deg'],x['subsolar_latitude_deg']]for x in catalogue]),
        coarse_sun_mask=np.array([x['coarse']for x in catalogue],bool),**table)
    result=dict(complete=True,source_sha256={n:sha(n)for n in SOURCES},protocol_sha256=sha(protocol_path),publication_commit=publication_commit,
        output_sha256=sha(out),calendar_seconds=time.monotonic()-tick,solar_state_counts={str(v):int((table['solar_variants']==v).sum())for v in(0,1,2)},reasons=reasons,
        sun_catalogue=catalogue,terrain_envelope=envelope.__dict__,query_truth_used=False,neural_forwards=0,
        original_scene_mesh_sha256=p['scene_mesh_sha256'],original_scene_ceiling_m=original_scene_ceiling,map_mesh_sha256=p['mesh_sha256'],
        primary_search='Only5degree/.5degree coarse mask initializes hypotheses;1.25degree/.25degree columns are accessed by bounded local refinement. Known query Sun is secondary-track metadata only.',
        coverage_limit='Original full renderer-scene maximum-height assumption, retained under map cropping, and conservative inscribed central square. Missing terrain/holes are rejected; rays escaping the square before the declared ceiling stay unknown.',
        perturbation='Exact ray reintersections after uniform whole-map vertical shifts0,+/-5,+/-10,+/-20m. Not arbitrary local DEM noise.',
        bpy_version=bpy.app.version_string)
    with Path(str(out)+'.json').open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k not in('sun_catalogue','source_sha256')},indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--publication-commit',required=True)
    args=sys.argv[sys.argv.index('--')+1:]if '--'in sys.argv else sys.argv[1:];a=p.parse_args(args);run(a.protocol,a.publication_commit)
