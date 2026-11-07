"""Inventory only timed REFERENCE-role renders; no rendering or model forward.

The immutable full96-unit audit precedes any library preparation. Reference
camera/Sun hypotheses were declared before acquisition. Query-role images and
query depth/pose are never selected as inference library assets.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2';os.environ['VECLIB_MAXIMUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,math,subprocess,time
import numpy as np
from solar_eval_data_v69 import (validate_gate,validate_sealed_unit,lighting_schedule,load_npz)
from historical_compact_adapters_v73 import sha,read
from direct_render_adapter_v73 import geometry_digest,reference_preparation_cost

SOURCES={'build_direct_render_library_v73.py','direct_render_adapter_v73.py','historical_compact_adapters_v73.py',
 'solar_eval_data_v69.py','solar_physics_v63.py','solar_sun_search_v68.py','verify_solar_development_v68.py'}


def write_new(path,value):
    with Path(path).open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def run(protocol_path,commit):
    p=read(protocol_path)
    if p.get('prepared_only',False):raise ValueError('Prepared library protocol cannot run')
    if (p['phase']!='reference_camera_render_library'or p['cpu_threads']!=2 or p['native_size']!=512 or
        p['max_calendar_seconds']!=1800 or p['calibration_or_final_access']or p['query_truth_used']or
        p['reuse_policy']!='reference_role_only_same_full_primary_mesh'or
        p['sun_catalogue_source']!='original_predeclared_24_light_schedule'):
        raise ValueError('Declared reference-only native library protocol required')
    if len(commit)!=40 or any(c not in'0123456789abcdef'for c in commit):raise ValueError('Exact published source revision required')
    if not SOURCES<=set(p['source_sha256']):raise ValueError('Library producer source closure incomplete')
    for n,h in p['source_sha256'].items():
        if sha(n)!=h or subprocess.check_output(['git','show',commit+':'+n])!=Path(n).read_bytes():raise ValueError('Library producer differs from public source freeze:'+n)
    if subprocess.check_output(['git','show',commit+':'+protocol_path])!=Path(protocol_path).read_bytes():raise ValueError('Prospective library protocol not published')
    tick=time.monotonic()
    def clock():
        if time.monotonic()-tick>=1800:raise TimeoutError('Original library inventory allowance exhausted')
    for key in('dataset_summary','geometry_audit','original_master','reference_assets_index'):
        if sha(p[key])!=p[key+'_sha256']:raise ValueError('Pinned complete library input changed:'+key)
    data=read(p['dataset_summary']);audit=read(p['geometry_audit']);master=read(p['original_master']);assets=read(p['reference_assets_index'])
    gate=dict(p,schema_version=69,phase='reference_asset_index',split='model_selection',light_pairs=[[0,0],[16,16],[0,17],[16,1],[8,17],[8,1]],
              geometry_audit_source_sha256=p['source_sha256']['verify_solar_development_v68.py'])
    selected=validate_gate(gate,data,audit,master,assets)
    root=Path(p['output'])
    if root.exists():raise FileExistsError('Preserve every completed or partial render-library inventory')
    records={r['unit_index']:r for r in data['units']};rows={r['unit_index']:r for r in audit['units']};assetmap={r['unit_index']:r for r in assets['units']}
    libraries=[];total_render_seconds=0.
    for i,row in sorted(selected.items()):
        clock();spec=master['units'][i];unit=validate_sealed_unit(records[i],rows[i],spec,clock)
        asset=assetmap[i];lookup=read(asset['solar_lookup_protocol'])
        if sha(asset['solar_lookup_protocol'])!=asset['solar_lookup_protocol_sha256']:raise ValueError('Primary reference-map protocol changed')
        if lookup['query_truth_used']is not False or lookup['mesh_sha256']!=lookup['scene_mesh_sha256']or lookup['scene_mesh_sha256']!=spec['mesh_sha256']:
            raise ValueError('Reused full-scene reference renders cannot receive a larger or different map than primary inference')
        view=unit['summary']['views']['reference'];geometry_path=unit['output']/view['path']
        if view['sha256']!=lookup['reference_geometry_sha256']or sha(geometry_path)!=view['sha256']:raise ValueError('Reference render camera/map geometry identity differs')
        g=load_npz(geometry_path,('axial_depth','intrinsics','cam2world'))
        camera=spec['camera_pairs'][0]['reference']
        if not np.array_equal(g['cam2world'],np.asarray(camera['cam2world']))or not np.array_equal(g['intrinsics'],np.asarray(camera['intrinsics'])):raise ValueError('Only the original known REFERENCE camera may form this library')
        pair=spec['camera_pairs'][0];pairid=row['component']+'/'+pair['family']+'-'+str(pair['pair_index'])
        lights={x['index']:x for x in lighting_schedule(spec['site']['centre_latitude_deg'])}
        reference_rows=[r for r in unit['summary']['images']if r['role']=='reference']
        if len(reference_rows)!=24 or {r['light']for r in reference_rows}!=set(range(24)):raise ValueError('All original generic reference lights required, no image/outcome selection')
        preparation=float(view['centre_ray_seconds']);entries=[]
        for image in sorted(reference_rows,key=lambda x:x['light']):
            seconds=image['render_seconds']
            if not isinstance(seconds,(int,float))or isinstance(seconds,bool)or not math.isfinite(seconds)or seconds<=0:raise ValueError('Measured original render time required')
            preparation+=seconds
            path=unit['output']/image['path']
            entries.append(dict(light_index=image['light'],sun_world=lights[image['light']]['sun_enu'],primary_generic=True,
                image=str(path),image_sha256=image['sha256'],native_shape=[512,512],source_role='reference',
                original_render_seconds=seconds,source_summary_sha256=records[i]['summary_sha256']))
        if not math.isfinite(preparation)or preparation<=0:raise ValueError('Measured complete reference preparation required')
        conservative_preparation,attributable_preparation=reference_preparation_cost(records[i],unit['summary'])
        total_render_seconds+=conservative_preparation
        libraries.append(dict(unit_index=i,component=row['component'],approved_reference_key=pairid,
            reference_geometry_sha256=view['sha256'],reference_geometry_array_sha256=geometry_digest(g['axial_depth'],g['intrinsics'],g['cam2world']),
            map_mesh_sha256=lookup['mesh_sha256'],scene_mesh_sha256=spec['mesh_sha256'],source_role='reference',
            query_camera_fields_used=False,reference_intrinsics=g['intrinsics'].tolist(),reference_cam2world=g['cam2world'].tolist(),
            original_reference_preparation_seconds=conservative_preparation,attributable_reference_image_geometry_seconds=attributable_preparation,
            preparation_cost_scope='Conservative whole original camera-pair outer wall interval, including startup/I/O and UNUSED query work. Ref-only startup allocation is not measured separately.',source_summary= str(unit['output']/'summary.json'),
            source_summary_sha256=records[i]['summary_sha256'],source_seal_sha256=records[i]['seal_sha256'],entries=entries))
    clock()
    for key in('dataset_summary','geometry_audit','original_master','reference_assets_index'):
        if sha(p[key])!=p[key+'_sha256']:raise ValueError('Library input changed during inventory')
    root.mkdir(parents=True)
    result=dict(schema_version=73,complete=True,native_size=512,split='model_selection',calibration_or_final_access=False,
        query_truth_used=False,query_role_images_used=False,viewpoint_search='reference_camera_only',libraries=libraries,
        producer_source_sha256=p['source_sha256'],protocol_sha256=sha(protocol_path),publication_commit=commit,
        original_render_preparation_seconds=total_render_seconds,incremental_render_seconds=0.,
        shared_geometry_preparation_note='Conservative preparation bound includes WHOLE original pair outer interval, including unused query work, startup and I/O; attributable reference render/geometry components are also supplied. Report amortization and avoid double counting shared map cost.',
        library_sun_prior='Original prospectively declared24 physical reference-light states, not query-selected labels. Coverage is narrower than the generic fine Sun catalogue; exact off-grid secondary Suns are unavailable.',
        simulation_scope='Synthetic reference renders share the data renderer/material. Render-and-match agreement on simulation is not independent optical-image or orbital navigation accuracy.',
        runtime_cost_policy='Report original render/preparation separately, inventory time and all online search/SIFT/PnP. Declare amortization count; never present cached rendering as zero preparation.',
        source_image_copies=0,photometric_images_decoded=0,neural_forwards=0,ray_queries=0)
    write_new(root/'manifest.json',result)
    receipt=dict(complete=True,manifest_sha256=sha(root/'manifest.json'),protocol_sha256=sha(protocol_path),publication_commit=commit,
        calendar_seconds=time.monotonic()-tick,original_render_preparation_seconds=total_render_seconds,incremental_render_seconds=0.,
        libraries=len(libraries),entries=sum(len(x['entries'])for x in libraries),scope='Metadata/hash inventory of existing reference-role renders only; native adapter control still required.')
    write_new(root/'COMPLETE.json',receipt)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--publication-commit',required=True)
    a=p.parse_args();r=run(a.protocol,a.publication_commit);print(json.dumps({k:v for k,v in r.items()if k!='libraries'},indent=2))
