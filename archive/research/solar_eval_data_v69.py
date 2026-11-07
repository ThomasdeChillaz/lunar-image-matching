"""Source-gated native development manifest and scoring-asset construction.

No image matching, neural imports, training, rendering or reserve access occurs.
Full original-scene bounds certify negatives; query truth is stored separately.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import math
import subprocess
import time
import numpy as np
from PIL import Image
from solar_physics_v63 import lighting_schedule
from solar_sun_search_v68 import sun_search_catalogue

LIGHT_PAIRS = ((0, 0), (16, 16), (0, 17), (16, 1), (8, 17), (8, 1))
FRAME = 'lunar_body_fixed_cartesian_xy_metres'
SOURCES = ('solar_eval_data_v69.py', 'build_solar_eval_data_v69.py', 'index_solar_reference_assets_v69.py', 'solar_physics_v63.py', 'solar_sun_search_v68.py')
CORE_REFERENCE_KEYS = {'depth','intrinsics','cam2world','normals','sun_vectors','solar_states','solar_variants','reference_sun','world_variants'}
OPTIONAL_REFERENCE_KEYS = {'sun_parameters','coarse_sun_mask','solar_materialized_mask'}
HEIGHT_VARIANTS = (0.,-5.,5.,-10.,10.,-20.,20.)
LOOKUP_SOURCES = {'solar_lookup_v70.py','solar_lookup_worker_v70.py','solar_map_v63.py','solar_physics_v63.py','solar_active_v63.py','north_terrain_pilot_v32.py'}
PRECOMPUTE_SOURCES = {'prepare_solar_reference_v63.py','solar_map_v63.py','solar_physics_v63.py','solar_active_v63.py','north_terrain_pilot_v32.py','solar_sun_search_v68.py'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def exact_integer(value, what):
    if type(value) is not int: raise ValueError('Exact integer required: '+what)
    return value


def safe_child(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or str(p) in ('', '.'):
        raise ValueError('Unit payload paths must stay inside their sealed unit')
    return Path(root)/p


def verify_hash(path, expected, what):
    if not isinstance(expected,str) or len(expected)!=64 or sha(path)!=expected:
        raise ValueError('Frozen input changed: '+what)


def verify_publication(protocol_path, commit, protocol):
    if len(commit or '')!=40 or any(c not in '0123456789abcdef'for c in commit):raise ValueError('Exact source publication commit required')
    if protocol.get('prepared_only',False):raise ValueError('Prepared template cannot build scientific assets')
    if set(protocol['source_sha256']) != set(SOURCES):
        raise ValueError('Exact producer and physical-lighting source closure required')
    if subprocess.check_output(['git','show',commit+':'+str(protocol_path)]) != Path(protocol_path).read_bytes():
        raise ValueError('Manifest protocol must be source-published before building assets')
    for name,h in protocol['source_sha256'].items():
        verify_hash(name,h,name)
        if subprocess.check_output(['git','show',commit+':'+name]) != Path(name).read_bytes():
            raise ValueError('Executed producer dependency is not source-published: '+name)


def validate_gate(protocol, data, audit, master, assets):
    """Metadata-only admission. An oracle failure NEVER drops a camera case."""
    if (protocol['schema_version']!=69 or protocol['phase']not in('development_case_assets','reference_asset_index') or
        protocol['split']!='model_selection' or protocol['calibration_or_final_access'] or
        protocol['cpu_threads']!=2 or protocol['max_calendar_seconds']!=1800 or
        tuple(map(tuple,protocol['light_pairs']))!=LIGHT_PAIRS):
        raise ValueError('Fixed, selection-only 1800-second two-thread asset protocol required')
    if (not data['complete'] or data['rendered_images']!=4608 or len(data['units'])!=96 or
        data.get('calibration_or_final_access') or data.get('matcher_forwards') or data.get('optimizer_updates')):
        raise ValueError('Complete data-only 96-unit dataset required')
    if (not audit['passed'] or not audit['complete_dataset'] or audit['expected_units']!=96 or
        audit['completed_units']!=96 or len(audit['units'])!=96 or
        not isinstance(audit['oracle_success'],(int,float))or isinstance(audit['oracle_success'],bool)or
        not .95<=audit['oracle_success']<=1 or
        audit['calibration_or_final_access']):
        raise ValueError('Full >=95% independently passed geometry audit required')
    rows = audit['units']; ids = [exact_integer(u['unit_index'],'audit unit index') for u in rows]
    data_ids = [exact_integer(u['unit_index'],'dataset unit index') for u in data['units']]
    if set(ids)!=set(range(96)) or len(set(ids))!=96 or set(data_ids)!=set(range(96)) or len(set(data_ids))!=96:
        raise ValueError('Every original camera unit must occur exactly once')
    if len(master['units'])!=96: raise ValueError('Original full physical schedule must contain 96 units')
    if any(u['source_sha256']['solar_physics_v63.py']!=protocol['source_sha256']['solar_physics_v63.py']for u in master['units']):
        raise ValueError('Lighting metadata must use the actual unchanged renderer physics source')
    group_splits = {}
    for row in rows:
        i=row['unit_index']; spec=master['units'][i]; site=spec['site']
        if row['split'] not in ('training','model_selection') or row['component']!=site['component'] or row['split']!=site['split']:
            raise ValueError('Audit/original split and component identity differ')
        old=group_splits.setdefault(row['component'],row['split'])
        if old!=row['split']: raise ValueError('Terrain appears in multiple splits')
    if dict(Counter(group_splits.values()))!={'training':16,'model_selection':8}:
        raise ValueError('Original 16 training and eight selection terrain groups required')
    if any(sum(r['component']==g for r in rows)!=4 for g in group_splits):
        raise ValueError('Four original camera units per terrain required')
    if audit['terrain_counts']!={'training':16,'model_selection':8}:
        raise ValueError('Geometry group count receipt differs')
    eligible = [r for r in rows if r['eligible']]
    directions=exact_integer(audit['oracle_pose_directions'],'oracle directions')
    correct=exact_integer(audit['correct_oracle_poses'],'correct oracle poses')
    if directions!=2*len(eligible) or not 0<=correct<=directions or directions==0 or abs(correct/directions-audit['oracle_success'])>1e-12:
        raise ValueError('All eligible directions and oracle failures must stay in the denominator')
    training_groups={g for g,split in group_splits.items()if split=='training'}
    if {r['component']for r in eligible if r['split']=='training'}!=training_groups:
        raise ValueError('At least one eligible camera per original training terrain required')
    # Selection terrains with no independently eligible camera remain explicit
    # inventory exclusions. Oracle/pose failures NEVER alter eligibility.
    if (assets.get('prepared_only',False)or assets['schema_version']!=69 or assets['split']!='model_selection' or
        assets['calibration_or_final_access'] or assets['geometry_audit_sha256']!=protocol['geometry_audit_sha256']):
        raise ValueError('Reference-only asset index must bind the full geometry gate')
    selected={r['unit_index']:r for r in eligible if r['split']=='model_selection'}
    asset_rows=assets['units']; asset_ids=[exact_integer(a['unit_index'],'reference asset index')for a in asset_rows]
    if set(asset_ids)!=set(selected) or len(set(asset_ids))!=len(asset_ids):
        raise ValueError('Every eligible selection camera needs exactly one reference-only asset')
    if (data['original_master_sha256']!=protocol['original_master_sha256'] or
        audit['original_master_sha256']!=protocol['original_master_sha256'] or
        audit['completion_summary_sha256']!=protocol['dataset_summary_sha256'] or
        audit['audit_source_sha256']!=protocol['geometry_audit_source_sha256']):
        raise ValueError('Full audit/dataset/original source provenance differs')
    return selected


def selection_metadata(audit):
    """Preserve the generated32-camera inventory without dropping pose failures."""
    rows=sorted((r for r in audit['units']if r['split']=='model_selection'),key=lambda r:r['unit_index'])
    groups=sorted({r['component']for r in rows});eligible_groups=sorted({r['component']for r in rows if r['eligible']})
    return dict(terrain_groups=groups,generated_selection_units=rows,generated_selection_unit_count=len(rows),
        eligible_selection_unit_count=sum(r['eligible']for r in rows),excluded_selection_units=[r for r in rows if not r['eligible']],
        eligible_terrain_groups=eligible_groups,zero_eligible_terrain_groups=sorted(set(groups)-set(eligible_groups)))


def validate_sealed_unit(record, audit_row, spec, clock):
    root=Path(record['output']); summary_path=root/'summary.json'; seal_path=root/'COMPLETE.json'
    for name,path in [('summary',summary_path),('seal',seal_path)]:
        expected=record[name+'_sha256']
        verify_hash(path,expected,str(path))
        if audit_row['source_'+name+'_sha256']!=expected: raise ValueError('Audit source receipt changed')
    if Path(audit_row['source_output'])!=root: raise ValueError('Audit names another unit output')
    seal=read(seal_path); expected=set(seal['sha256'])
    actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=seal_path}
    if actual!=expected: raise ValueError('Sealed unit file inventory differs')
    for relative,h in seal['sha256'].items():
        clock();verify_hash(safe_child(root,relative),h,relative)
    summary=read(summary_path)
    if (not summary['complete'] or summary['rendered_images']!=48 or summary['matcher_forwards'] or
        summary['optimizer_updates'] or summary['training_allowed'] or summary['calibration_or_final_access']):
        raise ValueError('Only complete unchanged physical data units are admitted')
    images={}
    for row in summary['images']:
        key=(row['role'],exact_integer(row['light'],'lighting index'))
        if key in images: raise ValueError('Duplicated role/light')
        path=safe_child(root,row['path']);verify_hash(path,row['sha256'],str(path));images[key]=(str(path),row['sha256'])
    if set(images)!={(role,l)for role in ('reference','query')for l in range(24)}:
        raise ValueError('Both roles and every physical light required')
    if set(summary['views'])!={'reference','query'} or {r['source']for r in summary['labels']}!={'reference','query'} or len(summary['labels'])!=2:
        raise ValueError('Bidirectional geometry and labels required')
    if spec['site']['component']!=audit_row['component']: raise ValueError('Unit terrain changed')
    return dict(output=root,summary=summary,images=images)


def load_npz(path,keys=None):
    with np.load(path,allow_pickle=False)as z:return {k:z[k].copy()for k in(z.files if keys is None else keys)}


def body_scene_bounds(mesh):
    """The complete mesh encloses all full-image first-hit surface footprints.

    Unlike a bounding box of pixel-centre hits, every point on every triangle is
    enclosed. A shared body-fixed XY frame also avoids comparing local ENUs.
    """
    vertices=np.asarray(mesh['vertices'],float); faces=np.asarray(mesh['faces'])
    origin=np.asarray(mesh['body_origin'],float); rows=np.asarray(mesh['enu_rows'],float)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<3 or not np.isfinite(vertices).all()or
        faces.ndim!=2 or faces.shape[1]!=3 or len(faces)<1 or not np.issubdtype(faces.dtype,np.integer) or
        faces.min()<0 or faces.max()>=len(vertices) or origin.shape!=(3,) or rows.shape!=(3,3) or
        not np.isfinite(origin).all()or not np.allclose(rows@rows.T,np.eye(3),atol=1e-12,rtol=0)or
        not np.isclose(np.linalg.det(rows),1.,atol=1e-12,rtol=0)):
        raise ValueError('Complete finite original mesh and body/ENU frame required')
    body=vertices[:,0,None]*rows[0]+vertices[:,1,None]*rows[1]+vertices[:,2,None]*rows[2]+origin
    lo=body[:,:2].min(axis=0);hi=body[:,:2].max(axis=0)
    # Outward round-off margin makes the bounding certificate conservative.
    margin=1e-6+32*np.finfo(float).eps*np.max(np.abs(body))
    bounds=np.r_[lo-margin,hi+margin]
    if not(bounds[0]<bounds[2]and bounds[1]<bounds[3]):raise ValueError('Nonempty full-scene footprint required')
    return bounds.tolist(),float(vertices[:,2].max()),origin,rows


def bounds_disjoint(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=(4,)or b.shape!=(4,)or not np.isfinite([a,b]).all()or not(a[0]<a[2]and a[1]<a[3]and b[0]<b[2]and b[1]<b[3]):
        raise ValueError('Finite full-scene footprint bounds required')
    return bool(a[2]<b[0]or b[2]<a[0]or a[3]<b[1]or b[3]<a[1])


def convert_sun_frame(query_sun,query_enu_rows,reference_enu_rows):
    s=np.asarray(query_sun,float);q=np.asarray(query_enu_rows,float);r=np.asarray(reference_enu_rows,float)
    if s.shape!=(3,)or q.shape!=(3,3)or r.shape!=(3,3)or not np.isfinite([s]).all()or not np.isclose(np.linalg.norm(s),1.,atol=1e-10):
        raise ValueError('Unit query Sun and two finite ENU transforms required')
    for rows in (q,r):
        if not np.isfinite(rows).all()or not np.allclose(rows@rows.T,np.eye(3),rtol=0,atol=1e-12)or not np.isclose(np.linalg.det(rows),1.,rtol=0,atol=1e-12):
            raise ValueError('Right-handed body/ENU frame required')
    out=s@q@r.T
    if not np.isclose(np.linalg.norm(out),1.,atol=1e-10):raise ValueError('Sun frame conversion changed length')
    return out.tolist()


def plan_cases(units):
    """Every eligible camera x six lights, plus one matched disjoint negative.

    All input units are already eligible selection units. Pairing is fixed by
    camera family/index, cyclic terrain ordering and unit index, never imagery.
    If a negative cannot be certified, fail rather than silently omit a pair.
    """
    byid={u['unit_index']:u for u in units}
    if len(byid)!=len(units)or not units:raise ValueError('Unique eligible selection units required')
    groups=sorted({u['component']for u in units});rank={g:i for i,g in enumerate(groups)}
    if len(groups)!=8:raise ValueError('Eight selection terrain groups required')
    rows=[]
    for i in sorted(byid):
        u=byid[i]
        if u['split']!='model_selection' or not u['eligible']:raise ValueError('Selection eligibility required')
        candidates=[v for v in units if v['component']!=u['component']and bounds_disjoint(u['body_bounds'],v['body_bounds'])]
        candidates.sort(key=lambda v:(v['family']!=u['family'],v['pair_index']!=u['pair_index'],
                        (rank[v['component']]-rank[u['component']])%len(groups),v['unit_index']))
        if not candidates:raise ValueError('No certified foreign-terrain nonoverlap endpoint for unit '+str(i))
        v=candidates[0]
        for a,b in LIGHT_PAIRS:
            positive=f'u{i:03d}-l{a:02d}-{b:02d}-positive'
            rows.append(dict(case_id=positive,unit_index=i,query_unit_index=i,light_a=a,light_b=b,positive=True))
            rows.append(dict(case_id=f'u{i:03d}-to-u{v["unit_index"]:03d}-l{a:02d}-{b:02d}-nonoverlap',
                             unit_index=i,query_unit_index=v['unit_index'],light_a=a,light_b=b,positive=False,matched_positive_case=positive))
    return rows


def validate_reference_archive(path):
    z=load_npz(path);keys=set(z)
    if not CORE_REFERENCE_KEYS<=keys or keys-CORE_REFERENCE_KEYS-OPTIONAL_REFERENCE_KEYS:
        raise ValueError('Strict reference-only archive; query truth and extra keys forbidden')
    s=len(z['sun_vectors'])
    shapes={'depth':(512,512),'intrinsics':(3,3),'cam2world':(4,4),'normals':(256,3),
            'sun_vectors':(s,3),'solar_states':(256,s),'solar_variants':(256,s,7),
            'reference_sun':(3,),'world_variants':(256,7,3)}
    if s<1 or any(z[k].shape!=v for k,v in shapes.items())or any(not np.isfinite(z[k]).all()for k in CORE_REFERENCE_KEYS):
        raise ValueError('Finite native512 reference geometry and 256 anchors required')
    if not(z['depth']>0).all()or not np.isin(z['solar_states'],[0,1,2]).all()or not np.isin(z['solar_variants'],[0,1,2]).all():
        raise ValueError('Explicit depth and clear/shadow/unknown states required')
    if not np.allclose(np.linalg.norm(z['normals'],axis=1),1.,atol=1e-6)or not np.allclose(np.linalg.norm(z['sun_vectors'],axis=1),1.,atol=1e-6):
        raise ValueError('Unit physical directions required')
    if 'sun_parameters'in z and (z['sun_parameters'].shape!=(s,2)or not np.isfinite(z['sun_parameters']).all()):raise ValueError('Finite per-Sun parameters required')
    for k in ('coarse_sun_mask','solar_materialized_mask'):
        if k in z and (z[k].shape!=(s,)or z[k].dtype!=bool):raise ValueError('Explicit boolean Sun availability mask required')
    if 'coarse_sun_mask'in z:
        coarse=z['coarse_sun_mask']
        if not coarse.any():raise ValueError('Nonempty coarse Sun initialization required')
        materialized=z.get('solar_materialized_mask',np.ones(s,bool))
        if not materialized[coarse].all():raise ValueError('Every coarse Sun must be physically computed')
    return z




def validate_generic_catalogue(geometry,latitude_deg):
    """No exact query-Sun columns, even masked, belong in primary archives."""
    expected=sun_search_catalogue(latitude_deg)
    suns=np.array([x['sun_enu']for x in expected])
    parameters=np.array([[x['hour_angle_deg'],x['subsolar_latitude_deg']]for x in expected])
    coarse=np.array([x['coarse']for x in expected],bool)
    if (not np.array_equal(geometry['sun_vectors'],suns)or
        not np.array_equal(geometry.get('sun_parameters'),parameters)or
        not np.array_equal(geometry.get('coarse_sun_mask'),coarse)or
        'solar_materialized_mask'not in geometry):
        raise ValueError('Primary archive must contain exactly the a-priori generic Sun catalogue and availability mask')
    return True


def validate_physical_assets(asset,approved,protocol,commit,approved_key):
    """Validate published lazy lookup and measured offline precompute lineage.

    This reads/hashes metadata only. It neither constructs a provider nor runs
    any worker/ray. Primary archives contain generic Sun catalogues only.
    """
    pinned_inputs={}
    for key in ('solar_lookup_protocol','reference_precompute_protocol','reference_precompute_receipt'):
        verify_hash(asset[key],asset[key+'_sha256'],key);pinned_inputs[asset[key]]=asset[key+'_sha256']
    lookup=read(asset['solar_lookup_protocol']);pre=read(asset['reference_precompute_protocol']);receipt=read(asset['reference_precompute_receipt'])
    lookup_required={'schema_version','cpu_threads','query_truth_used','max_lookup_calendar_seconds','max_request_seconds',
        'mesh','mesh_sha256','scene_mesh','scene_mesh_sha256','reference_geometry','reference_geometry_sha256',
        'geometry_audit','geometry_audit_sha256','approved_reference_key','source_sha256'}
    if set(lookup)!=lookup_required or lookup['schema_version']!=70 or lookup['cpu_threads']!=2 or lookup['query_truth_used']is not False:
        raise ValueError('Strict, reference-only lazy physical lookup protocol required')
    for key,limit in(('max_lookup_calendar_seconds',900),('max_request_seconds',300)):
        if type(lookup[key])is not int or not 0<lookup[key]<=limit:raise ValueError('Bounded lazy physical lookup required')
    if set(lookup['source_sha256'])!=LOOKUP_SOURCES or set(pre['source_sha256'])!=PRECOMPUTE_SOURCES:
        raise ValueError('Exact offline/online physical source closures required')
    for p in (lookup,pre):
        if (p['query_truth_used']is not False or p['cpu_threads']!=2 or
            p['reference_geometry_sha256']!=approved['reference_geometry_sha256']or
            p['scene_mesh_sha256']!=approved['scene_mesh_sha256']or
            p['geometry_audit_sha256']!=protocol['geometry_audit_sha256']or p['approved_reference_key']!=approved_key):
            raise ValueError('Physical protocols must bind only the approved reference/full scene and full audit')
        for name in('mesh','scene_mesh','reference_geometry','geometry_audit'):
            verify_hash(p[name],p[name+'_sha256'],'physical '+name);pinned_inputs[p[name]]=p[name+'_sha256']
        for name,h in p['source_sha256'].items():
            verify_hash(name,h,'physical source '+name)
            if subprocess.check_output(['git','show',commit+':'+name])!=Path(name).read_bytes():raise ValueError('Physical source not published:'+name)
    for key in('solar_lookup_protocol','reference_precompute_protocol'):
        if subprocess.check_output(['git','show',commit+':'+asset[key]])!=Path(asset[key]).read_bytes():raise ValueError('Physical protocol not published:'+key)
    if (pre['output']!=asset['reference_geometry']or receipt['output_sha256']!=asset['reference_geometry_sha256']or
        receipt['protocol_sha256']!=asset['reference_precompute_protocol_sha256']or
        receipt['original_scene_mesh_sha256']!=asset['scene_mesh_sha256']or
        receipt['original_scene_ceiling_m']!=asset['scene_maximum_height_m']or
        receipt['map_mesh_sha256']!=pre['mesh_sha256']or not receipt['complete']or
        receipt['query_truth_used']is not False or receipt['neural_forwards']!=0 or
        receipt['source_sha256']!=pre['source_sha256']):
        raise ValueError('Actual reference-only precompute receipt does not bind this archive/protocol/ceiling')
    elapsed=receipt['calendar_seconds'];cap=pre['max_calendar_seconds']
    if (not isinstance(elapsed,(int,float))or isinstance(elapsed,bool)or not math.isfinite(elapsed)or
        type(cap)is not int or not 0<elapsed<=cap<=900):raise ValueError('Measured bounded precompute time required')
    if lookup['mesh_sha256']!=pre['mesh_sha256']or lookup['approved_reference_key']!=pre['approved_reference_key']:
        raise ValueError('Online lazy rays and offline coarse rays need identical reference-map identity')
    return float(elapsed),pinned_inputs


def materialize_case(plan,units,generated):
    """Pure truth-boundary mapping; no file read, ray or image inference."""
    u=units[plan['unit_index']];v=units[plan['query_unit_index']];a,b=plan['light_a'],plan['light_b']
    ref,rh=u['sealed']['images'][('reference',a)];query,qh=v['sealed']['images'][('query',b)]
    asset=u['asset'];inputs={ref:rh,query:qh,asset['reference_geometry']:asset['reference_geometry_sha256']}
    inputs.update(asset.get('additional_input_sha256',{}));inputs.update(u['physical_inputs'])
    case=dict(plan,split='model_selection',component=u['component'],query_component=v['component'],
        reference=ref,query=query,reference_geometry=asset['reference_geometry'],
        query_intrinsics=v['raw']['query']['intrinsics'].tolist(),reference_sun_world=u['suns'][a],solar_lookup_protocol=asset['solar_lookup_protocol'],
        secondary_known_query_sun_world=convert_sun_frame(v['suns'][b],v['enu_rows'],u['enu_rows']),
        input_sha256=inputs,
        inference_inputs=dict(reference=ref,query=query,reference_geometry=asset['reference_geometry'],
            query_intrinsics=v['raw']['query']['intrinsics'].tolist(),reference_sun_world=u['suns'][a],solar_lookup_protocol=asset['solar_lookup_protocol']),
        secondary_inference_metadata=dict(query_sun_frame='reference_scene_enu',exact_vector_required=True,never_nearest_grid=True),
        reference_asset_metadata=asset,
        frame_metadata=dict(footprint_frame=FRAME,reference_world='reference_scene_enu',query_sun_converted_through_body_fixed=True),
        scoring_inputs={})
    if case['positive']:
        case.update({k:generated[u['unit_index']][k]for k in ('scoring_labels','scoring_geometry')})
        case['input_sha256'].update(generated[u['unit_index']]['scoring_input_sha256'])
        case['scoring_inputs']=dict(labels=case['scoring_labels'],geometry=case['scoring_geometry'])
    else:
        case['nonoverlap_certificate']=dict(frame=FRAME,full_image_footprints=True,
            reference_bounds=u['body_bounds'],query_bounds=v['body_bounds'],
            support='Conservative bounds of every triangle in both ORIGINAL complete finite scene meshes; not a box of sampled pixels.',
            reference_scene_mesh_sha256=u['asset']['scene_mesh_sha256'],query_scene_mesh_sha256=v['asset']['scene_mesh_sha256'])
    return case



def build_reference_index(protocol_path,commit):
    """Inventory measured reference tables after the full gate, without rays.

    The caller supplies one published precompute/lookup protocol pair per
    eligible selection unit. This replaces hand-written archive/hash metadata.
    """
    p=read(protocol_path);verify_publication(protocol_path,commit,p)
    if p['phase']!='reference_asset_index':raise ValueError('Explicit reference-index phase required')
    tick=time.monotonic()
    for key in('dataset_summary','geometry_audit','original_master'):
        verify_hash(p[key],p[key+'_sha256'],key)
    verify_hash(p['geometry_audit_source'],p['geometry_audit_source_sha256'],'full auditor source')
    if subprocess.check_output(['git','show',commit+':'+p['geometry_audit_source']])!=Path(p['geometry_audit_source']).read_bytes():raise ValueError('Full auditor source must be published')
    data=read(p['dataset_summary']);audit=read(p['geometry_audit']);master=read(p['original_master'])
    provisional=dict(schema_version=69,split='model_selection',calibration_or_final_access=False,geometry_audit_sha256=p['geometry_audit_sha256'],
        units=[dict(unit_index=r['unit_index'])for r in audit['units']if r['split']=='model_selection'and r['eligible']])
    selected=validate_gate(p,data,audit,master,provisional)
    configurations=p['reference_unit_protocols']
    if set(configurations)!={str(i)for i in selected}:raise ValueError('Exactly every eligible selection unit needs two declared physical protocols')
    if Path(p['output']).exists():raise FileExistsError('Preserve prior reference asset index')
    units=[]
    for i,u in sorted(selected.items()):
        if time.monotonic()-tick>=1800:raise TimeoutError('Original reference-index build allowance exhausted')
        conf=configurations[str(i)]
        for key in('reference_precompute_protocol','solar_lookup_protocol'):
            verify_hash(conf[key],conf[key+'_sha256'],key)
        pre=read(conf['reference_precompute_protocol']);receipt_path=str(pre['output'])+'.json';receipt=read(receipt_path)
        pairid=u['component']+'/'+u['family']+'-'+str(u['pair_index']);approved=audit['approved_reference_inputs'][pairid]
        archive=pre['output'];verify_hash(archive,receipt['output_sha256'],'completed reference archive')
        row=dict(unit_index=i,component=u['component'],reference_geometry=archive,reference_geometry_sha256=receipt['output_sha256'],
            source_reference_geometry_path=approved['reference_geometry_path'],source_reference_geometry_sha256=approved['reference_geometry_sha256'],
            scene_mesh_path=approved['scene_mesh_path'],scene_mesh_sha256=approved['scene_mesh_sha256'],
            scene_maximum_height_m=approved['scene_maximum_height_m'],source_audit_sha256=p['geometry_audit_sha256'],
            reference_sun_by_light={str(x['index']):x['sun_enu']for x in lighting_schedule(master['units'][i]['site']['centre_latitude_deg'])if x['index']in(0,8,16)},
            height_variants_metres=list(HEIGHT_VARIANTS),unknown_outside_reference_coverage=True,
            solar_lookup_protocol=conf['solar_lookup_protocol'],solar_lookup_protocol_sha256=conf['solar_lookup_protocol_sha256'],
            reference_precompute_protocol=conf['reference_precompute_protocol'],reference_precompute_protocol_sha256=conf['reference_precompute_protocol_sha256'],
            reference_precompute_receipt=receipt_path,reference_precompute_receipt_sha256=sha(receipt_path))
        elapsed,_=validate_physical_assets(row,approved,p,commit,pairid)
        archive_data=validate_reference_archive(archive);validate_generic_catalogue(archive_data,master['units'][i]['site']['centre_latitude_deg'])
        row['measured_reference_precompute_seconds']=elapsed
        units.append(row)
    for key in('dataset_summary','geometry_audit','original_master'):verify_hash(p[key],p[key+'_sha256'],key)
    for n,h in p['source_sha256'].items():verify_hash(n,h,n)
    result=dict(schema_version=69,complete=True,split='model_selection',calibration_or_final_access=False,query_truth_used=False,
        geometry_audit_sha256=p['geometry_audit_sha256'],dataset_summary_sha256=p['dataset_summary_sha256'],original_master_sha256=p['original_master_sha256'],
        units=units,source_sha256=p['source_sha256'],protocol_sha256=sha(protocol_path),publication_commit=commit,
        calendar_seconds=time.monotonic()-tick,photometric_images_decoded=0,ray_queries=0,matcher_forwards=0,optimizer_updates=0,
        scope='Source-gated inventory of already measured reference-only physical archives. No ray work or scientific matcher result.')
    result.update(selection_metadata(audit));result['eligible_unit_indices']=sorted(selected)
    write_new(p['output'],result)
    return result


def build(protocol_path,commit):
    protocol=read(protocol_path);verify_publication(protocol_path,commit,protocol)
    if protocol['phase']!='development_case_assets':raise ValueError('Case/asset producer requires its explicit phase')
    tick=time.monotonic()
    def clock():
        if time.monotonic()-tick>=1800:raise TimeoutError('Original 1800-second asset build allowance exhausted')
    for k in ('dataset_summary','geometry_audit','original_master','reference_assets_index'):
        verify_hash(protocol[k],protocol[k+'_sha256'],k)
    verify_hash(protocol['geometry_audit_source'],protocol['geometry_audit_source_sha256'],'full geometry auditor source')
    if subprocess.check_output(['git','show',commit+':'+protocol['geometry_audit_source']])!=Path(protocol['geometry_audit_source']).read_bytes():raise ValueError('Full geometry auditor source is not published')
    data=read(protocol['dataset_summary']);audit=read(protocol['geometry_audit']);master=read(protocol['original_master']);assets=read(protocol['reference_assets_index'])
    selected=validate_gate(protocol,data,audit,master,assets)
    output=Path(protocol['output'])
    if output.exists():raise FileExistsError('Preserve prior data assets; no overwrite or implicit resume')
    # Gate and the public contract precede ALL asset materialization.
    records={r['unit_index']:r for r in data['units']};audited={r['unit_index']:r for r in audit['units']};sealed={}
    for i in range(96):clock();sealed[i]=validate_sealed_unit(records[i],audited[i],master['units'][i],clock)
    byasset={a['unit_index']:a for a in assets['units']};units={};meshes={}
    for i,row in sorted(selected.items()):
        clock();spec=master['units'][i];pair=spec['camera_pairs'][0];source=sealed[i];asset=byasset[i]
        if len(spec['camera_pairs'])!=1:raise ValueError('Exactly one original camera pair per unit required')
        if (pair['family'],pair['pair_index'])!=(row['family'],row['pair_index']):raise ValueError('Audited camera family/index differs from original schedule')
        pairid=row['component']+'/'+row['family']+'-'+str(row['pair_index'])
        approved=audit['approved_reference_inputs'][pairid]
        expected_fields={'component':row['component'],'source_reference_geometry_path':approved['reference_geometry_path'],
                         'source_reference_geometry_sha256':approved['reference_geometry_sha256'],
                         'scene_mesh_path':approved['scene_mesh_path'],'scene_mesh_sha256':approved['scene_mesh_sha256']}
        if any(asset[k]!=v for k,v in expected_fields.items()):raise ValueError('Reference asset/source full-mesh provenance differs')
        if asset['source_audit_sha256']!=protocol['geometry_audit_sha256']:raise ValueError('Reference asset audit identity differs')
        for k in ('reference_geometry','source_reference_geometry_path','scene_mesh_path'):
            h=asset[{'reference_geometry':'reference_geometry_sha256','source_reference_geometry_path':'source_reference_geometry_sha256','scene_mesh_path':'scene_mesh_sha256'}[k]]
            verify_hash(asset[k],h,k)
        for p,h in asset.get('additional_input_sha256',{}).items():verify_hash(p,h,'reference physical lookup input')
        meshpath=asset['scene_mesh_path']
        if meshpath not in meshes:meshes[meshpath]=body_scene_bounds(load_npz(meshpath))
        bounds,ceiling,origin,enu=meshes[meshpath]
        if (ceiling!=approved['scene_maximum_height_m'] or ceiling!=asset['scene_maximum_height_m'] or
            tuple(asset['height_variants_metres'])!=HEIGHT_VARIANTS or asset['unknown_outside_reference_coverage']is not True):
            raise ValueError('Original full-scene ceiling, height perturbations and unknown crop policy must remain explicit')
        precompute_seconds,physical_inputs=validate_physical_assets(asset,approved,protocol,commit,pairid)
        geometry=validate_reference_archive(asset['reference_geometry'])
        validate_generic_catalogue(geometry,spec['site']['centre_latitude_deg'])
        raw={}
        for role in ('reference','query'):
            view=source['summary']['views'][role];p=safe_child(source['output'],view['path']);verify_hash(p,view['sha256'],role+' geometry')
            g=load_npz(p,('axial_depth','valid','intrinsics','cam2world','body_origin','enu_rows'))
            if (g['axial_depth'].shape!=(512,512)or not g['valid'].all()or not np.array_equal(g['body_origin'],origin)or
                not np.array_equal(g['enu_rows'],enu)or not np.array_equal(g['intrinsics'],np.asarray(pair[role]['intrinsics']))or
                not np.array_equal(g['cam2world'],np.asarray(pair[role]['cam2world']))):
                raise ValueError('Native complete source geometry/body frames/cameras differ')
            raw[role]=g
        if not np.array_equal(geometry['depth'],raw['reference']['axial_depth'])or not np.array_equal(geometry['intrinsics'],raw['reference']['intrinsics'])or not np.array_equal(geometry['cam2world'],raw['reference']['cam2world']):
            raise ValueError('Reference inference archive changes original geometry')
        suns={x['index']:x['sun_enu']for x in lighting_schedule(spec['site']['centre_latitude_deg'])}
        for light in (0,8,16):
            if not np.array_equal(np.asarray(asset['reference_sun_by_light'][str(light)]),np.asarray(suns[light])):
                raise ValueError('Actual reference illumination metadata differs from physical rendering schedule')
        units[i]=dict(row,body_bounds=bounds,enu_rows=enu,body_origin=origin,ceiling=ceiling,
                      raw=raw,suns=suns,asset=asset,sealed=source,precompute_seconds=precompute_seconds,physical_inputs=physical_inputs)
    planned=plan_cases(list(units.values()))
    output.mkdir(parents=True);generated={}
    try:
        write_new(output/'execution.json',dict(protocol_sha256=sha(protocol_path),publication_commit=commit,
                   source_sha256=protocol['source_sha256'],started_utc=datetime.now(timezone.utc).isoformat(),
                   max_calendar_seconds=1800,cpu_threads=2,matcher_forwards=0,optimizer_updates=0))
        for i,u in sorted(units.items()):
            clock();folder=output/f'u{i:03d}';folder.mkdir()
            label=[r for r in u['sealed']['summary']['labels']if r['source']=='reference'][0]
            p=safe_child(u['sealed']['output'],label['path']);verify_hash(p,label['sha256'],'exact reference labels');z=load_npz(p)
            if z['source_points'].shape!=(676,2)or z['target_points'].shape!=(676,2)or z['visible'].shape!=(676,):raise ValueError('All exact native source anchors required')
            labels_path=folder/'scoring_labels.npz';np.savez_compressed(labels_path,reference_xy=z['source_points'],query_xy=z['target_points'],visible=z['visible'],query_cam2world=u['raw']['query']['cam2world'])
            geometry_path=folder/'scoring_geometry.npz';np.savez_compressed(geometry_path,
                reference_depth=u['raw']['reference']['axial_depth'],query_depth=u['raw']['query']['axial_depth'],
                reference_intrinsics=u['raw']['reference']['intrinsics'],reference_cam2world=u['raw']['reference']['cam2world'],
                query_intrinsics=u['raw']['query']['intrinsics'],query_cam2world=u['raw']['query']['cam2world'])
            generated[i]=dict(scoring_labels=str(labels_path),scoring_geometry=str(geometry_path),
                              scoring_input_sha256={str(labels_path):sha(labels_path),str(geometry_path):sha(geometry_path)})
        cases=[]
        for plan in planned:
            clock();case=materialize_case(plan,units,generated)
            for path in (case['reference'],case['query']):
                with Image.open(path)as im:
                    if im.size!=(512,512):raise ValueError('Native image extent must stay 512x512')
            cases.append(case)
        clock()
        for k in ('dataset_summary','geometry_audit','original_master','reference_assets_index'):verify_hash(protocol[k],protocol[k+'_sha256'],k)
        for n,h in protocol['source_sha256'].items():verify_hash(n,h,n)
        result=dict(schema_version=70,split='model_selection',calibration_or_final_access=False,
            complete=True,terrain_groups=sorted({u['component']for u in units.values()}),
            footprint_frame=FRAME,light_pairs=[list(p)for p in LIGHT_PAIRS],eligible_unit_indices=sorted(units),
            positive_cases=sum(c['positive']for c in cases),negative_cases=sum(not c['positive']for c in cases),cases=cases,
            reference_precompute_receipts={u['asset']['reference_geometry']:{'receipt_path':u['asset']['reference_precompute_receipt'],'sha256':u['asset']['reference_precompute_receipt_sha256'],'calendar_seconds':u['precompute_seconds']}for u in units.values()},
            protocol_sha256=sha(protocol_path),publication_commit=commit,geometry_audit=protocol['geometry_audit'],geometry_audit_sha256=protocol['geometry_audit_sha256'],
            dataset_summary_sha256=protocol['dataset_summary_sha256'],original_master_sha256=protocol['original_master_sha256'],reference_assets_index_sha256=protocol['reference_assets_index_sha256'],
            source_sha256=protocol['source_sha256'],calendar_seconds=time.monotonic()-tick,
            photometric_images_decoded=0,matcher_forwards=0,optimizer_updates=0,
            query_truth_policy='Query depth/pose and exact projected labels live only in scoring NPZs. True query Sun is explicit SECONDARY metadata only, transformed into reference ENU; estimated-Sun inference must not read it.',
            dependence='Eight original selection terrains, four camera pairs each where eligible, six light conditions; repeated conditions are not independent geography.',
            negative_pairing='Fixed family/index preference, cyclic terrain ordering and unit index; every eligible positive gets one certified different-terrain nonoverlap; no image-based cherry picking.')
        result.update(selection_metadata(audit))
        write_new(output/'manifest.json',result)
        write_new(output/'COMPLETE.json',dict(sha256={str(p.relative_to(output)):sha(p)for p in sorted(output.rglob('*'))if p.is_file()}))
        return result
    except BaseException as exc:
        write_new(output/'FAILED.json',dict(complete=False,error_type=type(exc).__name__,error=str(exc),calendar_seconds=time.monotonic()-tick))
        raise
