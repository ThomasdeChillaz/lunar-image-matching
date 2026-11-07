"""Reference-camera physical-relighting library + SIFT + common final PnP.

This is an explicitly limited render-and-match baseline, not viewpoint rendering
or optical-image validation. Missing published libraries/controls fail closed.
No renderer is started here; original measured render costs remain in provenance.
"""
from pathlib import Path
import hashlib,json,math,subprocess
import numpy as np
from historical_compact_adapters_v73 import (BaselineUnavailable,sha,read,value_digest,
    validate_images,validate_points,validate_control_outputs)

METHOD='reference_camera_render_match'
SOURCES={'direct_render_adapter_v73.py','historical_compact_adapters_v73.py','solar_evaluation_v70.py',
 'native_baseline_pose_v7.py','localization_robust_pose_v4.py','localization_pose_refit_v3.py',
 'lunar_geometry_v5.py','lunar_geometry_v2.py'}


def geometry_digest(depth,k,pose):
    h=hashlib.sha256()
    for value in(depth,k,pose):
        a=np.ascontiguousarray(value,dtype='<f8');h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def available_sun_entries(entries,known_sun):
    """Never silently replace an exact supplied Sun with the nearest render."""
    primary=[e for e in entries if e['primary_generic']is True]
    if not primary:raise BaselineUnavailable('No prospectively declared generic reference Sun renders')
    if known_sun is None:return primary
    s=np.asarray(known_sun,float)
    if s.shape!=(3,)or not np.isfinite(s).all()or not np.isclose(np.linalg.norm(s),1.,atol=1e-10):raise ValueError('Exact unit supplied Sun vector required')
    result=[e for e in primary if np.allclose(np.asarray(e['sun_world']),s,rtol=0,atol=1e-12)]
    if not result:raise BaselineUnavailable('Exact supplied-Sun render unavailable; no nearest-grid or query-render substitution')
    return result



def reference_preparation_cost(record,summary):
    """Measured attribution plus a conservative whole-pair wall-time bound.

    The outer interval includes startup, I/O and unused query work. It is an
    upper bound on reference-only preparation, not a fabricated exact split.
    """
    outer=record.get('outer_calendar_seconds');internal=summary['calendar_seconds']
    reference=summary['views']['reference']['centre_ray_seconds']+sum(r['render_seconds']for r in summary['images']if r['role']=='reference')
    if any(not isinstance(x,(int,float))or isinstance(x,bool)or not math.isfinite(x)or x<=0 for x in(outer,internal,reference)):
        raise BaselineUnavailable('Original outer/reference rendering cost is not fully measured; no zero-cost preparation claim')
    if outer+1e-6<internal or internal+1e-6<reference:raise ValueError('Original rendering cost intervals are inconsistent')
    return float(outer),float(reference)


def validate_render_profile(profile):
    if profile.get('prepared_only',False)or not profile.get('available',True):raise BaselineUnavailable(profile.get('unavailable_reason','No completed source-published reference-only render library and native controls'))
    if profile['backend']!='direct_render'or profile['method']!=METHOD or profile['cpu_threads']!=2 or profile['native_size']!=512 or profile['calibration_or_final_access']:
        raise ValueError('Explicit development native512 render-and-match profile required')
    config=profile['configuration']
    if (config['viewpoint_search']!='reference_camera_only'or config['descriptor']!='SIFT'or
        config['ratio_threshold']!=.75 or config['nfeatures']!=2048 or type(config['top_pose_candidates'])is not int or
        not 1<=config['top_pose_candidates']<=8 or config['known_sun_policy']!='exact_render_or_unavailable'):
        raise ValueError('The declared limited direct-render control cannot silently change methods')
    commit=profile['publication_commit']
    if len(commit)!=40 or any(c not in'0123456789abcdef'for c in commit):raise ValueError('Exact source publication required')
    if not SOURCES<=set(profile['source_sha256']):raise ValueError('Direct-render source closure incomplete')
    for n,h in profile['source_sha256'].items():
        if sha(n)!=h or subprocess.check_output(['git','show',commit+':'+n])!=Path(n).read_bytes():raise ValueError('Direct-render source differs from public freeze:'+n)
    for n,h in profile['artifact_sha256'].items():
        if sha(n)!=h:raise ValueError('Render-library/control artifact changed:'+n)
    evidence=profile['render_evidence']
    for k in('protocol','manifest','receipt','controls'):
        if sha(evidence[k])!=evidence[k+'_sha256']:raise ValueError('Render evidence changed:'+k)
    protocol=read(evidence['protocol']);manifest=read(evidence['manifest']);receipt=read(evidence['receipt'])
    if subprocess.check_output(['git','show',commit+':'+evidence['protocol']])!=Path(evidence['protocol']).read_bytes():raise ValueError('Reference-library inventory protocol not published')
    if (not manifest['complete']or manifest['native_size']!=512 or manifest['split']!='model_selection'or
        manifest['query_truth_used']or manifest['query_role_images_used']or manifest['calibration_or_final_access']or
        manifest['viewpoint_search']!='reference_camera_only'or
        not receipt['complete']or receipt['manifest_sha256']!=evidence['manifest_sha256']or
        receipt['protocol_sha256']!=evidence['protocol_sha256']):raise ValueError('Complete audited reference-only render-library receipt required')
    if protocol['source_sha256']!=manifest['producer_source_sha256']:raise ValueError('Library producer provenance differs')
    for key in('dataset_summary','geometry_audit','original_master','reference_assets_index'):
        if sha(protocol[key])!=protocol[key+'_sha256']:raise ValueError('Original completed library input changed:'+key)
    dataset=read(protocol['dataset_summary']);records={r['unit_index']:r for r in dataset['units']}
    audit=read(protocol['geometry_audit'])
    if sha(protocol['geometry_audit'])!=protocol['geometry_audit_sha256']or not audit['passed']or not audit['complete_dataset']or audit['completed_units']!=96 or not .95<=audit['oracle_success']<=1:
        raise ValueError('Full96 >=95% reference geometry qualification required')
    for n,h in protocol['source_sha256'].items():
        if sha(n)!=h or subprocess.check_output(['git','show',commit+':'+n])!=Path(n).read_bytes():raise ValueError('Library producer source differs from published bytes:'+n)
    if (len(audit['units'])!=96 or audit['oracle_pose_directions']!=2*sum(bool(u['eligible'])for u in audit['units'])or
        audit['correct_oracle_poses']/audit['oracle_pose_directions']!=audit['oracle_success']):raise ValueError('Every eligible geometry direction must remain in library gate')
    lookup={};offline=0.
    for library in manifest['libraries']:
        key=library['reference_geometry_sha256']
        if key in lookup:raise ValueError('Ambiguous reference render-library identity')
        approved=audit['approved_reference_inputs'][library['approved_reference_key']]
        if (approved['reference_geometry_sha256']!=key or approved['scene_mesh_sha256']!=library['scene_mesh_sha256']or
            library['map_mesh_sha256']!=library['scene_mesh_sha256']):raise ValueError('Reused full-scene renders must use exactly the primary full reference map; cropped-map reuse is not fair')
        if library['source_role']!='reference'or library['query_camera_fields_used']or len(library['entries'])!=24:
            raise ValueError('Only all24 predeclared REFERENCE-camera renders may form this library')
        if sha(library['source_summary'])!=library['source_summary_sha256']:raise ValueError('Original timed reference-render source changed')
        source=read(library['source_summary']);reference_rows={r['light']:r for r in source['images']if r['role']=='reference'}
        if set(reference_rows)!=set(range(24))or {e['light_index']for e in library['entries']}!=set(range(24)):
            raise ValueError('Every original reference Sun hypothesis required')
        for entry in library['entries']:
            original=reference_rows[entry['light_index']]
            if (entry['source_role']!='reference'or entry['primary_generic']is not True or
                entry['image_sha256']!=original['sha256']or Path(entry['image'])!=Path(library['source_summary']).parent/original['path']):
                raise ValueError('A query-camera/undeclared image cannot enter the primary reference library')
            sun=np.asarray(entry['sun_world'],float)
            if sun.shape!=(3,)or not np.isfinite(sun).all()or not np.isclose(np.linalg.norm(sun),1.,atol=1e-10):raise ValueError('Unit generic reference-world Sun required')
        raw_path=Path(library['source_summary']).parent/source['views']['reference']['path']
        if sha(raw_path)!=key:raise ValueError('Reference source geometry changed')
        with np.load(raw_path,allow_pickle=False)as raw:
            if geometry_digest(raw['axial_depth'],raw['intrinsics'],raw['cam2world'])!=library['reference_geometry_array_sha256']:
                raise ValueError('Library camera/native array identity differs from qualified reference geometry')
        record=records[library['unit_index']]
        if record['summary_sha256']!=library['source_summary_sha256']or Path(record['output'])/'summary.json'!=Path(library['source_summary']):raise ValueError('Library no longer belongs to completed source unit')
        expected_cost,attributable_cost=reference_preparation_cost(record,source)
        cost=library['original_reference_preparation_seconds']
        if not np.isclose(library['attributable_reference_image_geometry_seconds'],attributable_cost,rtol=0,atol=1e-9):raise ValueError('Measured reference timing components differ')
        if not np.isclose(cost,expected_cost,rtol=0,atol=1e-9):raise ValueError('Offline reference rendering/preparation cost was omitted or changed')
        if not isinstance(cost,(int,float))or isinstance(cost,bool)or not math.isfinite(cost)or cost<=0:raise ValueError('Actual original reference rendering/preparation cost required')
        offline+=cost;lookup[key]=library
    controls=read(evidence['controls'])
    if controls.get('actual_render_forward')is not True:raise ValueError('An actual source-bound rendering/native-matching control is required')
    copy=dict(controls,actual_pretrained_forward=True)
    validate_control_outputs(copy,profile)
    return manifest,lookup,dict(method=METHOD,library_manifest_sha256=evidence['manifest_sha256'],
        offline_original_reference_preparation_seconds=offline,inventory_seconds=receipt['calendar_seconds'],
        incremental_render_seconds=0.,native_size=512,synthetic_reference_library=True,
        limitation='One known reference-camera viewpoint; no query-view rendering, no optical navigation validation. Fixed24-light library cannot supply arbitrary off-grid known-Sun cases.',
        timing='Original rendering/geometry preparation is reported separately; complete online cost includes image I/O, SIFT search and every common PnP candidate. No cross-case image-feature cache.',
        common_reference_map_only=True,source_sha256=profile['source_sha256'])


class SiftLibraryMatcher:
    def __init__(self,configuration):
        import cv2
        cv2.setNumThreads(1);self.cv2=cv2;self.detector=cv2.SIFT_create(nfeatures=configuration['nfeatures']);self.ratio=configuration['ratio_threshold']
    def query_features(self,image):return self.detector.detectAndCompute(image,None)
    def match(self,render,query_features):
        cv2=self.cv2;ka,da=self.detector.detectAndCompute(render,None);kb,db=query_features
        if da is None or db is None or len(db)<2:return np.empty((0,2)),np.empty((0,2)),0.
        pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(da,db,k=2)
        matches=sorted([x[0]for x in pairs if len(x)==2 and x[0].distance<self.ratio*x[1].distance],key=lambda x:(x.distance,x.queryIdx,x.trainIdx))
        unique={}
        for match in matches:unique.setdefault(match.trainIdx,match)
        matches=list(unique.values())
        return np.array([ka[x.queryIdx].pt for x in matches]).reshape(-1,2),np.array([kb[x.trainIdx].pt for x in matches]).reshape(-1,2),-float(np.mean([x.distance for x in matches]))if matches else 0.


def load_native_render(entry):
    from PIL import Image
    if sha(entry['image'])!=entry['image_sha256']:raise ValueError('A reference render changed after inventory')
    with Image.open(entry['image'])as source:
        if source.size!=(512,512)or source.mode not in('L','RGB'):raise ValueError('Native512 eight-bit reference content required; no resize or bit-depth substitute')
        image=np.asarray(source.convert('L')).copy()
    return image


class DirectRenderAdapter:
    def __init__(self,profile,matcher_factory=None,image_loader=None,estimator=None):
        self.profile=profile;self.manifest,self.libraries,self.provenance=validate_render_profile(profile)
        self.matcher=(matcher_factory or SiftLibraryMatcher)(profile['configuration']);self.image_loader=image_loader or load_native_render
        if estimator is None:
            from solar_evaluation_v70 import common_pose
            estimator=common_pose
        self.estimator=estimator
    def infer(self,inp):
        validate_images(inp)
        if inp.solar_lookup is None or not hasattr(inp.solar_lookup,'protocol'):raise BaselineUnavailable('Identical reference-map identity/coverage is unavailable')
        physical=inp.solar_lookup.protocol;key=physical['reference_geometry_sha256']
        if key not in self.libraries:raise BaselineUnavailable('No declared library for this qualified reference geometry')
        library=self.libraries[key]
        if physical['mesh_sha256']!=library['map_mesh_sha256']or physical['scene_mesh_sha256']!=library['scene_mesh_sha256']:
            raise BaselineUnavailable('Render library map differs from primary map/height perturbation; no unperturbed or larger-map substitution')
        if geometry_digest(inp.geometry.depth,inp.geometry.intrinsics,inp.geometry.cam2world)!=library['reference_geometry_array_sha256']:
            raise ValueError('Native reference depth/camera differs from library rendering camera')
        entries=available_sun_entries(library['entries'],inp.known_query_sun)
        features=self.matcher.query_features(inp.query);proposals=[]
        for entry in entries:
            render=self.image_loader(entry);p,q,appearance=self.matcher.match(render,features);p,q=validate_points(p,q)
            proposals.append(dict(entry=entry,p=p,q=q,appearance=float(appearance)))
        proposals.sort(key=lambda h:(-len(h['p']),-h['appearance'],h['entry']['light_index']))
        poses=[]
        for candidate in proposals[:self.profile['configuration']['top_pose_candidates']]:
            pose,info=self.estimator(candidate['p'],candidate['q'],inp)
            poses.append(dict(candidate,pose=pose,info=info))
        if not poses:return dict(reference_xy=np.empty((0,2)),query_xy=np.empty((0,2)),pose=None,reason='empty_reference_library')
        poses.sort(key=lambda h:(h['pose']is None,-int(h['info'].get('inliers',0)),
            float(h['info'].get('clipped_squared_reprojection_cost',float('inf'))),-len(h['p']),h['entry']['light_index']))
        best=poses[0];pose,info=self.estimator(best['p'],best['q'],inp)
        return dict(reference_xy=best['p'],query_xy=best['q'],pose=pose,reason=info.get('reason'),common_estimator=info,
            estimated_sun=np.asarray(best['entry']['sun_world']),query_truth_used=False,
            render_search=dict(reference_viewpoint_only=True,library_entries_considered=len(entries),common_pnp_candidates=len(poses),final_common_pnp_calls=1,
                exact_known_sun=inp.known_query_sun is not None,selected_light_index=best['entry']['light_index'],
                original_reference_preparation_seconds=library['original_reference_preparation_seconds'],incremental_render_seconds=0.,
                preparation_cost_scope=library.get('preparation_cost_scope','conservative whole-pair outer wall-time bound'),
                original_render_source='synthetic physical reference-role images, never query-role images',optical_accuracy_verified=False))


def make_direct_render(profile):return DirectRenderAdapter(profile)


def render_profile_inputs(profile):
    result=dict(profile['artifact_sha256']);e=profile['render_evidence']
    for key in('protocol','manifest','receipt','controls'):result[e[key]]=e[key+'_sha256']
    manifest=read(e['manifest'])
    for library in manifest['libraries']:
        result[library['source_summary']]=library['source_summary_sha256']
        for entry in library['entries']:result[entry['image']]=entry['image_sha256']
    return result
