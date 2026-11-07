"""Fresh clean-image comparison of shared-light initialization and patch policies.

Called only by the separately admitted controller. Query truth is scoring-only.
"""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'solar-blind-v138'))
sys.path.insert(0,str(HERE.parent/'solar-local-v129'))
from catalogue_v138 import Catalogue,ProjectionCache
from blind_matcher_v138 import active,initialize
from reference_cache import CachedReferenceSift,FeatureStore

METHODS=('direct','joint_no_extra','uniform','appearance','geometry','active')
pipeline=active.joint.native.control.pipeline


def decode_clean16(path):
    raw=cv2.imdecode(np.frombuffer(Path(path).read_bytes(),dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    if raw is None or raw.shape!=(512,512,3) or raw.dtype!=np.uint16:
        raise ValueError('Native512 RGB16 required; no RGB8 decoder/downcast substitute')
    rgb=raw[...,::-1].astype(float)/65535.
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    return linear@np.array([.2126,.7152,.0722])


def direct(catalogue,query,depth,k,pose,qk,valid):
    tick=perf_counter()
    store=CachedReferenceSift.store
    before=(store.hits,store.misses,len(store.miss_seconds))
    matcher=CachedReferenceSift(dict(nfeatures=2048,ratio_threshold=.75))
    observed=pipeline.encode_image(query)[0]
    features=matcher.query_features(observed); proposals=[]
    for i,image in enumerate(catalogue.images):
        p,q,appearance=matcher.match(pipeline.encode_image(image,valid)[0],features)
        proposals.append(dict(index=i,p=p,q=q,appearance=appearance))
    proposals.sort(key=lambda r:(-len(r['p']),-r['appearance'],catalogue.light_ids[r['index']]))
    candidates=[]
    for row in proposals[:8]:
        candidates.append(dict(active.joint.frame.solve(row['p'],row['q'],depth,k,pose,qk),
                               light_slot=row['index'],matches=len(row['p'])))
    candidates.sort(key=lambda r:(r['pose'] is None,-r['estimator'].get('inliers',0),
        r['estimator'].get('clipped_squared_reprojection_cost',float('inf')),
        -r['matches'],catalogue.light_ids[r['light_slot']]))
    best=candidates[0]
    return dict(best,conditional_stage_seconds=perf_counter()-tick,
        estimated_catalogue_sun_proposal=None if best['matches']<6 else catalogue.sun_world[best['light_slot']].tolist(),
        reference_images_considered=len(proposals),matched_query_reference_added=False,
        reference_feature_cache=dict(hits=store.hits-before[0],misses=store.misses-before[1],
            preparation_seconds=sum(store.miss_seconds[before[2]:]),query_features_cached=False),
        configuration=dict(nfeatures=2048,ratio=.75,top_pose_candidates=8,
                           input='Independent1/99-percentile linear-luminance encoding; reference depth mask'))


def models(catalogue,query,depth,k,pose,qk,valid,cache,rotation,*,extra_budget=32,control_poses=None):
    tick=perf_counter(); hypotheses,initialization=initialize(catalogue,query,depth,k,pose,qk,
        reference_valid=valid,cache=cache,poses=control_poses)
    init_seconds=perf_counter()-tick
    active.joint.native.direction=active.joint.safe_direction
    first=direct(catalogue,query,depth,k,pose,qk,valid) if rotation%2==0 else None
    if hypotheses:
        best=hypotheses[0]; reference=catalogue.images[best['light_slot']]
        up=pose[:3,:3].T@np.array([0.,0.,1.])
        tick=perf_counter()
        separate=active.run(reference,query,depth,k,pose,qk,best['pose'],reference_valid=valid,
            height_direction_reference=up,policies=('uniform',),budget=0)['methods']['joint_no_extra']
        noextra_seconds=perf_counter()-tick
        order=active.POLICIES[rotation:]+active.POLICIES[:rotation]
        result=active.run(reference,query,depth,k,pose,qk,best['pose'],reference_valid=valid,
            height_direction_reference=up,policies=order,budget=extra_budget)
        for field in ('reference_xy','query_xy','pose','estimator'):
            if separate[field]!=result['methods']['joint_no_extra'][field]:
                raise ValueError('Separate/internal no-extra control changed: '+field)
        result['methods']['joint_no_extra']=dict(separate,conditional_stage_seconds=noextra_seconds)
        estimated=best['catalogue_sun_world']
    else:
        null=active.joint.frame.solve([],[],depth,k,pose,qk)
        result=dict(methods={n:dict(null,reason='no_joint_light_camera_initializer',conditional_stage_seconds=0.)
                             for n in METHODS[1:]})
        order=active.POLICIES[rotation:]+active.POLICIES[:rotation];estimated=None
    baseline=first if first is not None else direct(catalogue,query,depth,k,pose,qk,valid)
    methods=dict(direct=baseline,**result['methods'])
    methods={n:dict(methods[n],online_seconds=methods[n]['conditional_stage_seconds']+(0. if n=='direct' else init_seconds)) for n in METHODS}
    for name in METHODS[1:]:methods[name]['estimated_catalogue_sun_proposal']=estimated
    return dict(methods=methods,initialization=initialization,executed_policy_order=list(order),
        direct_order='first' if first is not None else 'last',fresh_initializer=True,query_truth_used=False,
        input_encoding='Native512 cleanRGB16 Standard/sRGB inverted to linear luminance',
        solar_visibility_certified=False,accepted_pose_study=False,parameters_added=0)


def setup():
    cv2.setNumThreads(1);CachedReferenceSift.store=FeatureStore();return ProjectionCache()


def generated(layered):
    rng=np.random.default_rng(20261106)
    images=np.array([.1+rng.random((32,32)).repeat(16,0).repeat(16,1) for _ in range(24)])
    angle=np.linspace(-1.5,1.5,24)
    suns=np.c_[np.sqrt(.96)*np.cos(angle),np.sqrt(.96)*np.sin(angle),np.full(24,.2)]
    catalog=Catalogue(images,suns,tuple(range(24)))
    k=np.array([[440.,0.,255.5],[0.,440.,255.5],[0.,0.,1.]])
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=2000.
    truth=pose.copy();truth[0,3]=20.
    depth=np.empty((512,512));query=np.full((512,512),.1)
    for lo,hi,shift in (((0,180,10),(180,333,8),(333,512,6)) if layered else ((0,512,8),)):
        depth[lo:hi]=440.*20./shift;query[lo:hi,:-shift]=1.2*images[8,lo:hi,shift:]+.03
    return catalog,query,depth,k,pose,k,np.ones_like(depth,bool),truth


def timing(write):
    samples=[]
    for i,layered in enumerate((False,True)):
        args=generated(layered);tick=perf_counter();raw=models(*args[:-1],setup(),i)
        for name,m in raw['methods'].items():
            if m['pose'] is None or np.linalg.norm(np.asarray(m['pose'])[:3,3]-args[-1][:3,3])>=2.:
                raise ValueError('Full-budget generated pose gate failed: '+name)
            if name in active.POLICIES and m['acquisition']['attempted_extra_patches']!=32:
                raise ValueError('Full32-search cost pilot cannot shrink its patch budget')
        elapsed=perf_counter()-tick
        samples.append(dict(layered=layered,elapsed_seconds=elapsed,all_six_pose_controls_pass=True,
            cameras=raw['initialization']['coarse']['pose_count'],lights=24,extra_searches=32))
        print(samples[-1],flush=True)
    forecast=1.5*8*max(r['elapsed_seconds'] for r in samples)+30.
    write('timing.json',dict(procedural_only=True,physical_sunlight_simulation=False,samples=samples,
        full_pipeline=True,lunar_images_opened=False,comparison_forecast_seconds=forecast,
        generated_pattern_controls_not_sunlight=True))


def predict(protocol,write,checked):
    rows=[];cache=setup()
    inputs=json.loads(checked(protocol['inference_manifest']).read_bytes())
    for i,case in enumerate(inputs):
        tick=perf_counter()
        catalog=Catalogue(np.array([decode_clean16(checked(r['image']['path'])) for r in case['catalogue']]),
            np.array([r['sun_world'] for r in case['catalogue']]),tuple(r['light_id'] for r in case['catalogue']))
        query=decode_clean16(checked(case['query_image']['path']))
        with np.load(checked(case['reference_geometry']['path']),allow_pickle=False) as data:
            depth,k,pose,valid=(data[n].copy() for n in ('axial_depth','intrinsics','cam2world','valid'))
        io=perf_counter()-tick
        raw=models(catalog,query,depth,k,pose,np.asarray(case['query_intrinsics']),valid,cache,i%4)
        for method in raw['methods'].values(): method['online_seconds']+=io
        rows.append(dict(id=case['id'],result=raw,input_decode_seconds=io))
        print(dict(completed=i+1,total=len(inputs),id=case['id']),flush=True)
    write('predictions.json',dict(rows=rows,query_truth_used=False,accepted_results=False))


def fields(inp,label,checked):
    with np.load(checked(inp['reference_geometry']['path']),allow_pickle=False) as data:
        da,ka,ca=(data[n].copy() for n in ('axial_depth','intrinsics','cam2world'))
    with np.load(checked(label['query_geometry']['path']),allow_pickle=False) as data:
        db,kb,cb=(data[n].copy() for n in ('axial_depth','intrinsics','cam2world'))
    return da,db,ka,ca,kb,cb


def score(protocol,write,checked,read):
    from lunar_geometry_v5 import correspondences_z,median_reference_range_z
    raw=read('predictions.json');inputs=json.loads(checked(protocol['inference_manifest']).read_bytes())
    labels=json.loads(checked(protocol['scoring_manifest']).read_bytes());rows=[]
    for record,inp,label in zip(raw['rows'],inputs,labels,strict=True):
        if record['id']!=inp['id'] or record['id']!=label['id']:raise ValueError('Case order changed')
        geom=fields(inp,label,checked) if label['positive'] else None; methods={}
        for name,m in record['result']['methods'].items():
            p,q=(np.asarray(m[n]).reshape(-1,2) for n in ('reference_xy','query_xy'))
            returned=m['pose'] is not None;correct=strict=False;error=None;counts={}
            if geom is not None:
                _,target,_,visible=correspondences_z(*geom,points=p);residual=np.linalg.norm(q[visible]-target,axis=1)
                counts={str(t):dict(correct=int((residual<=t).sum()),visible=int(visible.sum()),returned=len(p)) for t in (1,2,4,8,12)}
                if returned:
                    camera=np.asarray(m['pose']);relative=camera[:3,:3].T@geom[-1][:3,:3]
                    angle=float(np.degrees(np.arctan2(np.linalg.norm([relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]]),np.trace(relative)-1)))
                    shift=float(np.linalg.norm(camera[:3,3]-geom[-1][:3,3]));distance=median_reference_range_z(geom[0],geom[2])
                    correct=angle<=2. and shift<=.01*distance;strict=angle<=.5 and shift<=.001*distance
                    error=dict(rotation_deg=angle,translation_m=shift,reference_range_m=distance)
            methods[name]=dict(correct=bool(correct),strict=bool(strict),pose_returned=returned,
                wrong_positive=bool(label['positive'] and returned and not correct),raw_nonoverlap=bool(not label['positive'] and returned),
                correspondence=counts,pose_error=error)
            sun=m['estimated_catalogue_sun_proposal']
            methods[name]['sun_proposal_error_deg']=(None if sun is None or not label['positive'] else
                float(np.degrees(np.arccos(np.clip(np.asarray(sun)@np.asarray(label['query_sun_scoring_only']),-1.,1.)))))
        rows.append(dict(id=record['id'],methods=methods))
    write('scores.json',rows)


def replay(protocol,write,checked,read):
    # Load only the independent saved-arithmetic implementation, not a matcher.
    path=HERE.parent/'solar-active-v137/replay_saved.py'
    spec=spec_from_file_location('independent_geometry_for_v139',path);independent=module_from_spec(spec);spec.loader.exec_module(independent)
    raw,scores=read('predictions.json'),read('scores.json')
    inputs=json.loads(checked(protocol['inference_manifest']).read_bytes());labels=json.loads(checked(protocol['scoring_manifest']).read_bytes())
    if not len(raw['rows'])==len(scores)==len(inputs)==len(labels)==8:
        raise ValueError('Full eight-case study required')
    if raw['query_truth_used'] or len({r['id'] for r in inputs})!=8 or sum(r['positive'] for r in labels)!=4:
        raise ValueError('Original truth-free eight-case/four-positive grid required')
    counts={n:dict(correct=0,strict=0,wrong_positive=0,raw_nonoverlap=0) for n in METHODS};cells=0
    for record,scored,inp,label in zip(raw['rows'],scores,inputs,labels,strict=True):
        if not record['id']==scored['id']==inp['id']==label['id']:raise ValueError('Case IDs changed during replay')
        if tuple(record['result']['methods'])!=METHODS:raise ValueError('Missing comparison method')
        acquisition=[record['result']['methods'][n].get('acquisition') for n in active.POLICIES]
        if any(a is None for a in acquisition):
            if any(a is not None for a in acquisition) or record['result']['initialization']['photometric_hypotheses']:
                raise ValueError('Missing acquisition despite a valid initializer')
        else:
            if len({a['attempted_extra_patches'] for a in acquisition})!=1 or len({tuple(a['cell_schedule']) for a in acquisition})!=1:
                raise ValueError('Unequal acquisition budgets/spatial slots')
            for a in acquisition:
                if (a['requested_extra_patches']!=32 or not 0<=a['attempted_extra_patches']<=32
                    or not len(set(a['selected_indices']))==len(a['selected_indices'])==a['attempted_extra_patches']):
                    raise ValueError('Duplicated, uncharged or changed acquisition budget')
        geom=fields(inp,label,checked) if label['positive'] else None
        for name,m in record['result']['methods'].items():
            for coordinates in ('reference_xy','query_xy'):
                endpoints=np.asarray(m[coordinates]).reshape(-1,2)
                if not np.isfinite(endpoints).all() or not ((endpoints>=0)&(endpoints<=511)).all():
                    raise ValueError('Invalid native pixel endpoints')
            actual=independent.arithmetic(m,label['positive'],geom);expected=scored['methods'][name]
            for key in ('correct','strict','pose_returned','wrong_positive','raw_nonoverlap','correspondence'):
                if actual[key]!=expected[key]:raise ValueError('Independent metric differs: '+key)
            sun=m['estimated_catalogue_sun_proposal']
            if sun is not None and label['positive']:
                a,b=np.asarray(sun),np.asarray(label['query_sun_scoring_only'])
                angle=float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b)),a@b)))
                if abs(angle-expected['sun_proposal_error_deg'])>1e-5:raise ValueError('Independent Sun-proposal arithmetic differs')
            elif expected['sun_proposal_error_deg'] is not None:raise ValueError('No Sun error for absent/negative proposals')
            if actual['pose_error'] is not None:
                for key,value in actual['pose_error'].items():
                    if abs(value-expected['pose_error'][key])>1e-5:raise ValueError('Independent pose arithmetic differs')
            for key in counts[name]:counts[name][key]+=int(actual[key])
            cells+=1
    write('replay.json',dict(passed=True,method_case_cells=cells,methods=counts,
        positive_pairs=4,nonoverlap_pairs=4,accepted_pose_study=False,
        calibration_or_final_access=False,independent_geometry_arithmetic=True))
