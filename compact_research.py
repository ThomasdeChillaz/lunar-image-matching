"""Separate development, calibration, public freezing and single-use final scoring."""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import argparse
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path
import resource
from time import perf_counter
import urllib.request

import cv2
import numpy as np

from compact_confidence import evidence_vector,fit_confidence,calibrate,accepts,counts,scores
from dataset import sha256
from geometry import transform,endpoint_error,evaluation_points
from lunar_geometry_v2 import correspondences,estimate_pose,pose_error,world_points
from lunargeo_dataset import write_new
from matchers import sift

SOURCES=['compact_research.py','compact_lunar.py','compact_confidence.py','train_compact_v2.py',
         'lunargeo_dataset.py','lunar_geometry_v2.py','modern_baselines_v2.py','matchers.py','geometry.py',
         'compact_sensor_transfer.py','baseline_infer_v2.py','raw_correspondences_v2.py']
SOURCES+=['contrastive_terrain.py','confidence.py','structural_refinement.py','research_quality_gate.py',
          'roma_matcher.py','matchanything_matcher.py','eloftr_matcher.py','loftr_matcher.py',
          'polar_reference.py','dataset.py','synthetic.py','verify_compact_3d.py']
FAMILIES=('translation','rotation','scale','combined')


def validate_manifest(manifest,check_files=False):
    seen={};hash_splits={};ids=set()
    for tile in manifest['tiles']:
        split=tile['split']
        if split not in ('train','model_selection','calibration','test') or tile['tile_id'] in ids:
            raise ValueError('Unexpected split or repeated terrain ID')
        ids.add(tile['tile_id'])
        if seen.setdefault(tile['component'],split)!=split:raise ValueError('Terrain component crosses splits')
        for source in tile['source_hashes']:
            if hash_splits.setdefault(source['camera'],split)!=split:raise ValueError('Source camera leakage')
        if check_files and sha256(tile['cache'])!=tile['cache_sha256']:raise ValueError('Cached input differs')
    if set(seen.values())!={'train','model_selection','calibration','test'}:
        raise ValueError('Need four separate data partitions')
    if len([s for s in seen.values() if s=='test'])<12:raise ValueError('At least12 final spatial components required')
    return manifest


def case_plan(manifest,split):
    if split not in ('model_selection','calibration','test'):raise ValueError('Explicit evaluation split required')
    tiles=[t for t in manifest['tiles'] if t['split']==split];cases=[]
    key=lambda text:hashlib.sha256(('compact-v2:'+text).encode()).hexdigest()
    for t in tiles:
        for i,j in itertools.combinations(range(3),2):
            for family in FAMILIES:
                cases.append(dict(pair_id=f'{t["tile_id"]}:{i}-{j}:{family}',tile_id=t['tile_id'],
                    query_tile_id=t['tile_id'],component=t['component'],split=split,negative=False,
                    light_a=i,light_b=j,family=family,track='controlled'))
        partners=sorted([p for p in tiles if p['component']!=t['component']],key=lambda p:key(t['tile_id']+p['tile_id']))
        chosen=[];components=set()
        for p in partners:
            if p['component'] in components:continue
            chosen.append(p);components.add(p['component'])
            if len(chosen)==3:break
        if len(chosen)<3:raise ValueError('Need3 disjoint negative partner components per evaluation split')
        for p in chosen:
            for i in range(3):
                cases.append(dict(pair_id=f'{t["tile_id"]}:negative:{p["tile_id"]}:{i}',tile_id=t['tile_id'],
                    query_tile_id=p['tile_id'],component=t['component'],query_component=p['component'],
                    split=split,negative=True,light_a=i,light_b=(i+1)%3,family='nonoverlap',track='controlled'))
    return cases


def common_plan(manifest,split):
    """Six metadata-selected cases per spatial component; same budget for all."""
    all_cases=case_plan(manifest,split);chosen=[]
    for component in sorted({c['component'] for c in all_cases}):
        group=[c for c in all_cases if c['component']==component]
        tile=min({c['tile_id'] for c in group},key=lambda s:hashlib.sha256(s.encode()).hexdigest())
        positives=[c for c in group if c['tile_id']==tile and not c['negative']]
        phase=int(hashlib.sha256(component.encode()).hexdigest()[:4],16)
        for i,(a,b) in enumerate(itertools.combinations(range(3),2)):
            chosen.append(next(c for c in positives if c['light_a']==a and c['light_b']==b and c['family']==FAMILIES[(phase+i)%4]))
        negatives=[c for c in group if c['tile_id']==tile and c['negative']]
        partners=sorted({c['query_component'] for c in negatives})
        for i,partner in enumerate(partners[:3]):
            chosen.append(next(c for c in negatives if c['query_component']==partner and c['light_a']==i))
    return chosen


def load_tile(tile,cache):
    name=tile['tile_id']
    if name not in cache:
        if sha256(tile['cache'])!=tile['cache_sha256']:raise ValueError('Input cache changed')
        with np.load(tile['cache'],allow_pickle=False) as data:cache[name]={k:data[k].copy() for k in data.files}
    return cache[name]


def controlled_inputs(case,tiles,cache):
    source=load_tile(tiles[case['tile_id']],cache)
    query=load_tile(tiles[case['query_tile_id']],cache)
    a,ma=source['images'][case['light_a']],source['masks'][case['light_a']]
    b,mb=query['images'][case['light_b']],query['masks'][case['light_b']]
    if case['negative']:return a,b,ma,mb,None
    family=case['family'];angle=25. if family in ('rotation','combined') else 0.
    scale=.85 if family=='scale' else (1.15 if family=='combined' else 1.)
    matrix=cv2.getRotationMatrix2D(((a.shape[1]-1)/2,(a.shape[0]-1)/2),angle,scale)
    if family in ('translation','combined'):matrix[:,2]+=[6.,-4.]
    b=cv2.warpAffine(b,matrix,a.shape[::-1],flags=cv2.INTER_LINEAR)
    mb=cv2.warpAffine(mb.astype(np.float32),matrix,a.shape[::-1],flags=cv2.INTER_LINEAR)>=.99999
    return a,b,ma,mb,matrix


def matcher_for(method,model=None,source=None,device='cpu',dependency_dir=None):
    if method=='compact':
        from compact_lunar import CompactMatcher
        return CompactMatcher(model,device)
    if method=='sift':
        from raw_correspondences_v2 import EvidenceMatcher
        return EvidenceMatcher(sift,'sift')
    if method=='original':
        from contrastive_terrain import ContrastiveTerrainMatcher
        from raw_correspondences_v2 import EvidenceMatcher
        return EvidenceMatcher(ContrastiveTerrainMatcher('runs/contrastive-terrain-train-v1',device=device),'original')
    if method=='matchanything':
        from matchanything_matcher import MatchAnythingMatcher
        from raw_correspondences_v2 import EvidenceMatcher
        return EvidenceMatcher(MatchAnythingMatcher(dependency_dir=dependency_dir,device=device,factor=4),'matchanything')
    if method=='roma':
        from roma_matcher import RoMaMatcher
        from raw_correspondences_v2 import EvidenceMatcher
        if device!='cpu':raise ValueError('Pinned original RoMa integration is CPU only')
        return EvidenceMatcher(RoMaMatcher(),'roma')
    if method in ('loma_b','loma_r','romav2'):
        from modern_baselines_v2 import ModernMatcher
        return ModernMatcher(method,source,'common',device)
    raise ValueError('Unknown method; no fallback is permitted')


def predict(matcher,a,b,ma,mb,method):
    cv2.setRNGSeed(20261030)
    start=perf_counter();found=matcher(a,b,ma,mb)
    extra=dict(getattr(matcher,'last_evidence',{}))
    raw_points=getattr(matcher,'all_points',(found.reference_points,found.query_points))
    cv2.setRNGSeed(20261030);reverse=matcher(b,a,mb,ma)
    if getattr(matcher,'device','cpu')=='mps':
        import torch
        torch.mps.synchronize()
    features=evidence_vector(found,a.shape,extra,reverse)
    original_accepted=None
    if method=='original':
        from confidence import diagnostics
        from structural_refinement import AlignmentObjective
        from research_quality_gate import accepts as original_accepts
        policy=json.loads(Path('reports/reserve_protocol_v1.json').read_text())['policy']
        quality=AlignmentObjective(a,b,ma,mb).quality(found.matrix)
        evidence=dict(estimated=found.matrix is not None,
            selection_quality={} if quality is None else dict(quality,agreement_fraction=0.,agreeing_other_families=[]),
            **diagnostics(found,reverse,a.shape))
        original_accepted=original_accepts(evidence,policy)
    return found,reverse,features,raw_points,(perf_counter()-start)*1000,original_accepted


def evaluate(manifest_path,output,method,model=None,split='model_selection',source=None,device='cpu',
             dependency_dir=None,protocol=None,publication_commit=None,limit=None,common=False):
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(1)
    manifest=validate_manifest(json.loads(Path(manifest_path).read_text()))
    if split=='test':
        if protocol is None or publication_commit is None:raise ValueError('Final test requires publicly published frozen protocol')
        doc=validate_freeze(protocol,publication_commit)
        if sha256(manifest_path)!=doc['manifest_sha256'] or case_plan(manifest,split)!=doc['cases']:
            raise ValueError('Evaluation dataset or case plan differs from public freeze')
        key=str(model) if model else method
        if key not in doc['policy_index']:raise ValueError('Method has no frozen calibrated operating point')
        policy=json.loads(Path(doc['policy_index'][key]).read_text())
        expected_model=doc['models'].get(str(model)) if model else None
        if model and (expected_model is None or sha256(Path(model)/'selected.pt')!=expected_model):raise ValueError('Unfrozen model')
    cases=case_plan(manifest,split)
    if common:cases=common_plan(manifest,split)
    if limit is not None:
        if split=='test':raise ValueError('Final test subsets must be declared in the public protocol')
        cases=cases[:limit]
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    frozen=dict(manifest_path=str(manifest_path),manifest_sha256=sha256(manifest_path),method=method,
        model=None if model is None else str(model),split=split,cases=cases,device=device,common_subset=common,
        source_sha256={p:sha256(p) for p in SOURCES},started_utc=datetime.now(timezone.utc).isoformat())
    write_new(output/'frozen_inputs.json',frozen)
    if split=='test':
        marker=Path(protocol).with_suffix('.test-started.json')
        if not marker.exists():write_new(marker,dict(protocol_sha256=sha256(protocol),publication_commit=publication_commit,first_inference_utc=frozen['started_utc']))
    matcher=matcher_for(method,model,source,device,dependency_dir)
    # Warmup uses a generated texture, never a selected or test outcome.
    from synthetic import texture
    warm=texture(20261030,128);matcher(warm,warm)
    tiles={t['tile_id']:t for t in manifest['tiles']};cache={};rows=[]
    with (output/'results.jsonl').open('w') as log:
        for number,case in enumerate(cases):
            a,b,ma,mb,truth=controlled_inputs(case,tiles,cache)
            found,reverse,features,points,ms,original_accepted=predict(matcher,a,b,ma,mb,method)
            error=None
            if truth is not None:
                evaluation=evaluation_points(a.shape,truth)
                targets=transform(evaluation,truth)
                ia=np.rint(evaluation).astype(int);ib=np.rint(targets).astype(int).clip(0,127)
                keep=ma[ia[:,1],ia[:,0]]&mb[ib[:,1],ib[:,0]]
                error=endpoint_error(found.matrix,truth,evaluation[keep])
            row=dict(case,method=method,error_px=error,correct=bool(error is not None and error<=3 and not case['negative']),
                features=features.tolist(),transform=None if found.matrix is None else found.matrix.tolist(),
                reverse_transform=None if reverse.matrix is None else reverse.matrix.tolist(),
                reference_points=None if found.reference_points is None else found.reference_points.tolist(),
                query_points=None if found.query_points is None else found.query_points.tolist(),
                support=found.support,reason=found.reason,bidirectional_ms=ms,original_frozen_accepted=original_accepted,
                input_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in dict(reference=a,query=b,reference_mask=ma,query_mask=mb).items()})
            native_scale=np.array(tiles[case['tile_id']]['source_size'][::-1])/np.array(a.shape[::-1])
            if not np.allclose(native_scale,native_scale[0]):raise ValueError('Scalar pixel errors require isotropic source resizing')
            row.update(error_native_px=None if error is None else float(error*native_scale[0]),
                native_pixels_per_benchmark_pixel=float(native_scale[0]))
            if split=='test':
                row['accepted']=accepts(policy,{'features':row['features']})
                row['deployed_transform']=row['transform'] if row['accepted'] else None
            rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if (number+1)%24==0:print(method,split,number+1,'/',len(cases),flush=True)
    write_new(output/'results.json',rows)
    summary=dict(method=method,split=split,cases=len(rows),raw=counts(rows,[r['transform'] is not None for r in rows]),
        median_bidirectional_ms=float(np.median([r['bidirectional_ms'] for r in rows])),
        peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_new(output/'summary.json',summary)
    write_new(output/'COMPLETE.json',dict(results_sha256=sha256(output/'results.json'),frozen_inputs_sha256=sha256(output/'frozen_inputs.json')))
    return summary


def confidence(selection_run,calibration_run,output):
    a=json.loads((Path(selection_run)/'results.json').read_text())
    b=json.loads((Path(calibration_run)/'results.json').read_text())
    fa=json.loads((Path(selection_run)/'frozen_inputs.json').read_text())
    fb=json.loads((Path(calibration_run)/'frozen_inputs.json').read_text())
    if (fa['method'],fa['model'],fa['manifest_sha256'])!=(fb['method'],fb['model'],fb['manifest_sha256']):
        raise ValueError('Confidence/calibration method, model or dataset differ')
    if {r['component'] for r in a}&{r['component'] for r in b}:raise ValueError('Selection/calibration component leakage')
    manifest=json.loads(Path(fa['manifest_path']).read_text())
    allowed_a={c['pair_id'] for c in common_plan(manifest,'model_selection')}
    allowed_b={c['pair_id'] for c in common_plan(manifest,'calibration')}
    a=[r for r in a if r['pair_id'] in allowed_a];b=[r for r in b if r['pair_id'] in allowed_b]
    if {r['pair_id'] for r in a}!=allowed_a or {r['pair_id'] for r in b}!=allowed_b:
        raise ValueError('All methods require identical complete confidence-fitting/calibration case IDs')
    policy=calibrate(fit_confidence(a),b)
    policy.update(selection_results_sha256=sha256(Path(selection_run)/'results.json'),
                  calibration_results_sha256=sha256(Path(calibration_run)/'results.json'),method=fa['method'],model=fa['model'],
                  confidence_fit_cases=len(a),calibration_cases=len(b),
                  budget_note='Same metadata-selected common cases for every method. Small calibration sample gives no1percent risk guarantee.')
    write_new(output,policy);return policy


def stereo_plan(manifest,split):
    return [dict(pair_id=f'{t["tile_id"]}:3d:{i}-{j}',tile_id=t['tile_id'],component=t['component'],
        split=split,light_a=i,light_b=j,track='3d') for t in manifest['tiles'] if t['split']==split
        for i in range(3) for j in range(3)]


def stereo_evaluate(manifest_path,output,method,model=None,split='model_selection',source=None,
                    device='cpu',dependency_dir=None,protocol=None,publication_commit=None):
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(1)
    manifest=validate_manifest(json.loads(Path(manifest_path).read_text()))
    cases=stereo_plan(manifest,split)
    if split=='test':
        if protocol is None or publication_commit is None:raise ValueError('Public freeze required before3D final test')
        doc=validate_freeze(protocol,publication_commit)
        if sha256(manifest_path)!=doc['manifest_sha256'] or cases!=doc['stereo_cases']:raise ValueError('Unfrozen3D case plan')
        key=str(model) if model else method
        if key not in doc['policy_index']:raise ValueError('Unfrozen method')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    matcher=matcher_for(method,model,source,device,dependency_dir)
    tiles={t['tile_id']:t for t in manifest['tiles']};cache={};rows=[]
    from synthetic import texture
    warm=texture(20261030,128);matcher(warm,warm)
    write_new(output/'frozen_inputs.json',dict(manifest_path=str(manifest_path),manifest_sha256=sha256(manifest_path),
        method=method,model=model,split=split,cases=cases,source_sha256={p:sha256(p) for p in SOURCES}))
    for number,case in enumerate(cases):
        t=tiles[case['tile_id']];data=load_tile(t,cache)
        ia,ib=2*case['light_a'],2*case['light_b']+1
        a,b=data['stereo_images'][ia],data['stereo_images'][ib]
        ma,mb=data['stereo_masks'][ia],data['stereo_masks'][ib]
        start=perf_counter()
        from raw_correspondences_v2 import raw_points
        pa,pb=raw_points(matcher,method,a,b,ma,mb);point_source='raw_before_geometry'
        if getattr(matcher,'device','cpu')=='mps':torch.mps.synchronize()
        elapsed=(perf_counter()-start)*1000
        ka,pa_world=np.array(t['cameras'][0]['intrinsics']),np.array(t['cameras'][0]['cam2world'])
        kb,pb_world=np.array(t['cameras'][1]['intrinsics']),np.array(t['cameras'][1]['cam2world'])
        factor=np.array(t['source_size'][::-1])/128
        native=(np.asarray(pa)+.5)*factor-.5
        _,expected,world,valid=correspondences(data['depth_a'],data['depth_b'],ka,pa_world,kb,pb_world,points=native)
        expected=(expected+.5)/factor-.5
        errors=np.linalg.norm(np.asarray(pb)[valid]-expected,axis=1)
        # Map points derive solely from reference depth and source pixels. Query
        # predictions enter PnP as observations, never visibility or truth labels.
        source_index=np.rint(native).astype(int)
        reference_valid=((source_index>=0)&(source_index<np.array(data['depth_a'].shape[::-1]))).all(1)
        safe=source_index.clip(0,np.array(data['depth_a'].shape[::-1])-1)
        ranges=data['depth_a'][safe[:,1],safe[:,0]]
        reference_valid &= np.isfinite(ranges)&(ranges>0)
        map_points=world_points(native[reference_valid],ranges[reference_valid],ka,pa_world)
        cv2.setRNGSeed(20261030)
        pose=estimate_pose(map_points,(np.asarray(pb)[reference_valid]+.5)*factor-.5,kb) if len(map_points)>=6 else None
        stats=None if pose is None else pose_error(pose,pb_world)
        finite_depth=data['depth_a'][np.isfinite(data['depth_a'])&(data['depth_a']>0)]
        reference_range=float(np.median(finite_depth)) if len(finite_depth) else None
        row=dict(case,method=method,point_source=point_source,returned_matches=len(pa),
            reference_scorable_matches=int(valid.sum()),correct_correspondences=int((errors<=3).sum()),
            median_correspondence_error_px=None if not len(errors) else float(np.median(errors)),
            median_correspondence_error_native_px=None if not len(errors) else float(np.median(errors)*factor[0]),
            native_pixels_per_benchmark_pixel=float(factor[0]),
            pose_error=stats,pose_estimated=pose is not None,inference_ms=elapsed,
            estimated_cam2world=None if pose is None else pose.tolist(),reference_median_range_m=reference_range,
            pose_within_secondary_threshold=bool(stats is not None and reference_range is not None and
                stats['rotation_deg']<=2 and stats['translation_m']<=.01*reference_range),
            reference_points=np.asarray(pa).tolist(),query_points=np.asarray(pb).tolist(),
            scorable_mask=valid.tolist(),reference_target_points=expected.tolist(),
            warning='Rendered geometry. Correspondence coverage differs; raw and partial-affine-filtered frontends must not be pooled as identical3D protocols.')
        rows.append(row)
        if (number+1)%24==0:print(method,'3D',split,number+1,'/',len(cases),flush=True)
    write_new(output/'results.json',rows)
    write_new(output/'summary.json',dict(method=method,split=split,cases=len(rows),poses_estimated=sum(r['pose_estimated'] for r in rows),
        returned_matches=sum(r['returned_matches'] for r in rows),scorable_matches=sum(r['reference_scorable_matches'] for r in rows),
        correct_correspondences=sum(r['correct_correspondences'] for r in rows),
        poses_within_secondary_threshold=sum(r['pose_within_secondary_threshold'] for r in rows)))
    write_new(output/'COMPLETE.json',dict(results_sha256=sha256(output/'results.json'),frozen_inputs_sha256=sha256(output/'frozen_inputs.json')))
    return rows


def freeze(manifest_path,models,policies,output,sensor_protocol='reports/compact_sensor_protocol_v2.json'):
    manifest=validate_manifest(json.loads(Path(manifest_path).read_text()),True)
    doc=dict(schema_version=2,frozen_utc=datetime.now(timezone.utc).isoformat(),
        manifest_path=str(manifest_path),manifest_sha256=sha256(manifest_path),
        models={str(p):sha256(Path(p)/'selected.pt') for p in models},
        policies={str(p):sha256(p) for p in policies},source_sha256={p:sha256(p) for p in SOURCES},
        comparison_assets={p:sha256(p) for p in ['runs/contrastive-terrain-train-v1/selected.pt',
            'runs/contrastive-terrain-train-v1/training.json','runs/contrastive-terrain-train-v1/COMPLETE.json',
            'reports/reserve_protocol_v1.json','reports/compact_baseline_lock_v2.json',
            'reports/compact_selection_v2.json','roma_model_lock.json','matchanything_model_lock.json']},
        policy_index={(json.loads(Path(p).read_text())['model'] or json.loads(Path(p).read_text())['method']):str(p) for p in policies},
        cases=case_plan(manifest,'test'),stereo_cases=stereo_plan(manifest,'test'),
        common_cases=common_plan(manifest,'test'),sensor_protocol=str(sensor_protocol),sensor_protocol_sha256=sha256(sensor_protocol),
        success_targets=dict(correct_acceptance_gain_pp=10.,max_accepted_error=.01,max_negative_acceptance=.01,
            competitive_accuracy_tolerance_pp=3.,cpu_speedup=5.),
        secondary_3d_metric=dict(rotation_deg_max=2.,translation_max_fraction_reference_ray_range=.01,
            note='Declared simulated pose diagnostic, not a spacecraft requirement or verified navigation accuracy.'),
        pixel_units=dict(benchmark_size=[128,128],render_source_size=[512,512],
            registration_threshold_benchmark_px=3.,registration_threshold_native_px=12.,
            note='128px inference benchmark; error_px is benchmark pixels, error_native_px maps to original512px renders. POLAR1216px crop has9.5 native pixels per benchmark pixel.'),
        seeds=[20261030,20261031,20261032],
        scope='South-pole physical renders; all final results consumed once. Sensor reference uncertainty remains separate.',
        negative_dependency='Negative pairs reuse both terrain endpoints; report component dependence, not independent-binomial certainty.')
    write_new(output,doc);return doc


def validate_freeze(path,commit):
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Full public commit SHA required')
    path=Path(path);doc=json.loads(path.read_text())
    public=f'https://raw.githubusercontent.com/ThomasdeChillaz/lunar-image-matching/{commit}/{path.as_posix()}'
    with urllib.request.urlopen(public,timeout=45) as response:published=response.read()
    if hashlib.sha256(published).hexdigest()!=sha256(path):raise ValueError('Public preregistration bytes differ')
    for name,digest in {**doc['source_sha256'],**doc['policies'],**doc.get('comparison_assets',{}),doc['manifest_path']:doc['manifest_sha256']}.items():
        if sha256(name)!=digest:raise ValueError('Frozen input/source/policy changed: '+name)
    for name,digest in doc['models'].items():
        if sha256(Path(name)/'selected.pt')!=digest:raise ValueError('Frozen checkpoint changed')
    return doc


def audit(run,output):
    run=Path(run);frozen=json.loads((run/'frozen_inputs.json').read_text());complete=json.loads((run/'COMPLETE.json').read_text())
    if sha256(run/'results.json')!=complete['results_sha256'] or sha256(run/'frozen_inputs.json')!=complete['frozen_inputs_sha256']:
        raise ValueError('Completed outcomes changed')
    manifest=json.loads(Path(frozen['manifest_path']).read_text());validate_manifest(manifest,True)
    if sha256(frozen['manifest_path'])!=frozen['manifest_sha256']:raise ValueError('Dataset changed')
    tiles={t['tile_id']:t for t in manifest['tiles']};cache={}
    rows=json.loads((run/'results.json').read_text())
    if len(rows)!=len(frozen['cases']) or len({r['pair_id'] for r in rows})!=len(rows):raise ValueError('Incomplete/repeated outcomes')
    for row,case in zip(rows,frozen['cases']):
        if row['pair_id']!=case['pair_id']:raise ValueError('Case order differs')
        a,b,ma,mb,truth=controlled_inputs(case,tiles,cache)
        expected={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in dict(reference=a,query=b,reference_mask=ma,query_mask=mb).items()}
        if row['input_sha256']!=expected:raise ValueError('Native input pixels differ')
        error=None
        if truth is not None and row['transform'] is not None:
            p=evaluation_points(a.shape,truth);q=transform(p,truth)
            ia=np.rint(p).astype(int);ib=np.rint(q).astype(int).clip(0,127)
            keep=ma[ia[:,1],ia[:,0]]&mb[ib[:,1],ib[:,0]]
            if keep.any():
                predicted=p[keep]@np.array(row['transform'])[:,:2].T+np.array(row['transform'])[:,2]
                error=float(np.median(np.sqrt(np.square(predicted-q[keep]).sum(1))))
        if (error is None)!=(row['error_px'] is None) or (error is not None and not np.isclose(error,row['error_px'],atol=1e-6)):
            raise ValueError('Independent analytic endpoint replay differs')
        if row['correct']!=bool(error is not None and error<=3 and not row['negative']):raise ValueError('Correctness differs')
        if 'error_native_px' in row:
            scale=np.array(tiles[case['tile_id']]['source_size'][::-1])/np.array(a.shape[::-1])
            native_error=None if error is None else error*scale[0]
            if (native_error is None)!=(row['error_native_px'] is None) or (native_error is not None and not np.isclose(native_error,row['error_native_px'])):
                raise ValueError('Native source pixel conversion differs')
    result=dict(passed=True,rows=len(rows),source_arrays_replayed=True,independent_analytic_error=True,
                results_sha256=complete['results_sha256'])
    write_new(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    e=sub.add_parser('evaluate');e.add_argument('--manifest',required=True);e.add_argument('--output',required=True)
    e.add_argument('--method',required=True);e.add_argument('--model');e.add_argument('--split',choices=['model_selection','calibration','test'],default='model_selection')
    e.add_argument('--source');e.add_argument('--device',choices=['cpu','mps'],default='cpu');e.add_argument('--dependency-dir')
    e.add_argument('--protocol');e.add_argument('--publication-commit');e.add_argument('--limit',type=int)
    e.add_argument('--common',action='store_true')
    s=sub.add_parser('stereo')
    s.add_argument('--manifest',required=True);s.add_argument('--output',required=True);s.add_argument('--method',required=True)
    s.add_argument('--model');s.add_argument('--source');s.add_argument('--device',choices=['cpu','mps'],default='cpu')
    s.add_argument('--split',choices=['model_selection','calibration','test'],default='model_selection')
    s.add_argument('--dependency-dir');s.add_argument('--protocol');s.add_argument('--publication-commit')
    c=sub.add_parser('confidence');c.add_argument('--selection-run',required=True);c.add_argument('--calibration-run',required=True);c.add_argument('--output',required=True)
    f=sub.add_parser('freeze');f.add_argument('--manifest',required=True);f.add_argument('--models',nargs='+',required=True);f.add_argument('--policies',nargs='+',required=True);f.add_argument('--output',required=True)
    f.add_argument('--sensor-protocol',default='reports/compact_sensor_protocol_v2.json')
    v=sub.add_parser('audit');v.add_argument('run');v.add_argument('--output',required=True)
    a=vars(p.parse_args());action=a.pop('action')
    if action=='evaluate':
        a['manifest_path']=a.pop('manifest');print(json.dumps(evaluate(**a),indent=2))
    elif action=='stereo':a['manifest_path']=a.pop('manifest');stereo_evaluate(**a)
    elif action=='confidence':confidence(**a)
    elif action=='freeze':a['manifest_path']=a.pop('manifest');freeze(**a)
    else:audit(**a)
