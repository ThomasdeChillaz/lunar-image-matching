"""Frozen zero-shot controls on real orbital pixels, separate from navigation.

Known warps of one real observation provide exact pixel-geometry supervision.
Different observations provide diagnostics only: publisher map coordinates have
unknown uncertainty and must never become independent accuracy labels.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import load_pair, safe_path, sha256, validate_manifest
from evaluate import endpoint_statistics
from matchers import sift
from reserve_evaluation import SEED, array_hashes, infer, metrics, validate_freeze
from research_geometry import FAMILIES, prepare


def freeze(source, terrain_protocol, output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    terrain=validate_freeze(terrain_protocol)
    manifest=json.loads(Path(source).read_text())
    assets=validate_manifest(manifest)
    cases=[]
    for asset in sorted(assets.values(),key=lambda a:a['asset_id']):
        for repeat in range(2):
            for family in FAMILIES:
                cases.append(dict(pair_id=f'{asset["asset_id"]}_real_warp_{repeat}_{family}',
                    asset_id=asset['asset_id'],region_id=asset['region_id'],geometry_family=family,
                    dataset_kind='real_photo_known_warp',label_source='exact_geometry'))
        for other_region in sorted({a['region_id'] for a in assets.values()}-{asset['region_id']}):
            partner=sorted([a for a in assets.values() if a['region_id']==other_region],key=lambda a:a['asset_id'])[0]
            cases.append(dict(pair_id=f'{asset["asset_id"]}_real_negative_{partner["asset_id"]}',
                asset_id=asset['asset_id'],query_asset_id=partner['asset_id'],region_id=asset['region_id'],
                geometry_family='nonoverlap_negative',dataset_kind='real_photo_nonoverlap',label_source='known_nonoverlap'))
    files=['real_sensor_controls.py','reserve_evaluation.py','dataset.py','research_geometry.py',
        'research_quality_gate.py','contrastive_terrain.py','structural_refinement.py','confidence.py','matchers.py','evaluate.py','geometry.py']
    result=dict(schema_version=1,frozen_utc=datetime.now(timezone.utc).isoformat(),manifest=manifest,
        source_manifest_sha256=sha256(source),terrain_protocol_sha256=sha256(terrain_protocol),
        terrain_protocol_path=str(terrain_protocol),training=terrain['training'],policy=terrain['policy'],
        source_sha256={f:sha256(f) for f in files},cases=cases,
        repeat_pairs=manifest['pairs'],methods=['terrain_raw','terrain_deployed','sift'],
        preprocessing='Native center-square crop, fixed INTER_AREA downsample to128. Native nodata mask area-fraction must equal1. No per-image contrast fit.',
        scope='Frozen Tycho-trained model; zero fitting on these pixels. S1/S2/S3 are three previously inspected source scenes, not 42 independent terrains or an untouched new geographic reserve.',
        accuracy_boundary='Exact geometry is known only for warps of a single real photo. Repeat-observation map disagreement and cycles are unverified diagnostics. No independently verified orbital navigation accuracy or metric accuracy.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),regions=sorted({a['region_id'] for a in assets.values()}),
        repeat_diagnostics=len(manifest['pairs']),freeze_sha256=sha256(output))),flush=True)


def real_image(asset):
    import rasterio
    path=safe_path('.',asset['path'])
    if sha256(path)!=asset['sha256']:
        raise ValueError('Real source image changed')
    with rasterio.open(path) as ds:
        if ds.count!=1 or ds.dtypes[0]!='uint8':
            raise ValueError('Expected original single-band uint8 real observation')
        image=ds.read(1)
        mask=ds.read_masks(1)>0
    size=min(image.shape)
    y,x=[(d-size)//2 for d in image.shape]
    image,mask=image[y:y+size,x:x+size],mask[y:y+size,x:x+size]
    small=cv2.resize(np.where(mask,image,0),(128,128),interpolation=cv2.INTER_AREA)
    valid=cv2.resize(mask.astype(np.float32),(128,128),interpolation=cv2.INTER_AREA)>=1.-1e-6
    small[~valid]=0
    return small,valid


def run(protocol,output):
    import torch
    from contrastive_terrain import ContrastiveTerrainMatcher
    doc=json.loads(Path(protocol).read_text())
    terrain=validate_freeze(doc['terrain_protocol_path'])
    if sha256(doc['terrain_protocol_path'])!=doc['terrain_protocol_sha256']:
        raise ValueError('Changed frozen model/gate contract')
    for f,digest in doc['source_sha256'].items():
        if sha256(f)!=digest:
            raise ValueError('Changed real-control source')
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output/'frozen_protocol.json').write_bytes(Path(protocol).read_bytes())
    (output/'started_utc.txt').write_text(datetime.now(timezone.utc).isoformat())
    assets=validate_manifest(doc['manifest'],'.')
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    model=ContrastiveTerrainMatcher(doc['training'])
    images,rows={},[]
    def image(name):
        if name not in images:
            images[name]=real_image(assets[name])
        return images[name]
    for i,case in enumerate(doc['cases'],1):
        a,ma=image(case['asset_id'])
        if case['label_source']=='known_nonoverlap':
            b,mb=image(case['query_asset_id'])
            points,targets=None,None
        else:
            b,mb,_,points,targets,_=prepare(a,a,ma,ma,case['pair_id'],case['geometry_family'])
        prediction=infer(a,b,ma,mb,model,doc['policy'])
        cv2.setRNGSeed(SEED)
        baseline=sift(a,b,ma,mb)
        for method,fit in [('terrain_raw',prediction['raw']),('terrain_deployed',prediction['deployed']),('sift',baseline)]:
            error,rmse=(None,None) if points is None else endpoint_statistics(fit.matrix,points,targets)
            row=dict(**case,method=method,estimated=fit.matrix is not None,
                correct=bool(error is not None and error<=3),error_px=error,rmse_px=rmse,
                transform=None if fit.matrix is None else fit.matrix.tolist(),
                input_array_sha256=array_hashes(a,b,ma,mb),gate_accepted=prediction['accepted'] if method!='sift' else None)
            rows.append(row)
        if i%72==0:
            print(f'{i}/{len(doc["cases"])} real-photo control cases',flush=True)
    diagnostics=[]
    for i,pair in enumerate(doc['repeat_pairs'],1):
        a,b,ma,mb,_=load_pair(pair,assets,'.',size=128)
        prediction=infer(a,b,ma,mb,model,doc['policy'])
        raw=prediction['raw']
        probes=np.array([[x,y] for y in range(16,112,16) for x in range(16,112,16)])
        probes=probes[ma[probes[:,1],probes[:,0]] & mb[probes[:,1],probes[:,0]]]
        map_error,_=endpoint_statistics(raw.matrix,probes,probes)
        diagnostics.append(dict(pair_id=pair['pair_id'],region_id=pair['region_id'],
            dataset_kind='real_repeat_diagnostic',label_source='unverified_map',correct=None,
            independent_error_px=None,reference_uncertainty_px=None,
            provisional_map_disagreement_px=map_error,raw_estimated=raw.matrix is not None,
            gate_accepted=prediction['accepted'],transform=None if raw.matrix is None else raw.matrix.tolist(),
            input_array_sha256=array_hashes(a,b,ma,mb),**prediction['evidence']))
        if i%40==0:
            print(f'{i}/{len(doc["repeat_pairs"])} unverified repeat diagnostics',flush=True)
    summary=dict(by_region={},by_method={},repeat_diagnostics={},scope=doc['scope'],accuracy_boundary=doc['accuracy_boundary'])
    for method in doc['methods']:
        summary['by_method'][method]=metrics([r for r in rows if r['method']==method])
        for region in sorted({a['region_id'] for a in assets.values()}):
            summary['by_region'][f'{method}:{region}']=metrics([r for r in rows if r['method']==method and r['region_id']==region])
    for region in sorted({r['region_id'] for r in diagnostics}):
        selected=[r for r in diagnostics if r['region_id']==region]
        summary['repeat_diagnostics'][region]=dict(cases=len(selected),raw_fits=sum(r['raw_estimated'] for r in selected),
            gate_accepted=sum(r['gate_accepted'] for r in selected),independent_accuracy='unavailable')
    (output/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    (output/'repeat_diagnostics.json').write_text(json.dumps(diagnostics,indent=2,allow_nan=False))
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (output/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows),repeat_diagnostics=len(diagnostics),
        freeze_sha256=sha256(protocol),results_sha256=sha256(output/'results.json'),
        diagnostics_sha256=sha256(output/'repeat_diagnostics.json'),summary_sha256=sha256(output/'summary.json')),indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','run'])
    p.add_argument('--source',default='expanded_solar_manifest.json')
    p.add_argument('--terrain-protocol',default='reports/reserve_protocol_v1.json')
    p.add_argument('--protocol',required=True)
    p.add_argument('--output')
    args=p.parse_args()
    if args.action=='freeze':
        freeze(args.source,args.terrain_protocol,args.protocol)
    else:
        run(args.protocol,args.output)
