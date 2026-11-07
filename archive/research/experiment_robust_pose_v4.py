"""Compare common pose solvers on development matches, then diagnose ceilings."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from dataset import sha256
from localization_diagnostics_v3 import map_points
from localization_robust_pose_v4 import CONFIGS, estimate
from lunar_geometry_v2 import pose_error
from lunargeo_dataset import write_new

MANIFEST='reports/localization_training_data_v3.json'
BASELINE='runs/localization-diagnostics-v3-validated/results.json.gz'
SOURCES=['experiment_robust_pose_v4.py','localization_robust_pose_v4.py',
         'localization_pose_refit_v3.py','localization_diagnostics_v3.py','lunar_geometry_v2.py']


def read_gzip(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def load():
    manifest=json.loads(Path(MANIFEST).read_text())
    if manifest['calibration_or_test_pixels_loaded']!=0:raise ValueError('Development-only manifest required')
    tiles={t['tile_id']:t for t in manifest['tiles'] if t['split']=='model_selection'}
    data={}
    for key,t in tiles.items():
        if sha256(t['native_cache'])!=t['native_cache_sha256']:raise ValueError('Native cache differs')
        with np.load(t['native_cache'],allow_pickle=False) as z:data[key]={k:z[k].copy() for k in z.files}
    native=[r for r in read_gzip(BASELINE) if r['experiment']=='deterministic_native']
    for r in native:
        r['pair_id']=r['tile_id']+f'/lights-{r["light_a"]}-{r["light_b"]}'
        r['key']=3*r['light_a']+r['light_b']
    models={'frozen_native':native};paths={'frozen_native':BASELINE}
    for seed in (20261101,20261102,20261103):
        path=f'runs/localization-v3-balanced-development/reliability-{seed}/development_results.json.gz'
        complete=json.loads(Path(path).with_name('COMPLETE.json').read_text())
        if sha256(path)!=complete['results_sha256']:raise ValueError('Frozen model results differ')
        name=f'balanced_{seed}';models[name]=read_gzip(path);paths[name]=path
    expected={c['pair_id'] for c in manifest['cases'] if c['split']=='model_selection'}
    if any(len(rows)!=72 or {r['pair_id'] for r in rows}!=expected for rows in models.values()):
        raise ValueError('All eligible common development pairs required')
    return manifest,tiles,data,models,paths


def score(row,tiles,data,config,mode='predicted'):
    t=tiles[row['tile_id']];z=data[row['tile_id']];a,b=t['cameras']
    p=np.asarray(row['reference_points_native'],float).reshape(-1,2)
    q=np.asarray(row['query_points_native'],float).reshape(-1,2)
    if mode!='predicted':
        if mode in ('oracle_all_visible','oracle_window'):
            k=row['key'];p=z[f'p{k}'];q=z[f'target{k}'];use=z[f'visible{k}'].copy()
            if mode=='oracle_window':use&=(np.abs(q-np.rint(z[f'q{k}']))<=16).all(1)
            # Same native feature-support constraints as the compact refiner.
            use&=((p>=13)&(p<np.array(z['depth_a'].shape[::-1])-13)).all(1)
            center=np.rint(z[f'q{k}'])
            use&=((center>=28)&(center<np.array(z['depth_b'].shape[::-1])-28)).all(1)
        elif mode=='oracle_correct_predicted':
            visible=np.asarray(row['visible'],bool)
            target=np.asarray(row['target_points_native'],float).reshape(-1,2)
            use=np.zeros(len(p),bool);use[visible]=np.linalg.norm(q[visible]-target,axis=1)<=2
        else:raise ValueError('Unknown diagnostic')
        p,q=p[use],q[use]
    world,valid=map_points(p,z['depth_a'],a['intrinsics'],a['cam2world'])
    tick=perf_counter();pose,info=estimate(world,q[valid],b['intrinsics'],config)
    elapsed=(perf_counter()-tick)*1000
    # Query pose is consumed only after estimation. Oracle modes are explicitly
    # labelled ground-truth-assisted controls and never candidate inference.
    error=None if pose is None else pose_error(pose,np.asarray(b['cam2world']))
    da=z['depth_a'];distance=float(np.median(da[np.isfinite(da)&(da>0)]))
    correct=bool(error and error['rotation_deg']<=2 and error['translation_m']<=.01*distance)
    strict=bool(error and error['rotation_deg']<=.5 and error['translation_m']<=.001*distance)
    return dict(pair_id=row['pair_id'],tile_id=row['tile_id'],component=t['component'],mode=mode,
                solver=config['name'],matches=len(p),pose_correct=correct,strict_correct=strict,
                pose_error=error,pose=None if pose is None else pose.tolist(),estimator=info,
                reference_range_m=distance,latency_ms=elapsed)


def summarize(rows):
    return dict(cases=len(rows),correct=sum(r['pose_correct'] for r in rows),
                strict=sum(r['strict_correct'] for r in rows),estimated=sum(r['pose'] is not None for r in rows),
                median_pose_ms=float(np.median([r['latency_ms'] for r in rows])))


def run(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Use new immutable output')
    output.mkdir(parents=True);cv2.setNumThreads(1)
    manifest,tiles,data,models,paths=load();bank=set(manifest['selection_bank'])
    protocol=dict(scope='Development solver hypotheses; no final-test/calibration access.',
        created_utc=datetime.now(timezone.utc).isoformat(),configs=CONFIGS,selection_bank=sorted(bank),
        selection_rule='Joint correct-pose count on baseline and predefined primary bank; then baseline count; then declared config order.',
        input_hashes={p:sha256(p) for p in [MANIFEST]+list(paths.values())},
        sources={p:sha256(p) for p in SOURCES},calibration_or_test_pixels_loaded=0,
        inference_inputs='Reference 3D geometry, query intrinsics and saved image correspondences only.',
        oracle_modes='GT-assisted diagnostic ceilings only; never reported as model inference.',
        opencv=cv2.__version__)
    write_new(output/'protocol.json',protocol)
    for name in SOURCES:
        p=output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(name).read_bytes())
    records=[];selection=[]
    for config in CONFIGS:
        per={}
        for name in ('frozen_native','balanced_20261101'):
            rows=[dict(score(r,tiles,data,config),model=name,stage='selection') for r in models[name] if r['pair_id'] in bank]
            records.extend(rows);per[name]=summarize(rows)
        selection.append(dict(config=config['name'],results=per))
        print(config['name'],per,flush=True)
    def key(item):
        a,b=item['results']['frozen_native']['correct'],item['results']['balanced_20261101']['correct']
        return a+b,a,-next(i for i,c in enumerate(CONFIGS) if c['name']==item['config'])
    chosen=max(selection,key=key)['config'];config=next(c for c in CONFIGS if c['name']==chosen)
    full={};controls={}
    for name,source in models.items():
        for c in [CONFIGS[0]]+([] if chosen==CONFIGS[0]['name'] else [config]):
            rows=[dict(score(r,tiles,data,c),model=name,stage='expanded_development') for r in source]
            records.extend(rows);full[name+'/'+c['name']]=summarize(rows)
        for mode in ('oracle_all_visible','oracle_window','oracle_correct_predicted'):
            rows=[dict(score(r,tiles,data,config,mode),model=name,stage='oracle_diagnostic') for r in source]
            records.extend(rows);controls[name+'/'+mode]=summarize(rows)
        print(name,'full', {k:v for k,v in full.items() if k.startswith(name+'/')},flush=True)
    with gzip.open(output/'results.json.gz','wt') as f:json.dump(records,f,allow_nan=False)
    summary=dict(scope=protocol['scope'],selected_solver=chosen,selection=selection,
        expanded_development=full,oracle_controls=controls,records=len(records),results_sha256=sha256(output/'results.json.gz'))
    write_new(output/'summary.json',summary);print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
