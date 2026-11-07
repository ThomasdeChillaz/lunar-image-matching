"""Training-image crop/full-feature control; no fitting or accuracy claim."""
import argparse
from datetime import datetime,timezone
import json,subprocess
from pathlib import Path
from time import perf_counter
import numpy as np

from native_crop_alignment_v57 import native_example


def error_stats(a,b):
    a,b=[np.asarray(x,dtype=float)for x in (a,b)]
    if a.shape!=b.shape or a.ndim!=2 or not len(a) or not np.isfinite(a).all()or not np.isfinite(b).all():
        raise ValueError('Finite nonempty paired feature matrices required')
    e=a-b;l2=np.sqrt((e*e).sum(1))
    return dict(max_abs=float(np.abs(e).max()),mean_l2=float(l2.mean()),max_l2=float(l2.max()))


def run(output,commit):
    from north_optimization_probe_v54 import read,sha,fixed_specs,grid128
    from north_coarse_reachability_v53 import verify_ended
    verify_ended('runs/north-v56-rate-development-state.json')
    audit=read('reports/north_rate_development_audit_v56.json')
    if not audit['passed']or audit['replayed_rows']!=576 or audit['recorded_updates']!=2000:
        raise ValueError('Finished full development replay required before another neural job')
    qa=read('reports/native_crop_software_v57.json')
    if not qa['passed']or not qa['full_repository_regression']or not qa['fresh_smoke']:
        raise ValueError('Current software checks required')
    for name,digest in qa['source_sha256'].items():
        if sha(name)!=digest:raise ValueError('Source changed after software checks')
    from train_north_control_v39 import SOURCES as OLD_SOURCES
    sources=list(dict.fromkeys(OLD_SOURCES+['native_crop_alignment_v57.py','probe_native_crop_v57.py',
        'test_native_crop_alignment_v57.py','verify_native_crop_v57.py','test_native_crop_probe_v57.py',
        'north_coarse_reachability_v53.py','north_optimization_probe_v54.py']))
    if len(commit)!=40 or any(c not in '0123456789abcdef'for c in commit):raise ValueError('Exact public source required')
    for name in sources:
        if subprocess.check_output(['git','show',commit+':'+name])!=Path(name).read_bytes():raise ValueError('Control source differs from public freeze')
    root=Path(output)
    if root.exists():raise FileExistsError('Preserve every control attempt')
    import cv2,torch
    from compact_lunar import CompactMatcher
    from contrastive_terrain import sample_descriptors
    from north_training_bundle_v38 import load_bundle
    from train_search_window_v18 import configure
    configure();started=datetime.now(timezone.utc);tick=perf_counter();cap=300
    root.mkdir(parents=True);(root/'arrays').mkdir()
    manifest_path='reports/north_training_data_v38.json'
    protocol=dict(publication_commit=commit,source_sha256={n:sha(n)for n in sources},
        created_utc=started.isoformat(),max_calendar_seconds=cap,cpu_threads=2,
        manifest_sha256=sha(manifest_path),upstream_audit_sha256=sha('reports/north_rate_development_audit_v56.json'),
        software_sha256=sha('reports/native_crop_software_v57.json'),points_per_fixture=8,
        aligned_max_abs_tolerance=1e-4,legacy_fractional_offset=[.25,-.375],
        scope='Four fixed TRAINING-image conditions. Measure feature extraction consistency at identical coordinates. No optimizer, pose accuracy, model adoption or causal attribution to historical pose failures.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def clock():
        wall=(datetime.now(timezone.utc)-started).total_seconds()
        if wall<0 or wall>=cap or perf_counter()-tick>=cap:raise TimeoutError('Fixed five-minute crop-control allowance exhausted')
    try:
        manifest,tiles,data=load_bundle(manifest_path,train_only=True);specs=fixed_specs(manifest)
        coarse=CompactMatcher(manifest['coarse_model']);coarse.model.eval().requires_grad_(False)
        full={};rows=[];calls=0
        def dense(im):
            nonlocal calls
            clock();calls+=1
            with torch.inference_mode():return coarse.model(torch.from_numpy(im.copy()).float()[None,None]/255)[0]
        def at(features,xy):
            with torch.inference_mode():return sample_descriptors(features,torch.from_numpy(np.asarray(xy,np.float32))[None])[0].numpy()
        points=(grid128()+.5)*4-.5
        for spec in specs:
            z=data[spec['tile_id']];a,b=z['images'][spec['reference_image']],z['images'][spec['query_image']]
            for image_index,image in ((spec['reference_image'],a),(spec['query_image'],b)):
                if image_index not in full:full[image_index]=dense(image)
            targets=points if spec['identity_geometry']else z['forward_target']
            eligible=[]
            for i,(p,t,v)in enumerate(zip(points,targets,z['forward_visible'])):
                center=np.rint(t+[7,-5])
                if v and np.isfinite(t).all()and(center>=30).all()and(center<=481).all():eligible.append(i)
            if len(eligible)<8:raise ValueError('Eight geometry-supported control points required')
            selected=np.asarray(eligible)[np.linspace(0,len(eligible)-1,8,dtype=int)]
            for order,i in enumerate(selected):
                clock();p,t=points[i],targets[i];e=native_example(a,b,p,t+[7,-5],t,True)
                ref_patch=dense(e['reference_patch']);query_patch=dense(e['query_patch'])
                ref_full=at(full[spec['reference_image']],p[None]);ref_crop=at(ref_patch,e['reference_xy'][None])
                cx,cy=e['query_center'];native_full=full[spec['query_image']][0,:,cy-16:cy+17,cx-16:cx+17].permute(1,2,0).reshape(-1,32).numpy()
                native_crop=query_patch[0,:,12:45,12:45].permute(1,2,0).reshape(-1,32).numpy()
                # Same fractional coordinates, two orders: image interpolation
                # before CNN versus interpolation of already computed features.
                frac=e['query_center']+np.array([.25,-.375])
                legacy=cv2.getRectSubPix(b,(57,57),tuple(map(float,frac)))
                legacy_features=dense(legacy)[0,:,12:45,12:45].permute(1,2,0).reshape(-1,32).numpy()
                yy,xx=np.mgrid[-16:17,-16:17]
                coordinates=np.column_stack([xx.ravel(),yy.ravel()])+frac
                fractional_full=at(full[spec['query_image']],coordinates)
                name=spec['name']+'-'+str(order);path=root/'arrays'/(name+'.npz')
                np.savez_compressed(path,reference_full=ref_full,reference_crop=ref_crop,
                    native_full=native_full,native_crop=native_crop,
                    fractional_full=fractional_full,legacy_fractional_crop=legacy_features)
                rows.append(dict(fixture=spec['name'],point_index=int(i),reference_native=p.tolist(),
                    exact_target_native=t.tolist(),query_center=e['query_center'].tolist(),
                    arrays=str(path.relative_to(root)),arrays_sha256=sha(path),
                    aligned_reference=error_stats(ref_full,ref_crop),aligned_query=error_stats(native_full,native_crop),
                    fractional_query=error_stats(fractional_full,legacy_features)))
        clock();passed=all(max(x['aligned_reference']['max_abs'],x['aligned_query']['max_abs'])<=1e-4 for x in rows)
        result=dict(complete=True,aligned_controls_passed=passed,points=len(rows),neural_forward_calls=calls,
            rows=rows,optimizer_updates=0,calibration_or_final_pixels_loaded=0,model_promoted=False,
            protocol_sha256=sha(root/'protocol.json'),calendar_seconds=(datetime.now(timezone.utc)-started).total_seconds(),scope=protocol['scope'])
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        (root/'COMPLETE.json').write_text(json.dumps(dict(sha256={str(p.relative_to(root)):sha(p)for p in sorted(root.rglob('*'))if p.is_file()and p!=root/'COMPLETE.json'}),indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items()if k!='rows'},indent=2),flush=True);return result
    except BaseException as exc:
        (root/'FAILED.json').write_text(json.dumps(dict(error_type=type(exc).__name__,error=str(exc)),indent=2)+'\n');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='runs/native-crop-control-v57')
    p.add_argument('--publication-commit',required=True);a=p.parse_args();run(a.output,a.publication_commit)
