"""Blind free-CPU baseline inference and independent local scoring of outputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
from time import perf_counter
import zipfile

import cv2
import numpy as np

from compact_confidence import evidence_vector,counts
from dataset import sha256
from geometry import endpoint_error,evaluation_points,transform
from lunargeo_dataset import write_new
from matchers import fit_geometry,Match


def pack(manifest_path,output,splits=('model_selection','calibration'),protocol=None,publication_commit=None):
    from compact_research import common_plan,controlled_inputs,validate_manifest
    manifest=validate_manifest(json.loads(Path(manifest_path).read_text()))
    if 'test' in splits:
        from compact_research import validate_freeze
        if protocol is None or publication_commit is None:raise ValueError('Final baseline pixels require public model/policy freeze')
        frozen=validate_freeze(protocol,publication_commit)
        if sha256(manifest_path)!=frozen['manifest_sha256']:raise ValueError('Baseline test dataset differs from freeze')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    tiles={t['tile_id']:t for t in manifest['tiles']};cache={};entries=[]
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for split in splits:
            for number,case in enumerate(common_plan(manifest,split)):
                a,b,ma,mb,_=controlled_inputs(case,tiles,cache)
                blob=io.BytesIO();np.savez_compressed(blob,reference=a,query=b,reference_mask=ma,query_mask=mb)
                member=f'{split}/{number:04d}.npz';content=blob.getvalue();archive.writestr(member,content)
                entries.append(dict(pair_id=case['pair_id'],split=split,component=case['component'],
                    member=member,sha256=hashlib.sha256(content).hexdigest(),
                    input_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in dict(reference=a,query=b,reference_mask=ma,query_mask=mb).items()}))
        doc=dict(schema_version=2,manifest_sha256=sha256(manifest_path),cases=entries,
            source_sha256=sha256(__file__),license='CC BY4.0 derived MoonAnything pixels; original code MIT',
            attribution='C. Grethen,Y. Shi,S. Gasparini,G. Morin; MoonAnything DOI10.5281/zenodo.18415119. Derived grayscale128px, masks and controlled geometry.',
            protocol='No correctness labels or ground-truth transforms in this inference package.')
        if 'test' in splits:
            doc.update(publication_commit=publication_commit,protocol_sha256=sha256(protocol),protocol_path=str(protocol))
        archive.writestr('inputs.json',json.dumps(doc,indent=2))
    print('Packed',len(entries),'cases',output,sha256(output),flush=True)
    return doc


def infer(package,output,method,source=None,shard=0,shards=1,control_model=None):
    from compact_research import matcher_for
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(1)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    with zipfile.ZipFile(package) as archive:
        doc=json.loads(archive.read('inputs.json'))
        cases=doc['cases'][shard::shards]
        matcher=matcher_for(method,source=source,device='cpu')
        torch.set_num_threads(2)
        from synthetic import texture
        warm=texture(20261030,128);matcher(warm,warm)
        truth=np.array([[1.,0,8.],[0,1.,-4.]])
        shifted=cv2.warpAffine(warm,truth,(128,128))
        control=matcher(warm,shifted)
        control_error=endpoint_error(control.matrix,truth,evaluation_points(warm.shape,truth))
        if control_error is None or control_error>3:raise ValueError('Actual pretrained known-transform control failed')
        control=None
        if control_model:
            from compact_lunar import CompactMatcher
            control=CompactMatcher(control_model)
            control(warm,warm)
        rows=[];control_rows=[]
        for number,case in enumerate(cases):
            content=archive.read(case['member'])
            if hashlib.sha256(content).hexdigest()!=case['sha256']:raise ValueError('Native input package changed')
            with np.load(io.BytesIO(content),allow_pickle=False) as z:
                a,b,ma,mb=[z[n] for n in ('reference','query','reference_mask','query_mask')]
            start=perf_counter();cv2.setRNGSeed(20261030);forward=matcher(a,b,ma,mb)
            extra=dict(getattr(matcher,'last_evidence',{}))
            cv2.setRNGSeed(20261030);reverse=matcher(b,a,mb,ma)
            features=evidence_vector(forward,a.shape,extra,reverse)
            rows.append(dict(pair_id=case['pair_id'],split=case['split'],component=case['component'],
                method=method,transform=None if forward.matrix is None else forward.matrix.tolist(),
                reverse_transform=None if reverse.matrix is None else reverse.matrix.tolist(),
                reference_points=None if forward.reference_points is None else forward.reference_points.tolist(),
                query_points=None if forward.query_points is None else forward.query_points.tolist(),
                support=forward.support,reason=forward.reason,features=features.tolist(),
                bidirectional_ms=(perf_counter()-start)*1000,input_sha256=case['input_sha256']))
            if control is not None:
                cv2.setRNGSeed(20261030);started=perf_counter();cf=control(a,b,ma,mb)
                extra=dict(control.last_evidence)
                cv2.setRNGSeed(20261030);cr=control(b,a,mb,ma)
                features=evidence_vector(cf,a.shape,extra,cr)
                control_rows.append(dict(pair_id=case['pair_id'],model=control_model,
                    checkpoint_sha256=sha256(Path(control_model)/'selected.pt'),
                    transform=None if cf.matrix is None else cf.matrix.tolist(),features=features.tolist(),
                    same_host_bidirectional_ms=(perf_counter()-started)*1000))
            print(method,'blind',number+1,'/',len(cases),flush=True)
        write_new(output/'predictions.json',rows)
        if control_rows:write_new(output/'timing_control.json',control_rows)
        write_new(output/'COMPLETE.json',dict(predictions_sha256=sha256(output/'predictions.json'),
            package_sha256=sha256(package),source_sha256=sha256(__file__),shard=shard,shards=shards,
            method_provenance=getattr(matcher,'provenance',None),torch=torch.__version__,cpu_threads=torch.get_num_threads(),
            software_control_endpoint_px=control_error,platform=platform.platform(),
            peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_rss_unit='bytes' if platform.system()=='Darwin' else 'KiB'))
    return rows


def score(manifest_path,executions,output):
    from compact_research import common_plan,controlled_inputs,validate_manifest
    manifest=validate_manifest(json.loads(Path(manifest_path).read_text()),True)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    rows=[];origins=[]
    for path in sorted(Path(executions).rglob('predictions.json')):
        complete=json.loads(path.with_name('COMPLETE.json').read_text())
        if sha256(path)!=complete['predictions_sha256']:raise ValueError('Blind predictions changed')
        rows.extend(json.loads(path.read_text()));origins.append(dict(path=str(path),completion=complete))
    if not rows:raise ValueError('No completed blind predictions')
    tiles={t['tile_id']:t for t in manifest['tiles']};cache={};seen=set();out=[]
    for row in rows:
        key=(row['pair_id'],row['method'])
        if key in seen:raise ValueError('Duplicate baseline outcome')
        seen.add(key)
        cases={c['pair_id']:c for c in common_plan(manifest,row['split'])}
        case=cases[row['pair_id']]
        a,b,ma,mb,truth=controlled_inputs(case,tiles,cache)
        hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in dict(reference=a,query=b,reference_mask=ma,query_mask=mb).items()}
        if hashes!=row['input_sha256']:raise ValueError('Blind baseline native pixels differ')
        error=None
        if truth is not None:
            p=evaluation_points(a.shape,truth);q=transform(p,truth)
            ia=np.rint(p).astype(int);ib=np.rint(q).astype(int).clip(0,127)
            p=p[ma[ia[:,1],ia[:,0]]&mb[ib[:,1],ib[:,0]]]
            error=endpoint_error(row['transform'],truth,p)
        out.append(dict(case,**{k:v for k,v in row.items() if k not in case},error_px=error,
            correct=error is not None and error<=3 and not case['negative']))
    for method in {r['method'] for r in out}:
        for split in {r['split'] for r in out if r['method']==method}:
            selected=[r for r in out if r['method']==method and r['split']==split]
            order={c['pair_id']:i for i,c in enumerate(common_plan(manifest,split))}
            selected.sort(key=lambda r:order[r['pair_id']])
            if {r['pair_id'] for r in selected}!={c['pair_id'] for c in common_plan(manifest,split)}:
                raise ValueError('Incomplete common baseline case coverage')
            folder=output/method/split;folder.mkdir(parents=True)
            write_new(folder/'results.json',selected)
            write_new(folder/'frozen_inputs.json',dict(manifest_path=str(manifest_path),manifest_sha256=sha256(manifest_path),
                method=method,model=None,split=split,cases=common_plan(manifest,split),origins=origins))
            write_new(folder/'COMPLETE.json',dict(results_sha256=sha256(folder/'results.json'),frozen_inputs_sha256=sha256(folder/'frozen_inputs.json')))
    write_new(output/'audit.json',dict(passed=True,blind_predictions=len(out),native_pixels_verified=True,
        results_case_coverage_verified=True,origins=origins))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('pack');a.add_argument('--manifest',required=True);a.add_argument('--output',required=True)
    a.add_argument('--splits',nargs='+',default=['model_selection','calibration'])
    a.add_argument('--protocol');a.add_argument('--publication-commit')
    b=sub.add_parser('infer');b.add_argument('--package',required=True);b.add_argument('--output',required=True);b.add_argument('--method',required=True)
    b.add_argument('--source');b.add_argument('--shard',type=int,default=0);b.add_argument('--shards',type=int,default=1)
    b.add_argument('--control-model')
    c=sub.add_parser('score');c.add_argument('--manifest',required=True);c.add_argument('--executions',required=True);c.add_argument('--output',required=True)
    args=vars(p.parse_args());action=args.pop('action')
    if action=='pack':args['manifest_path']=args.pop('manifest');pack(**args)
    elif action=='score':args['manifest_path']=args.pop('manifest');score(**args)
    else:infer(**args)
