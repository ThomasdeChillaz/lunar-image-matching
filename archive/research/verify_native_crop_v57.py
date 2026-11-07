"""Independent saved-feature arithmetic replay; no neural forward."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from north_optimization_probe_v54 import read,sha


def independent_stats(a,b):
    delta=np.asarray(a,np.float64)-np.asarray(b,np.float64)
    lengths=np.linalg.norm(delta,axis=1)
    return dict(max_abs=float(np.max(np.abs(delta))),mean_l2=float(np.mean(lengths)),max_l2=float(np.max(lengths)))


def audit(run,output):
    root=Path(run);out=Path(output)
    if out.exists():raise FileExistsError('Preserve prior replay')
    summary=read(root/'summary.json');protocol=read(root/'protocol.json')
    if (not summary['complete']or summary['points']!=32 or summary['neural_forward_calls']!=100 or summary['optimizer_updates']!=0
            or summary['model_promoted']or summary['calibration_or_final_pixels_loaded']!=0
            or not 0<=summary['calendar_seconds']<=300
            or summary['protocol_sha256']!=sha(root/'protocol.json')):
        raise ValueError('Complete bounded training-only crop control required')
    sealed=read(root/'COMPLETE.json')['sha256']
    if {str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'COMPLETE.json'}!=set(sealed):
        raise ValueError('Complete feature artifact inventory differs')
    for name,digest in sealed.items():
        if sha(root/name)!=digest:raise ValueError('Sealed feature evidence changed')
    for name,digest in protocol['source_sha256'].items():
        if sha(name)!=digest or subprocess.check_output(['git','show',protocol['publication_commit']+':'+name])!=Path(name).read_bytes():
            raise ValueError('Executed/public source differs')
    names=('identity','relighting','viewpoint','relighting_and_viewpoint')
    if len(summary['rows'])!=32 or any(sum(r['fixture']==name for r in summary['rows'])!=8 for name in names):
        raise ValueError('All four declared training fixtures/eight points each required')
    seen=set();passed=True
    pairs={'aligned_reference':('reference_full','reference_crop'),
        'aligned_query':('native_full','native_crop'),
        'fractional_query':('fractional_full','legacy_fractional_crop')}
    for row in summary['rows']:
        key=(row['fixture'],row['point_index'])
        if key in seen:raise ValueError('Duplicate control point')
        seen.add(key);path=root/row['arrays']
        if sha(path)!=row['arrays_sha256']:raise ValueError('Feature array hash differs')
        with np.load(path,allow_pickle=False)as z:
            if set(z.files)!={v for p in pairs.values()for v in p}:raise ValueError('Feature array members differ')
            for field,(a,b)in pairs.items():
                expected=(1,32)if field=='aligned_reference'else(1089,32)
                if z[a].shape!=expected or z[b].shape!=expected or not np.isfinite(z[a]).all()or not np.isfinite(z[b]).all():
                    raise ValueError('Malformed saved features')
                actual=independent_stats(z[a],z[b])
                if any(abs(actual[k]-row[field][k])>1e-12 for k in actual):raise ValueError('Saved feature arithmetic differs')
                if field.startswith('aligned_'):passed &= actual['max_abs']<=protocol['aligned_max_abs_tolerance']
    if bool(passed)!=summary['aligned_controls_passed']:raise ValueError('Control verdict differs')
    aggregate={name:dict(points=8,
        aligned_max_abs=max(max(r['aligned_reference']['max_abs'],r['aligned_query']['max_abs'])for r in summary['rows']if r['fixture']==name),
        fractional_query_mean_l2=float(np.mean([r['fractional_query']['mean_l2']for r in summary['rows']if r['fixture']==name])))for name in names}
    result=dict(passed=True,aligned_controls_passed=bool(passed),replayed_points=32,
        per_fixture=aggregate,neural_forward_repeated=False,optimizer_updates=0,model_promoted=False,
        summary_sha256=sha(root/'summary.json'),source_sha256=sha(__file__),
        scope='Independent saved-feature arithmetic for one training terrain/four image conditions. Feature differences are not pose accuracy and do not establish why earlier learned models failed.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/native-crop-control-v57')
    p.add_argument('--output',default='reports/native_crop_audit_v57.json');a=p.parse_args();audit(a.run,a.output)
