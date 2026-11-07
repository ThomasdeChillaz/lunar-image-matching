"""Replay benchmark.py's CSV schema, separate from real-data JSON artifacts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dataset import sha256
from geometry import endpoint_error,evaluation_points
from lunargeo_dataset import write_new
from synthetic import texture,make_pair


def audit(root,output):
    root=Path(root);config=json.loads((root/'config.json').read_text());env=json.loads((root/'environment.json').read_text())
    for p,d in env['source_sha256'].items():
        if sha256(p)!=d:raise ValueError('Smoke source changed')
    rows=list(csv.DictReader((root/'results.csv').open()));seen=set();groups={}
    for r in rows:
        key=(int(r['seed']),r['case'],r['method'])
        if key in seen or r['dataset_kind']!='procedural':raise ValueError('Duplicate or nonprocedural smoke row')
        seen.add(key);j=next(i for i,c in enumerate(config['cases']) if c['name']==r['case']);seed=key[0]
        _,truth=make_pair(texture(seed,config['size']),config['cases'][j],seed*1000+j)
        transform=None if not r['transform'] else np.asarray(json.loads(r['transform']))
        error=endpoint_error(transform,truth,evaluation_points((config['size'],config['size']),truth))
        saved=None if not r['error_px'] else float(r['error_px'])
        if (error is None)!=(saved is None) or (error is not None and not np.isclose(error,saved,rtol=0,atol=1e-10)):
            raise ValueError('Controlled transform metric differs')
        success=error is not None and error<=config['success_threshold_px']
        if success!=(r['success']=='True'):raise ValueError('Smoke success criterion differs')
        groups.setdefault((r['method'],r['case']),[]).append((error,success,float(r['runtime_ms'])))
    expected={(config['seed']+i,c['name'],m) for i in range(config['seeds']) for c in config['cases'] for m in ('ncc','ncc+clahe','sift','sift+clahe')}
    if seen!=expected:raise ValueError('Missing smoke cases')
    summaries=[]
    for (method,case),part in sorted(groups.items()):
        errors=[e for e,_,_ in part if e is not None]
        summaries.append(dict(method=method,case=case,pairs=len(part),success_rate=sum(s for _,s,_ in part)/len(part),
            failed_registrations=sum(e is None for e,_,_ in part),median_error_registered_px=float(np.median(errors)) if errors else None,
            median_runtime_ms=float(np.median([t for _,_,t in part]))))
    if summaries!=json.loads((root/'summary.json').read_text()):raise ValueError('Saved smoke aggregate differs')
    result=dict(passed=True,scope='Procedural software controls, not lunar navigation accuracy.',rows=len(rows),summaries=len(summaries),
        results_sha256=sha256(root/'results.csv'),summary_sha256=sha256(root/'summary.json'),source_sha256=sha256(__file__))
    write_new(output,result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.run,a.output)
