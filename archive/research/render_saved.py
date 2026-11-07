"""Regenerate presentation artifacts from immutable measurements; never rerun matching."""
import argparse,json
from pathlib import Path
import numpy as np
from dataset import load_pair,sha256
from experiment_report import render_results


def render(path,root='.'):
    path=Path(path);rows=json.loads((path/'results.json').read_text());summary=json.loads((path/'summary.json').read_text());config=json.loads((path/'frozen_config.json').read_text());manifest=json.loads((path/'frozen_manifest.json').read_text());meta=json.loads((path/'environment.json').read_text())
    assets={a['asset_id']:a for a in manifest['assets']};pairs={p['pair_id']:p for p in manifest['pairs']};gallery=[]
    for row in rows:
        if row['dataset_kind']!='real_repeat_observation' or row['preprocessing']!='none':continue
        a,b,*_=load_pair(pairs[row['pair_id']],assets,root,config['image_size'])
        gallery.append(dict(row=row,reference=a,query=b,matrix=None if row['transform'] is None else np.array(row['transform']),reference_points=None,query_points=None))
    meta['source_sha256']['experiment_report.py']=sha256(Path(__file__).with_name('experiment_report.py'))
    meta['report_rerendered_from_saved_measurements']=True
    meta['report_note']='Presentation regenerated; matching measurements and frozen experiment inputs unchanged.'
    render_results(rows,summary,gallery,config,meta,path)
    (path/'environment.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--root',default='.');a=p.parse_args();render(a.directory,a.root)
