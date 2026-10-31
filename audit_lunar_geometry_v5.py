"""Audit source depth representation and camera pixel centers before refitting."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from audit_depth_conventions_v4 import closure
from dataset import sha256
from lunargeo_dataset import decode_depth,write_new
from train_balanced_localization_v3 import load_bundle


def run(output):
    manifest,tiles,data=load_bundle('reports/localization_training_data_v3.json');rows=[];source_checks=[];archives={}
    try:
        for key,t in tiles.items():
            z=data[key];mode=t['mode']
            if mode not in archives:archives[mode]=zipfile.ZipFile(Path('data/lunargeo')/(mode+'.zip'))
            for i,name in enumerate(('depth_a','depth_b')):
                raw=decode_depth(archives[mode],t['frames'][i]['depth'])
                if not np.array_equal(raw,z[name],equal_nan=True):raise ValueError('Processed depth differs from raw source EXR')
                source_checks.append(dict(tile_id=key,frame=i,raw_matches_cache=True,source_depth=t['frames'][i]['depth']))
            a,b=t['cameras'];ca,cb=np.array(a['cam2world']),np.array(b['cam2world'])
            for convention in ('ray','z'):
                for offset in (-.5,0.,.5):
                    ka,kb=[np.array(c['intrinsics']) for c in (a,b)]
                    for k in (ka,kb):k[:2,2]+=offset
                    rows.append(dict(tile_id=key,split=t['split'],convention=convention,principal_point_offset=offset,
                        **closure(z['depth_a'],z['depth_b'],ka,ca,kb,cb,convention,'normal')))
    finally:
        for archive in archives.values():archive.close()
    scores=[]
    for convention in ('ray','z'):
        for offset in (-.5,0.,.5):
            part=[r for r in rows if r['split']=='train' and r['convention']==convention and r['principal_point_offset']==offset]
            scores.append(dict(convention=convention,principal_point_offset=offset,
                training_median_of_median_absolute_m=float(np.median([r['median_absolute_m'] for r in part]))))
    winner=min(scores,key=lambda s:s['training_median_of_median_absolute_m'])
    result=dict(scope='Geometry-only source-convention audit. No matcher, fitted pose, calibration/test pixels or performance-driven convention selection.',
        source_sha256=sha256(__file__),closure_source_sha256=sha256('audit_depth_conventions_v4.py'),
        manifest_sha256=sha256('reports/localization_training_data_v3.json'),source_checks=source_checks,
        candidate_rule='Choose minimum median depth-closure residual on18 original training geometries; inspect8 selection geometries separately.',
        candidates=scores,selected=winner,rows=rows,
        interpretation='Optical-axis z matches the released EXRs more closely than ray range; published wording and legacy code must be reconciled explicitly.',
        caveat='Depth closure diagnoses internal geometry consistency. It does not independently characterize camera/image registration or real orbital accuracy.')
    write_new(output,result)
    print(json.dumps(dict(candidates=scores,selected=winner,raw_source_depth_arrays_verified=len(source_checks)),indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/localization_v5_geometry_audit.json');run(p.parse_args().output)
