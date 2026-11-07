"""Verify external-reference bytes, projection math and complete POLAR scoring."""
import argparse
import json
from pathlib import Path

import numpy as np

from dataset import sha256
from polar_reference import calibration,correspondences,project,score_points


def analytic_projection(laser,cal,camera):
    left=(laser-cal['CAM_LIDAR_TRANSLATION'])@cal['CAM_LIDAR_ROTATION']
    xyz=left if camera=='L' else left@cal['ROTATION_MATRIX'].T+cal['TRANSLATION_VECTOR']
    x,y=xyz[:,0]/xyz[:,2],xyz[:,1]/xyz[:,2]
    suffix='LEFT' if camera=='L' else 'RIGHT'
    k1,k2,p1,p2=cal[f'DISTORTION_COEFFICIENTS_{suffix}']
    r2=x*x+y*y
    radial=1+k1*r2+k2*r2*r2
    xd=x*radial+2*p1*x*y+p2*(r2+2*x*x)
    yd=y*radial+p1*(r2+2*y*y)+2*p2*x*y
    K=cal[f'CAMERA_MATRIX_{suffix}']
    if K[0,1]!=0:
        raise ValueError('Unexpected calibration skew')
    uv=np.column_stack([K[0,0]*xd+K[0,2],K[1,1]*yd+K[1,2]])
    width,height=cal['IMAGE_WIDTH_HEIGHT'].astype(int)
    size=min(width,height)
    return (uv-[(width-size)//2,(height-size)//2]+.5)*(128/size)-.5


def verify(run):
    run=Path(run)
    doc=json.loads((run/'frozen_protocol.json').read_text())
    complete=json.loads((run/'COMPLETE.json').read_text())
    for name,key in [('results.json','results_sha256'),('summary.json','summary_sha256'),('frozen_protocol.json','protocol_sha256')]:
        if sha256(run/name)!=complete[key]:
            raise ValueError('Changed completed POLAR output')
    for name,digest in {**doc['source_files_sha256'],**doc['source_sha256']}.items():
        if sha256(name)!=digest:
            raise ValueError('Changed POLAR source')
    rows=json.loads((run/'results.json').read_text())
    cases={c['pair_id']:c for c in doc['cases']}
    expected={(p,m) for p in cases for m in ('terrain_raw','terrain_deployed','sift')}
    if len(rows)!=len(expected) or {(r['pair_id'],r['method']) for r in rows}!=expected:
        raise ValueError('Incomplete/duplicate POLAR predictions')
    reference={}
    by_terrain={}
    for record in doc['provenance']:
        folder=Path(doc['root'])/record['terrain']
        laser=np.loadtxt(folder/'PosA.xyz',usecols=(2,3,4),dtype=np.float32)[::doc['laser_row_stride']]
        cal=calibration(folder/'stereo.calibration')
        for camera in ('L','R'):
            uv,depth=project(laser,cal,camera)
            manual=analytic_projection(laser,cal,camera)
            keep=(depth>.1)&np.isfinite(uv).all(1)&(uv>=0).all(1)&(uv<128).all(1)
            if not np.allclose(uv[keep],manual[keep],atol=1e-8,rtol=1e-10):
                raise ValueError('Independent Brown-Conrady/pixel-center projection disagrees')
        for ca,cb in [('L','L'),('R','R'),('L','R')]:
            reference[(record['terrain'],ca,cb)]=correspondences(laser,cal,ca,cb)
        by_terrain[record['terrain']]=dict(sampled_external_laser_points=len(laser),
            camera_calibration_sha256=sha256(folder/'stereo.calibration'))
    for row in rows:
        case=cases[row['pair_id']]
        if any(row[k]!=v for k,v in case.items()) or row['independent_navigation_error'] is not None or row['calibration_uncertainty_px'] is not None:
            raise ValueError('Changed case or invented navigation accuracy/uncertainty')
        a,b,bins=reference[(case['terrain'],case['reference_camera'],case['query_camera'])]
        p=None if row['reference_points'] is None else np.asarray(row['reference_points'])
        q=None if row['query_points'] is None else np.asarray(row['query_points'])
        scored=score_points(p,q,a,b,bins)
        if any(row[k]!=v for k,v in scored.items()):
            raise ValueError('Laser-reference scoring replay differs')
    return dict(cases=len(cases),rows=len(rows),by_terrain=by_terrain,
        source_bytes_verified=True,independent_brown_conrady_projection_verified=True,
        reference_assignments_and_errors_replayed=True,matcher_outputs_excluded_from_reference_construction=True,
        unknown_calibration_uncertainty_preserved=True,results_sha256=complete['results_sha256'],
        warning=doc['uncertainty'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    path=Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    result=verify(args.run)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
