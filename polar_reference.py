"""External laser-reference diagnostics on real NASA POLAR laboratory images.

Laser points and published calibration supply the reference, never another image
matcher. Results are nominal calibration agreement: calibration covariance and
cross-light camera stability are not published as pixel-accuracy bounds. This
does not verify lunar orbital navigation accuracy.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from matchers import sift
from reserve_evaluation import infer,validate_freeze


def calibration(path):
    result={}
    for line in Path(path).read_text().splitlines():
        key,values=line.split(':',1)
        result[key]=np.fromstring(values,sep=' ')
    for key in ('CAMERA_MATRIX_LEFT','CAMERA_MATRIX_RIGHT','ROTATION_MATRIX','CAM_LIDAR_ROTATION'):
        result[key]=result[key].reshape(3,3)
    for key in ('ROTATION_MATRIX','CAM_LIDAR_ROTATION'):
        if not np.allclose(result[key].T@result[key],np.eye(3),atol=1e-3):
            raise ValueError('Published rotation is not orthogonal')
    return result


def camera_points(laser,cal,camera):
    # Published CAM_LIDAR is camera -> LIDAR. Invert it for world -> camera.
    left=(laser-cal['CAM_LIDAR_TRANSLATION'])@cal['CAM_LIDAR_ROTATION']
    if camera=='L':
        return left
    if camera!='R':
        raise ValueError('Unknown camera')
    return left@cal['ROTATION_MATRIX'].T+cal['TRANSLATION_VECTOR']


def project(laser,cal,camera):
    xyz=camera_points(laser,cal,camera)
    suffix='LEFT' if camera=='L' else 'RIGHT'
    uv,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),cal[f'CAMERA_MATRIX_{suffix}'],cal[f'DISTORTION_COEFFICIENTS_{suffix}'])
    uv=uv.reshape(-1,2)
    width,height=cal['IMAGE_WIDTH_HEIGHT'].astype(int)
    size=min(width,height)
    offset=np.array([(width-size)//2,(height-size)//2])
    native=(uv-offset+.5)*(128/size)-.5
    return native,xyz[:,2]


def visible(points,depth):
    finite=np.isfinite(points).all(1)&np.isfinite(depth)&(depth>0)
    index=np.rint(np.where(np.isfinite(points),points,-1000)).astype(int)
    finite&=(index>=0).all(1)&(index<128).all(1)
    z=np.full(128*128,np.inf)
    flat=index[finite,1]*128+index[finite,0]
    np.minimum.at(z,flat,depth[finite])
    keep=np.zeros(len(points),bool)
    keep[finite]=depth[finite]<=z[flat]+.01
    return keep


def correspondences(laser,cal,reference_camera,query_camera):
    a,za=project(laser,cal,reference_camera)
    b,zb=project(laser,cal,query_camera)
    keep=visible(a,za)&visible(b,zb)
    a,b=a[keep],b[keep]
    if len(a)<100:
        raise ValueError('Published coordinate/calibration convention yields insufficient co-visible reference points; do not fit it to matcher predictions')
    bins={}
    for i,key in enumerate(np.floor(a*2).astype(int)):
        bins.setdefault(tuple(key),[]).append(i)
    return a,b,bins


def score_points(reference_points,query_points,truth_a,truth_b,bins):
    if reference_points is None:
        return dict(returned_inlier_matches=0,laser_scorable_matches=0,unscorable_matches=0,nominal_within3px=0,median_nominal_error_px=None,errors_px=[])
    errors=[]
    for p,q in zip(reference_points,query_points):
        key=np.floor(p*2).astype(int)
        candidates=[i for dy in (-1,0,1) for dx in (-1,0,1) for i in bins.get((key[0]+dx,key[1]+dy),[])]
        if not candidates:
            continue
        # Select the external reference by its REFERENCE pixel only. The query
        # prediction cannot choose a favorable ground-truth correspondence.
        distances=np.linalg.norm(truth_a[candidates]-p,axis=1)
        best=int(np.argmin(distances))
        if distances[best]>.35:
            continue
        errors.append(float(np.linalg.norm(q-truth_b[candidates[best]])))
    return dict(returned_inlier_matches=len(reference_points),laser_scorable_matches=len(errors),
        unscorable_matches=len(reference_points)-len(errors),nominal_within3px=sum(e<=3 for e in errors),
        median_nominal_error_px=float(np.median(errors)) if errors else None,errors_px=errors)


def sensor_image(path):
    values=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if values is None or values.ndim!=2 or values.dtype!=np.uint16 or values.shape!=(1216,1936):
        raise ValueError('Expected original NASA1936x1216 uint16 sensor image')
    values=values[:,360:1576]
    valid=values<=4095
    byte=np.rint(np.clip(values,0,4095)*255./4095).astype(np.uint8)
    small=cv2.resize(np.where(valid,byte,0),(128,128),interpolation=cv2.INTER_AREA)
    mask=cv2.resize(valid.astype(np.float32),(128,128),interpolation=cv2.INTER_AREA)>=1.-1e-6
    small[~mask]=0
    return small,mask


def freeze(root,terrain_protocol,output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    records=json.loads((Path(root)/'provenance.json').read_text())
    terrain=validate_freeze(terrain_protocol)
    files={}
    for record in records:
        for member in record['members']:
            path=Path(root)/record['terrain']/member['local_name']
            if sha256(path)!=member['sha256']:
                raise ValueError('NASA POLAR source changed')
            files[str(path)]=member['sha256']
    cases=[]
    for record in records:
        photos={}
        for member in record['members']:
            name=member['source_member']
            if name.endswith('.png'):
                camera='L' if '/CamL_' in name else 'R'
                sun=30 if '/Sun_30/' in name else 180
                photos[(camera,sun)]=str(Path(root)/record['terrain']/member['local_name'])
        pairs=[('L',30,'L',180),('R',30,'R',180),('L',30,'R',30),('L',180,'R',180),('L',30,'R',180),('L',180,'R',30)]
        for ca,sa,cb,sb in pairs:
            cases.append(dict(pair_id=f'{record["terrain"]}_{ca}{sa}_{cb}{sb}',terrain=record['terrain'],
                reference=photos[(ca,sa)],query=photos[(cb,sb)],reference_camera=ca,query_camera=cb,
                reference_sun_deg=sa,query_sun_deg=sb,
                kind='fixed_camera_relighting' if ca==cb else 'stereo_laser_reference'))
    sources=['polar_reference.py','reserve_evaluation.py','contrastive_terrain.py','matchers.py','research_quality_gate.py','structural_refinement.py','confidence.py']
    doc=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),root=str(root),cases=cases,provenance=records,
        terrain_protocol_path=str(terrain_protocol),terrain_protocol_sha256=sha256(terrain_protocol),
        source_files_sha256=files,source_sha256={p:sha256(p) for p in sources},policy=terrain['policy'],training=terrain['training'],
        laser_row_stride=8,reference_assignment_radius_px=.35,visibility_depth_tolerance_m=.01,
        preprocessing='Center1216-square sensor crop, fixed12bit->8bit mapping and INTER_AREA128; same for every light and method. No contrast fitting.',
        reference='Published Faro XYZ and unique per-terrain stereo/CAM_LIDAR calibration; nominal fixed-camera pixel identity for relighting. Never matcher-derived reference points.',
        uncertainty='NASA describes sub2mm laser accuracy. Calibration covariance, cross-light rig drift and per-point pixel uncertainty are not supplied. Laser nearest-reference tolerance and visibility filtering are explicit. Report nominal reference agreement, not uncertainty-certified navigation accuracy.',
        scope='Four laboratory analogue terrains, real sensors and physical illumination; zero training or threshold selection. Not photographs taken on the Moon and not a reproduced dense-stereo benchmark.')
    output.write_text(json.dumps(doc,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),freeze_sha256=sha256(output))),flush=True)


def run(protocol,output):
    import torch
    from contrastive_terrain import ContrastiveTerrainMatcher
    doc=json.loads(Path(protocol).read_text())
    validate_freeze(doc['terrain_protocol_path'])
    for path,digest in {**doc['source_files_sha256'],**doc['source_sha256']}.items():
        if sha256(path)!=digest:
            raise ValueError('Frozen POLAR source changed')
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output/'frozen_protocol.json').write_bytes(Path(protocol).read_bytes())
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model=ContrastiveTerrainMatcher(doc['training'])
    reference_cache={}
    rows=[]
    for case in doc['cases']:
        key=(case['terrain'],case['reference_camera'],case['query_camera'])
        if key not in reference_cache:
            folder=Path(doc['root'])/case['terrain']
            laser=np.loadtxt(folder/'PosA.xyz',usecols=(2,3,4),dtype=np.float32)[::doc['laser_row_stride']]
            cal=calibration(folder/'stereo.calibration')
            reference_cache[key]=correspondences(laser,cal,case['reference_camera'],case['query_camera'])
        truth_a,truth_b,bins=reference_cache[key]
        a,ma=sensor_image(case['reference'])
        b,mb=sensor_image(case['query'])
        predicted=infer(a,b,ma,mb,model,doc['policy'])
        cv2.setRNGSeed(20261030)
        baseline=sift(a,b,ma,mb)
        for method,fit in [('terrain_raw',predicted['raw']),('terrain_deployed',predicted['deployed']),('sift',baseline)]:
            stats=score_points(fit.reference_points,fit.query_points,truth_a,truth_b,bins)
            nominal=None
            if case['kind']=='fixed_camera_relighting' and fit.matrix is not None:
                points=np.array([[x,y] for x in range(16,112,16) for y in range(16,112,16)])
                residual=np.linalg.norm(points@fit.matrix[:,:2].T+fit.matrix[:,2]-points,axis=1)
                nominal=float(np.median(residual))
            rows.append(dict(**case,method=method,estimated=fit.matrix is not None,
                transform=None if fit.matrix is None else fit.matrix.tolist(),
                reference_points=None if fit.reference_points is None else fit.reference_points.tolist(),
                query_points=None if fit.query_points is None else fit.query_points.tolist(),
                gate_accepted=predicted['accepted'] if method!='sift' else None,
                nominal_fixed_camera_error_px=nominal,independent_navigation_error=None,
                calibration_uncertainty_px=None,**stats))
        print(case['pair_id'],'completed',flush=True)
    (output/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    summary=dict(rows=len(rows),cases=len(doc['cases']),scope=doc['scope'],reference=doc['reference'],uncertainty=doc['uncertainty'])
    for method in ('terrain_raw','terrain_deployed','sift'):
        selected=[r for r in rows if r['method']==method]
        summary[method]=dict(cases=len(selected),fits=sum(r['estimated'] for r in selected),
            laser_scorable_inlier_matches=sum(r['laser_scorable_matches'] for r in selected),
            nominal_inlier_matches_within3px=sum(r['nominal_within3px'] for r in selected),
            fixed_camera_cases=sum(r['kind']=='fixed_camera_relighting' for r in selected),
            fixed_camera_nominal_success=sum(r['kind']=='fixed_camera_relighting' and r['nominal_fixed_camera_error_px'] is not None and r['nominal_fixed_camera_error_px']<=3 for r in selected))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'COMPLETE.json').write_text(json.dumps(dict(results_sha256=sha256(output/'results.json'),
        protocol_sha256=sha256(protocol),summary_sha256=sha256(output/'summary.json')),indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','run'])
    p.add_argument('--root',default='data/polar')
    p.add_argument('--terrain-protocol',default='reports/reserve_protocol_v1.json')
    p.add_argument('--protocol',required=True)
    p.add_argument('--output')
    args=p.parse_args()
    if args.action=='freeze':
        freeze(args.root,args.terrain_protocol,args.protocol)
    else:
        run(args.protocol,args.output)
