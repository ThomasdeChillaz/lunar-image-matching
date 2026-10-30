"""Replay saved3D correspondences with independent unprojection/projection algebra."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new


def audit(run,output):
    run=Path(run);frozen=json.loads((run/'frozen_inputs.json').read_text())
    complete=json.loads((run/'COMPLETE.json').read_text())
    if sha256(run/'results.json')!=complete['results_sha256'] or sha256(run/'frozen_inputs.json')!=complete['frozen_inputs_sha256']:
        raise ValueError('Completed3D outcomes changed')
    manifest=json.loads(Path(frozen['manifest_path']).read_text())
    if sha256(frozen['manifest_path'])!=frozen['manifest_sha256']:raise ValueError('3D dataset changed')
    tiles={t['tile_id']:t for t in manifest['tiles']};rows=json.loads((run/'results.json').read_text());cache={}
    if [r['pair_id'] for r in rows]!=[r['pair_id'] for r in frozen['cases']]:raise ValueError('3D case coverage/order differs')
    for row,case in zip(rows,frozen['cases']):
        if any(row.get(k)!=v for k,v in case.items()):raise ValueError('3D saved case metadata differs')
        tile=tiles[row['tile_id']]
        if tile['tile_id'] not in cache:
            if sha256(tile['cache'])!=tile['cache_sha256']:raise ValueError('Depth/image cache changed')
            with np.load(tile['cache'],allow_pickle=False) as z:cache[tile['tile_id']]=(z['depth_a'].copy(),z['depth_b'].copy())
        da,db=cache[tile['tile_id']]
        a=np.array(row['reference_points'],float).reshape(-1,2)
        b=np.array(row['query_points'],float).reshape(-1,2)
        ka,pose_a=[np.array(tile['cameras'][0][name],float) for name in ('intrinsics','cam2world')]
        kb,pose_b=[np.array(tile['cameras'][1][name],float) for name in ('intrinsics','cam2world')]
        scale=np.array(tile['source_size'][::-1])/128
        native=(a+.5)*scale-.5
        index=np.rint(native).astype(int);safe=index.clip(0,np.array(da.shape[::-1])-1)
        valid=((index>=0)&(index<np.array(da.shape[::-1]))).all(1)
        ranges=da[safe[:,1],safe[:,0]];valid &= np.isfinite(ranges)&(ranges>0)
        rays=np.linalg.solve(ka,np.vstack([native.T,np.ones(len(native))])).T
        rays/=np.sqrt(np.square(rays).sum(1))[:,None]
        camera=rays*np.where(valid,ranges,1)[:,None]
        world=(pose_a@np.column_stack([camera,np.ones(len(camera))]).T).T[:,:3]
        # The scored camera convention is a rigid transform: its inverse uses
        # R-transpose. A generic4x4 inverse would reinterpret the author's
        # rounded rotation entries as a slightly non-rigid affine camera.
        world_to_query=np.eye(4);world_to_query[:3,:3]=pose_b[:3,:3].T
        world_to_query[:3,3]=-pose_b[:3,:3].T@pose_b[:3,3]
        query=(world_to_query@np.column_stack([world,np.ones(len(world))]).T).T[:,:3]
        homogeneous=(kb@query.T).T
        with np.errstate(divide='ignore',invalid='ignore'):expected=homogeneous[:,:2]/homogeneous[:,2,None]
        distance=np.sqrt(np.square(query).sum(1))
        valid &= query[:,2]>0
        valid &= np.isfinite(expected).all(1)&((expected>=0)&(expected<np.array(db.shape[::-1])-1)).all(1)
        if len(a):
            sample=cv2.remap(np.where(np.isfinite(db),db,0).astype(np.float32),
                expected[:,0,None].astype(np.float32),expected[:,1,None].astype(np.float32),cv2.INTER_LINEAR).ravel()
            support=cv2.remap((np.isfinite(db)&(db>0)).astype(np.float32),
                expected[:,0,None].astype(np.float32),expected[:,1,None].astype(np.float32),cv2.INTER_LINEAR).ravel()
            valid &= (support>=.9999)&(np.abs(sample-distance)<=.05+.01*distance)
        if valid.tolist()!=row['scorable_mask']:raise ValueError('Independent visibility replay differs')
        expected=(expected[valid]+.5)/scale-.5
        saved=np.array(row['reference_target_points'],float).reshape(-1,2)
        if not np.allclose(expected,saved,atol=1e-5):raise ValueError('Independent camera projection differs')
        errors=np.sqrt(np.square(b[valid]-expected).sum(1))
        if row['returned_matches']!=len(a) or row['reference_scorable_matches']!=int(valid.sum()):
            raise ValueError('3D correspondence denominators differ')
        if int((errors<=3).sum())!=row['correct_correspondences']:raise ValueError('3D correspondence scores differ')
        median=None if not len(errors) else float(np.median(errors))
        if (median is None)!=(row['median_correspondence_error_px'] is None) or (median is not None and not np.isclose(median,row['median_correspondence_error_px'],atol=1e-6)):
            raise ValueError('3D median correspondence error differs')
        if 'median_correspondence_error_native_px' in row and median is not None:
            if not np.isclose(median*scale[0],row['median_correspondence_error_native_px']):
                raise ValueError('3D native source pixel conversion differs')
        if row.get('estimated_cam2world') is not None:
            estimate=np.array(row['estimated_cam2world'],float)
            angle=np.degrees(np.arccos(np.clip((np.trace(estimate[:3,:3].T@pose_b[:3,:3])-1)/2,-1,1)))
            offset=np.linalg.norm(estimate[:3,3]-pose_b[:3,3])
            if not np.allclose([angle,offset],[row['pose_error']['rotation_deg'],row['pose_error']['translation_m']],atol=1e-6):
                raise ValueError('Pose error replay differs')
            finite=da[np.isfinite(da)&(da>0)];reference_range=float(np.median(finite))
            if not np.isclose(reference_range,row['reference_median_range_m']):raise ValueError('Pose reference range differs')
            correct=bool(angle<=2 and offset<=.01*reference_range)
            if correct!=row['pose_within_secondary_threshold']:raise ValueError('Secondary pose threshold differs')
    result=dict(passed=True,rows=len(rows),independent_algebra=True,raw_matches_before_planar_fitting=True,
        results_sha256=complete['results_sha256'])
    write_new(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(audit(a.run,a.output),indent=2))
