"""Cross-view depth closure tests: independent of matcher and fitted camera pose."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import sha256
from lunargeo_dataset import write_new
from train_balanced_localization_v3 import load_bundle


def closure(da,db,ka,ca,kb,cb,convention='ray',orientation='normal',stride=8):
    if orientation=='flip_y':da,db=np.flipud(da).copy(),np.flipud(db).copy()
    elif orientation=='transpose':da,db=da.T.copy(),db.T.copy()
    elif orientation!='normal':raise ValueError(orientation)
    y,x=np.mgrid[8:da.shape[0]-8:stride,8:da.shape[1]-8:stride]
    p=np.column_stack([x.ravel(),y.ravel(),np.ones(x.size)])
    depth=da[y.ravel(),x.ravel()];rays=p@np.linalg.inv(ka).T
    if convention=='ray':rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    elif convention!='z':raise ValueError(convention)
    world=(rays*np.where(np.isfinite(depth),depth,0)[:,None])@ca[:3,:3].T+ca[:3,3]
    local=(world-cb[:3,3])@cb[:3,:3];homogeneous=local@kb.T
    with np.errstate(divide='ignore',invalid='ignore'):q=homogeneous[:,:2]/homogeneous[:,2,None]
    inside=np.isfinite(depth)&(depth>0)&np.isfinite(q).all(1)&(local[:,2]>0)&((q>=0)&(q<np.array(db.shape[::-1])-1)).all(1)
    q=np.where(np.isfinite(q),q,-1).astype(np.float32)
    sampled=cv2.remap(np.where(np.isfinite(db),db,0).astype(np.float32),q[:,0,None],q[:,1,None],cv2.INTER_LINEAR).ravel()
    support=cv2.remap((np.isfinite(db)&(db>0)).astype(np.float32),q[:,0,None],q[:,1,None],cv2.INTER_LINEAR).ravel()
    inside&=support>=.9999
    predicted=np.linalg.norm(local,axis=1) if convention=='ray' else local[:,2]
    err=np.abs(predicted[inside]-sampled[inside]);relative=err/np.maximum(predicted[inside],1e-9)
    return dict(points=len(p),projected_inside=int(inside.sum()),median_absolute_m=float(np.median(err)) if len(err) else None,
        median_relative=float(np.median(relative)) if len(err) else None,p90_absolute_m=float(np.percentile(err,90)) if len(err) else None,
        fraction_below_relative_1e_4=float(np.mean(relative<=1e-4)) if len(err) else None,
        fraction_below_relative_1e_2=float(np.mean(relative<=1e-2)) if len(err) else None)


def run(output):
    manifest,tiles,data=load_bundle('reports/localization_training_data_v3.json');rows=[]
    for key,t in tiles.items():
        if t['split']!='model_selection':continue
        z=data[key];a,b=t['cameras'];ka,ca=np.array(a['intrinsics']),np.array(a['cam2world']);kb,cb=np.array(b['intrinsics']),np.array(b['cam2world'])
        for convention in ('ray','z'):
            for orientation in ('normal','flip_y','transpose'):
                rows.append(dict(tile_id=key,convention=convention,orientation=orientation,
                    **closure(z['depth_a'],z['depth_b'],ka,ca,kb,cb,convention,orientation)))
    result=dict(scope='Development cross-view depth closure, not matcher accuracy or independent orbital pose validation.',
        source_sha256=sha256(__file__),manifest_sha256=sha256('reports/localization_training_data_v3.json'),
        ground_truth_use='Released cameras/depth test their mutual consistency; no matcher or estimated pose is used.',
        caveat='Occlusions, interpolation and source/reference uncertainty remain. Lowest closure alone does not characterize image-registration accuracy.',rows=rows)
    write_new(output,result)
    for r in rows:
        if r['orientation']=='normal':print(r['tile_id'],r['convention'],'median_m',round(r['median_absolute_m'],4),'within1e-4',round(r['fraction_below_relative_1e_4'],3))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/localization_v4_depth_closure.json');run(p.parse_args().output)
