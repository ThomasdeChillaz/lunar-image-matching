"""Version corrected labels/footprints without changing any historical artifact."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dataset import sha256
from lunar_geometry_v2 import footprint_components
from lunar_geometry_v5 import world_points_z,correspondences_z
from lunargeo_dataset import write_new
from train_balanced_localization_v3 import load_bundle


def run(output,cache_root,audit_output):
    output,cache_root=Path(output),Path(cache_root)
    if output.exists() or cache_root.exists():raise FileExistsError('Preserve existing correction outputs')
    manifest,tiles,data=load_bundle('reports/localization_training_data_v3.json');records=[]
    for key,t in tiles.items():
        z=data[key];bounds=[]
        for name,camera in zip(('depth_a','depth_b'),t['cameras']):
            depth=z[name];y,x=np.mgrid[:depth.shape[0],:depth.shape[1]];d=depth.ravel()
            valid=np.isfinite(d)&(d>0);p=np.column_stack([x.ravel(),y.ravel()])[valid]
            world=world_points_z(p,d[valid],camera['intrinsics'],camera['cam2world'])
            bounds.extend([world.min(0),world.max(0)])
        bounds=np.asarray(bounds)
        records.append(dict(pair_group=key,split=t['split'],world_bounds=[bounds.min(0).tolist(),bounds.max(0).tolist()],
            source_groups=[hashlib.sha256(np.asarray(c['cam2world'],np.float64).tobytes()).hexdigest() for c in t['cameras']]))
    groups=footprint_components(records,buffer_m=200.);lookup={r['pair_group']:r for r in records}
    overlap=[g for g in groups if len({lookup[key]['split'] for key in g})>1]
    audit=dict(scope='Full-pixel axial-depth footprint audit on original18 train/eight selection geometries only.',
        source_sha256=sha256(__file__),geometry_sha256=sha256('lunar_geometry_v5.py'),
        original_manifest_sha256=sha256('reports/localization_training_data_v3.json'),components=groups,
        cross_split_components=overlap,train_selection_disjoint=not overlap,records=records,
        calibration_or_test_pixels_loaded=0,
        limitation='Spatial bounds plus200m and repeated source cameras are grouped; unknown trajectory/batch dependence remains. Not a new final test.')
    write_new(audit_output,audit)
    print('Corrected components',len(groups),'cross-split components',len(overlap),flush=True)
    if overlap:
        print('Corrected training preparation stopped at split gate. Preserve audit and repair partition before fitting.',flush=True)
        return audit
    cache_root.mkdir(parents=True);group_of={key:f'axial-v5-component-{i:04d}' for i,g in enumerate(groups) for key in g}
    new_tiles=[]
    for key,t in tiles.items():
        z=data[key];arrays={k:v.copy() for k,v in z.items()};a,b=t['cameras']
        for c in [c for c in manifest['cases'] if c['tile_id']==key]:
            k=c['key'];p=z[f'p{k}']
            _,target,_,visible=correspondences_z(z['depth_a'],z['depth_b'],a['intrinsics'],a['cam2world'],b['intrinsics'],b['cam2world'],points=p)
            truth=np.zeros_like(p);truth[visible]=target
            arrays[f'target{k}']=truth;arrays[f'visible{k}']=visible
        cache=cache_root/(hashlib.sha256(key.encode()).hexdigest()[:16]+'.npz');np.savez_compressed(cache,**arrays)
        new_tiles.append(dict(t,legacy_component=t['component'],component=group_of[key],
            world_bounds=lookup[key]['world_bounds'],native_cache=str(cache),native_cache_sha256=sha256(cache)))
    cases=[dict(c,legacy_component=c['component'],component=group_of[c['tile_id']]) for c in manifest['cases']]
    result=dict(manifest,schema_version=5,scope='Corrected axial-depth development labels and audited disjoint spatial groups. No final-test claim.',
        original_v3_manifest='reports/localization_training_data_v3.json',original_v3_manifest_sha256=sha256('reports/localization_training_data_v3.json'),
        geometry_convention='Optical-axis z depth; unchanged stored source intrinsics; native integer-centered xy; reference nearest-depth sampling.',
        source_sha256=sha256(__file__),geometry_source_sha256=sha256('lunar_geometry_v5.py'),
        split_audit=str(audit_output),split_audit_sha256=sha256(audit_output),train_selection_disjoint=True,
        tiles=new_tiles,cases=cases,train_components=len({t['component'] for t in new_tiles if t['split']=='train'}),
        selection_components=len({t['component'] for t in new_tiles if t['split']=='model_selection'}),
        geometric_labels='Reference axial-depth unprojection, camera projection and axial query-depth visibility; no matcher-derived GT.',
        visibility_tolerance='Preserved0.05m+1% depth for isolated convention correction; sensitivity remains a separate experiment.')
    write_new(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/localization_training_data_v5.json')
    p.add_argument('--cache-root',default='data/localization-v5-development');p.add_argument('--audit',default='reports/localization_v5_split_audit.json')
    a=p.parse_args();run(a.output,a.cache_root,a.audit)
