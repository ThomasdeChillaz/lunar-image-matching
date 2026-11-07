"""Acquire and audit the actual public LunarGeo sample before any model fitting."""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import cv2
import numpy as np

from dataset import sha256
from lunar_geometry_v2 import camera_check,world_points,footprint_components
from remote_archive import HTTPRangeReader

RECORD='https://zenodo.org/api/records/18415119'
MODES=('nadir','oblique','dynamicgood')


def write_new(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def fetch(root='data/lunargeo',modes=MODES):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(RECORD,timeout=45) as response:record=json.load(response)
    record_path=root/'zenodo_record.json'
    if not record_path.exists():record_path.write_text(json.dumps(record,indent=2)+'\n')
    source=next(f for f in record['files'] if f['key']=='StereoGeo.zip')
    cache_name='.ranges' if len(modes)!=1 or modes[0]=='nadir' else '.ranges-'+modes[0]
    reader=HTTPRangeReader(source['links']['self'],source['size'],block_size=16*1024*1024,cache_dir=root/cache_name)
    with zipfile.ZipFile(reader) as archive:
        for mode in modes:
            if mode not in MODES:raise ValueError('Unknown source trajectory')
            path=root/(mode+'.zip');provenance=path.with_suffix('.provenance.json')
            if path.exists() and provenance.exists():
                if sha256(path)!=json.loads(provenance.read_text())['sha256']:raise ValueError('Source archive changed')
                continue
            info=archive.getinfo('StereoGeo/SouthPole/'+mode+'.zip')
            digest=hashlib.sha256();size=0
            with archive.open(info) as incoming,path.with_suffix('.zip.part').open('wb') as outgoing:
                while True:
                    data=incoming.read(1024*1024)
                    if not data:break
                    outgoing.write(data);digest.update(data);size+=len(data)
            if size!=info.file_size:raise ValueError('Incomplete inner archive')
            path.with_suffix('.zip.part').replace(path)
            write_new(provenance,dict(source_record=record['doi'],source_archive=source['links']['self'],
                member=info.filename,crc32=info.CRC,bytes=size,sha256=digest.hexdigest()))


def read_array(archive,name):
    return np.load(io.BytesIO(archive.read(name)),allow_pickle=False)


def decode_depth(archive,name):
    depth=cv2.imdecode(np.frombuffer(archive.read(name),np.uint8),cv2.IMREAD_UNCHANGED)
    if depth is None:raise ValueError('OpenEXR decoding unavailable')
    if depth.ndim==3:
        if not np.allclose(depth,depth[:,:,:1],equal_nan=True):raise ValueError('Depth EXR channels disagree')
        depth=depth[:,:,0]
    if depth.ndim!=2:raise ValueError('Single-channel ray depth required')
    return depth


def source_triplets(archive,include_diagnostics=False):
    members=set(archive.namelist());indexed={}
    for name in sorted(members):
        match=re.fullmatch(r'(.*)/im_(\d+)\.npz',name)
        if match:
            folder,index=match[1],int(match[2])
            indexed[(folder,index)]=name
    grouped={}
    for (folder,index),name in sorted(indexed.items()):
        if index%2:continue
        names=[name,indexed.get((folder,index+1))]
        if names[1] is None:raise ValueError('Missing second camera of source stereo pair')
        signature=hashlib.sha256()
        for camera in names:
            with read_array(archive,camera) as values:
                k,p=camera_check(values['intrinsics'],values['cam2world'])
                signature.update(k.tobytes());signature.update(p.tobytes())
        grouped.setdefault((folder,signature.hexdigest()),[]).append((index,names))
    result=[];excluded=[]
    for (folder,signature),pairs in sorted(grouped.items()):
        if len(pairs)!=3:
            excluded.append(dict(folder=folder,geometry_sha256=signature,stereo_configurations=len(pairs),
                reason='Expected three lighting observations of exactly identical camera geometry'))
            continue
        pairs.sort();number=pairs[0][0]//2;frames=[]
        for name in [name for _,names in pairs for name in names]:
            stem=name[:-4]
            jpg=next((stem+ext for ext in ('.jpg','.png','.jpeg') if stem+ext in members),None)
            exr=next((stem+ext for ext in ('.exr','_depth.exr','.depth.exr') if stem+ext in members),None)
            if jpg is None or exr is None:raise ValueError(f'Missing image or ray depth for {stem}')
            frames.append(dict(camera=name,image=jpg,depth=exr))
        result.append((folder,number,frames))
    diagnostics=dict(camera_metadata_files=len(indexed),distinct_stereo_geometries=len(grouped),
        complete_three_light_geometries=len(result),excluded_geometries=excluded,
        grouping='Actual intrinsics and camera-to-world matrices, not six-consecutive-filename assumptions')
    return (result,diagnostics) if include_diagnostics else result


def inspect(root='data/lunargeo',output='reports/lunargeo_inventory_v2.json'):
    root=Path(root);rows=[];archives={};source_diagnostics={}
    for mode in MODES:
        path=root/(mode+'.zip')
        if not path.exists():raise FileNotFoundError(path)
        provenance=json.loads(path.with_suffix('.provenance.json').read_text())
        if sha256(path)!=provenance['sha256']:raise ValueError('Original archive SHA256 mismatch')
        archives[mode]=provenance
        with zipfile.ZipFile(path) as archive:
            candidates=[]
            triplets,source_diagnostics[mode]=source_triplets(archive,True)
            for folder,number,frames in triplets:
                with read_array(archive,frames[0]['camera']) as c:
                    center=np.asarray(c['cam2world'],float)[:3,3]
                candidates.append((float(np.linalg.norm(center)),folder,number,frames,center))
            # Resource-bounded, metadata-only sample. Low camera radii favor
            # smaller footprints; distance selection is followed by exact union
            # of full-depth surface footprints, never assumed independence.
            candidates.sort(key=lambda r:(r[0],hashlib.sha256(f'{mode}:{r[1]}:{r[2]}'.encode()).hexdigest()))
            selected=[]
            for entry in candidates:
                if all(np.linalg.norm(entry[4]-other[4])>=2500. for other in selected):selected.append(entry)
                if len(selected)==64:break
            print(mode,'released geometries',len(triplets),'metadata-selected',len(selected),flush=True)
            for _,folder,number,frames,_ in selected:
                cameras=[]
                for frame in frames:
                    with read_array(archive,frame['camera']) as c:
                        k,pose=camera_check(c['intrinsics'],c['cam2world'])
                        cameras.append((k,pose))
                for i in range(2,6):
                    if not all(np.array_equal(a,b) for a,b in zip(cameras[i],cameras[i%2])):
                        raise ValueError('Advertised repeated illumination does not share exact camera geometry')
                # Geometry-only inventory: no RGB images or matcher outputs read.
                bounds=[];valid_count=0
                for i in (0,1):
                    depth=decode_depth(archive,frames[i]['depth'])
                    yy,xx=np.mgrid[0:depth.shape[0],0:depth.shape[1]]
                    r=depth[yy,xx].ravel();p=np.column_stack([xx.ravel(),yy.ravel()])
                    valid=np.isfinite(r)&(r>0);valid_count+=int(valid.sum())
                    if valid.any():bounds.append(world_points(p[valid],r[valid],*cameras[i]))
                if not bounds:
                    rows.append(dict(pair_group=f'{mode}/{folder}/{number}',applicable=False,reason='no_finite_surface_depth'))
                    continue
                world=np.concatenate(bounds)
                poses=[p.tolist() for _,p in cameras[:2]]
                rows.append(dict(pair_group=f'{mode}/{folder}/{number}',mode=mode,folder=folder,number=number,
                    frames=frames,world_bounds=[world.min(0).tolist(),world.max(0).tolist()],
                    source_groups=[hashlib.sha256(np.asarray(p,dtype=np.float64).tobytes()).hexdigest() for p in poses],
                    cameras=[dict(intrinsics=k.tolist(),cam2world=p.tolist()) for k,p in cameras[:2]],
                    source_size=list(depth.shape),applicable=True,finite_geometry_samples=valid_count,
                    sun_angles_available=False))
    valid=[r for r in rows if r['applicable']]
    components=footprint_components(valid,buffer_m=200.)
    result=dict(schema_version=2,dataset='MoonAnything LunarGeo public SouthPole sample',
        source_doi='10.5281/zenodo.18415119',archives=archives,records=rows,components=components,
        source_diagnostics=source_diagnostics,
        component_count=len(components),source_sha256=sha256(__file__),
        geometry_source_sha256=sha256('lunar_geometry_v2.py'),
        inventory_utc=datetime.now(timezone.utc).isoformat(),
        geometry_convention='NPZ camera-to-world; EXR distance along unit camera ray; optical +Z',
        grouping='Union of world XYZ bounds from every finite surface pixel plus200m and identical source poses across all modes.',
        sampling='Up to64 geometries per mode, increasing camera radius then SHA tie-break; camera centers at least2500m apart within mode. No pixel appearance or matching outcome selection.',
        limitation='A common terrain model/rendering pipeline can correlate spatial components. No real navigation reference or statistical independence claim.',
        training_started=False,final_test_inference_started=False)
    write_new(output,result)
    print(json.dumps(dict(groups=len(valid),components=len(components),inapplicable=len(rows)-len(valid))),flush=True)
    return result


def assign_components(inventory):
    groups=inventory['components']
    if len(groups)<28:
        raise ValueError(f'Public sample has only {len(groups)} connected components; need28 for train/selection/calibration/test. Record data shortfall before training.')
    lookup={r['pair_group']:r for r in inventory['records'] if r['applicable']}
    digest=lambda s:hashlib.sha256(('compact-lunar-v2:'+s).encode()).hexdigest()
    sectors={0:[],1:[],2:[]}
    for i,group in enumerate(groups):
        center=np.mean([np.mean(lookup[n]['world_bounds'],axis=0) for n in group],axis=0)
        sector=int(((np.degrees(np.arctan2(center[1],center[0]))+360)%360)//120)
        sectors[sector].append(i)
    if min(map(len,sectors.values()))<4:raise ValueError('Need four final-test components in each of three polar sectors')
    selected={}
    for sector,indices in sectors.items():
        indices.sort(key=lambda i:digest(groups[i][0]))
        for i in indices[:4]:selected[i]='test'
    remaining=sorted([i for i in range(len(groups)) if i not in selected],key=lambda i:digest(groups[i][0]))
    for i in remaining[:4]:selected[i]='model_selection'
    for i in remaining[4:8]:selected[i]='calibration'
    for i in remaining[8:]:selected[i]='train'
    result={}
    for i,group in enumerate(groups):
        for name in group:result[name]=(selected[i],f'polar-component-{i:04d}')
    return result


def prepare(inventory_path,output='reports/compact_dataset_v2.json',root='data/lunargeo'):
    inventory=json.loads(Path(inventory_path).read_text())
    assignment=assign_components(inventory) # Fail before loading RGB if inadequate.
    rows=[];root=Path(root);cache_root=Path('data/compact-v2');cache_root.mkdir(parents=True,exist_ok=True)
    for mode in MODES:
        with zipfile.ZipFile(root/(mode+'.zip')) as archive:
            candidates=[r for r in inventory['records'] if r.get('mode')==mode and r['applicable']]
            # Bound compute with at most two metadata-hash-selected geometries per component.
            selected=[]
            for component in sorted({assignment[r['pair_group']][1] for r in candidates}):
                eligible=[r for r in candidates if assignment[r['pair_group']][1]==component]
                eligible.sort(key=lambda r:hashlib.sha256(r['pair_group'].encode()).hexdigest())
                selected.extend(eligible[:2])
            for record in selected:
                split,component=assignment[record['pair_group']]
                images=[];depths=[];source_hashes=[]
                for frame in record['frames']:
                    content=archive.read(frame['image'])
                    im=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_GRAYSCALE)
                    if im is None or list(im.shape)!=record['source_size']:raise ValueError('Camera/depth/RGB resolution mismatch')
                    images.append(cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA))
                    depth=decode_depth(archive,frame['depth']);depths.append(depth)
                    source_hashes.append(dict(image=hashlib.sha256(content).hexdigest(),
                        camera=hashlib.sha256(archive.read(frame['camera'])).hexdigest(),
                        depth=hashlib.sha256(archive.read(frame['depth'])).hexdigest()))
                masks=[cv2.resize((np.isfinite(d)&(d>0)).astype(np.float32),(128,128),interpolation=cv2.INTER_AREA)>=.99999 for d in depths]
                name=hashlib.sha256(record['pair_group'].encode()).hexdigest()[:16]
                cache=cache_root/(name+'.npz')
                if cache.exists():raise FileExistsError(cache)
                np.savez_compressed(cache,images=np.stack(images)[::2],masks=np.stack(masks)[::2],
                    stereo_images=np.stack(images),stereo_masks=np.stack(masks),depth_a=depths[0],depth_b=depths[1])
                rows.append(dict(tile_id=record['pair_group'],split=split,component=component,mode=mode,
                    cache=str(cache),cache_sha256=sha256(cache),world_bounds=record['world_bounds'],
                    cameras=record['cameras'],source_size=record['source_size'],source_hashes=source_hashes,
                    frames=record['frames'],sun_angles_available=False))
    result=dict(schema_version=2,dataset=inventory['dataset'],inventory_sha256=sha256(inventory_path),
        native_size=128,tiles=rows,source_sha256=sha256(__file__),
        splits=dict(Counter(r['split'] for r in rows)),
        components_by_split={s:len({r['component'] for r in rows if r['split']==s}) for s in ('train','model_selection','calibration','test')},
        final_test_sectors=3,selection='Metadata hashes only; up to2 geometries per component per trajectory mode; all source lighting variants stay together',
        warning='Rendered data, not acquired lunar pixels. Sun-angle fields absent from inspected NPZ files; use lighting-pair IDs, never invented angles.')
    write_new(output,result)
    print(json.dumps(result['splits']),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['fetch','inspect','prepare'])
    p.add_argument('--root',default='data/lunargeo');p.add_argument('--output');p.add_argument('--inventory',default='reports/lunargeo_inventory_v2.json')
    a=p.parse_args()
    if a.action=='fetch':fetch(a.root)
    elif a.action=='inspect':inspect(a.root,a.output or a.inventory)
    else:prepare(a.inventory,a.output or 'reports/compact_dataset_v2.json',a.root)
