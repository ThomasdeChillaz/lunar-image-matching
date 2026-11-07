"""Development-only six-light inputs and exact 3D training labels.

Preparation waits for the complete v35 geometry gate. Inference callers pass
only reference geometry; query geometry in this bundle is supervision/scoring.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from north_development_plan_v35 import development_plan,proposal_grid_native,quarter_turn_points
from north_render_plan_v31 import support_overlap


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def prepare(output,render_root,data_root,manifest_path):
    from PIL import Image
    plan,sites=development_plan();root,raw,dem=Path(output),Path(render_root),Path(data_root)
    manifest_path=Path(manifest_path)
    if root.exists()or manifest_path.exists():raise FileExistsError('Preserve previous bundle or failure')
    gate_path=Path('reports/north_development_audit_v37.json');gate=read(gate_path)
    if (not gate['passed']or not gate.get('solar_envelope_verified')
            or gate['eligible_geometry_pairs']!=72 or gate['oracle_pose_rate']<.95
            or gate['exact_supervision_points']!=243360 or len(gate['exact_supervision_records'])!=360):
        raise ValueError('Complete actual native/fractional/72geometry gate required')
    render_summary=read(raw/'summary.json');render_protocol=read(raw/'protocol.json')
    if render_protocol['data_summary_sha256']!=sha(dem/'summary.json'):raise ValueError('Acquired terrain summary changed')
    acquired=read(dem/'summary.json')
    if gate['renderer_summary_sha256']!=sha(raw/'summary.json'):raise ValueError('Audited renderer output changed')
    root.mkdir(parents=True);tiles=[];cases=[]
    protocol=dict(scope='New additional northern16train/eightselection study only; no reserve pixels.',
        source_sha256=sha(__file__),geometry_audit_sha256=sha(gate_path),
        renderer_summary_sha256=sha(raw/'summary.json'),six_lights=True,
        preprocessing='Pillow RGB-to-L conversion, uint8 native512; OpenCV area resize to128 for training.',
        query_geometry='Training/scoring only; never inference arguments.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for site,record in zip(sites,render_summary['records']):
        if site['component']!=record['component']:raise ValueError('Original component order differs')
        folder=raw/site['component'];geometries=[];images=[]
        for view in range(4):
            path=folder/f'view-{view}'/'geometry.npz'
            if sha(path)!=record['geometry_sha256'][f'view-{view}/geometry.npz']:raise ValueError('Audited geometry changed')
            with np.load(path,allow_pickle=False)as f:geometries.append({k:f[k]for k in ('axial_depth','cam2world','intrinsics','body_origin','enu_rows')})
            view_images=[]
            for light in range(6):
                path=folder/f'view-{view}'/f'light-{light}'/'image.png'
                expected=next(r['image_sha256']for r in record['images']if r['view']==view and r['light']==light)
                if sha(path)!=expected:raise ValueError('Audited source image changed')
                with Image.open(path)as im:image=np.asarray(im.convert('L')).copy()
                if image.shape!=(512,512)or image.dtype!=np.uint8:raise ValueError('Native grayscale shape differs')
                view_images.append(image)
            images.append(view_images)
        mesh_path=dem/site['component']/'mesh.npz'
        expected_mesh=next(r['mesh_sha256']for r in acquired['records']if r['component']==site['component'])
        if sha(mesh_path)!=expected_mesh:raise ValueError('Source mesh changed before global support conversion')
        with np.load(mesh_path,allow_pickle=False)as f:
            body=f['vertices'].astype(float)@f['enu_rows']+f['body_origin']
        bounds=[body.min(0).tolist(),body.max(0).tolist()]
        for query_view in (1,2,3):
            label_records={r['file']:r['sha256']for r in record['exact_supervision']}
            paths=[folder/f'supervision-forward-0-{query_view}.npz']+[
                folder/f'supervision-reverse-{query_view}-0-rotation-{r}.npz'for r in range(4)]
            labels=[]
            for p in paths:
                if sha(p)!=label_records[p.name]:raise ValueError('Exact supervision changed')
                with np.load(p,allow_pickle=False)as f:labels.append({k:f[k]for k in f.files})
            ref,query=geometries[0],geometries[query_view]
            tile_id=f'{site["component"]}/views-0-{query_view}'
            cache=root/(hashlib.sha256(tile_id.encode()).hexdigest()[:20]+'.npz')
            interleaved=np.stack([im for light in range(6)for im in (images[0][light],images[query_view][light])])
            np.savez_compressed(cache,images=interleaved,depth_a=ref['axial_depth'],depth_b=query['axial_depth'],
                forward_target=labels[0]['target_points'],forward_visible=labels[0]['visible'],
                reverse_target=np.stack([r['target_points']for r in labels[1:]]),
                reverse_visible=np.stack([r['visible']for r in labels[1:]]))
            tile=dict(tile_id=tile_id,component=site['component'],split='train'if site['split']=='training'else'model_selection',
                query_view=query_view,native_cache=str(cache),native_cache_sha256=sha(cache),world_bounds=bounds,
                support=dict(x=site['x'],y=site['y']),body_origin=ref['body_origin'].tolist(),enu_rows=ref['enu_rows'].tolist(),
                cameras=[dict(intrinsics=g['intrinsics'].tolist(),cam2world=g['cam2world'].tolist())for g in (ref,query)],
                exact_supervision_sha256={p.name:sha(p)for p in paths},lighting_count=6)
            tiles.append(tile)
            if tile['split']=='model_selection':
                for la,lb in plan['primary_lighting_pairs']:
                    cases.append(dict(pair_id=f'{tile_id}/lights-{la}-{lb}',tile_id=tile_id,component=tile['component'],
                        split='model_selection',light_a=la,light_b=lb,query_view=query_view,positive=True))
        print('PREPARED TRAINING COMPONENT',site['component'],flush=True)
    if len(tiles)!=72 or len(cases)!=96:raise ValueError('Complete development denominator differs')
    train={t['component']for t in tiles if t['split']=='train'};selection={t['component']for t in tiles if t['split']=='model_selection'}
    if len(train)!=16 or len(selection)!=8 or train&selection:raise ValueError('Train/selection grouping failed')
    for i,a in enumerate(sites):
        if any(support_overlap(a,b)for b in sites[i+1:]):raise ValueError('Full source supports overlap')
    negatives=[];index={t['tile_id']:t for t in tiles};ordered=sorted(selection)
    for case in cases:
        other=ordered[(ordered.index(case['component'])+1)%len(ordered)]
        other_tile=f'{other}/views-0-{case["query_view"]}'
        if other_tile not in index:raise ValueError('Missing fixed negative endpoint')
        negatives.append(dict(case,pair_id=case['pair_id']+'/nonoverlap-'+other,positive=False,
                              query_tile_id=other_tile,query_component=other))
    old=read('reports/localization_training_data_v5.json')
    result=dict(schema_version=38,scope=protocol['scope'],tiles=tiles,cases=cases,negative_cases=negatives,
        training_components=sorted(train),selection_components=ordered,train_selection_disjoint=True,
        calibration_or_test_pixels_loaded=0,excluded_pilot_pixels_loaded=0,lighting_count=6,
        train_geometries=48,selection_geometries=24,selection_positive_pairs=96,selection_negative_pairs=96,
        coarse_model=old['coarse_model'],coarse_checkpoint_sha256=old['coarse_checkpoint_sha256'],
        protocol_sha256=sha(root/'protocol.json'),source_sha256=sha(__file__),geometry_audit_sha256=sha(gate_path),
        renderer_summary_path=str(raw/'summary.json'),renderer_summary_sha256=sha(raw/'summary.json'),
        solar_contract_sha256=gate['solar_contract_sha256'],dated_solar_ephemeris_verified=False)
    manifest_path.write_text(json.dumps(result,indent=2)+'\n');return result


def load_bundle(path,train_only=False,selection_only=False):
    if train_only and selection_only:raise ValueError('Choose one data-loading split')
    m=read(path)
    if (m['schema_version']!=38 or m['lighting_count']!=6 or not m['train_selection_disjoint']
            or m['calibration_or_test_pixels_loaded']or m['excluded_pilot_pixels_loaded']):
        raise ValueError('Audited six-light development-only bundle required')
    if m['source_sha256']!=sha(__file__):raise ValueError('Bundle preparation/loading source changed')
    gate_path=Path('reports/north_development_audit_v37.json')
    if (sha(gate_path)!=m['geometry_audit_sha256']or not read(gate_path)['passed']
            or not read(gate_path).get('solar_envelope_verified')
            or m['solar_contract_sha256']!=sha('reports/north_solar_amendment_v37.json')):
        raise ValueError('Actual full-data geometry gate changed')
    _,sites=development_plan();expected={s['component']:s['split']for s in sites}
    expected_sites={s['component']:s for s in sites}
    expected_tiles={f'{s["component"]}/views-0-{v}'for s in sites for v in (1,2,3)}
    expected_cases=[]
    plan,_=development_plan()
    for site in sites:
        if site['split']!='model_selection':continue
        for view in (1,2,3):
            tile_id=f'{site["component"]}/views-0-{view}'
            for la,lb in plan['primary_lighting_pairs']:
                expected_cases.append(dict(pair_id=f'{tile_id}/lights-{la}-{lb}',tile_id=tile_id,
                    component=site['component'],split='model_selection',light_a=la,light_b=lb,query_view=view,positive=True))
    if m['cases']!=expected_cases:raise ValueError('Frozen complete positive case list changed')
    tiles={t['tile_id']:t for t in m['tiles']};data={}
    if set(tiles)!=expected_tiles:raise ValueError('Complete original geometry set required')
    if len(tiles)!=72 or len(m['cases'])!=96:raise ValueError('Complete source/case list required')
    for key,t in tiles.items():
        expected_split='train'if expected.get(t['component'])=='training'else'model_selection'
        if t['component']not in expected or t['split']!=expected_split:raise ValueError('Forbidden component/split')
        site=expected_sites[t['component']]
        if t['support']!={'x':site['x'],'y':site['y']}:raise ValueError('Global support coordinates changed')
        if train_only and t['split']!='train':continue
        if selection_only and t['split']!='model_selection':continue
        if sha(t['native_cache'])!=t['native_cache_sha256']:raise ValueError('Prepared native input changed')
        with np.load(t['native_cache'],allow_pickle=False)as f:data[key]={k:f[k].copy()for k in f.files}
        z=data[key]
        if z['images'].shape!=(12,512,512)or z['forward_target'].shape!=(676,2)or z['reverse_target'].shape!=(4,676,2):
            raise ValueError('Six-light image or fractional-label shape differs')
    if sha(Path(m['coarse_model'])/'selected.pt')!=m['coarse_checkpoint_sha256']:raise ValueError('Frozen starting checkpoint changed')
    if train_only:tiles={k:t for k,t in tiles.items()if t['split']=='train'}
    if selection_only:tiles={k:t for k,t in tiles.items()if t['split']=='model_selection'}
    return m,tiles,data


def prepare_training(tiles,data):
    prepared={}
    for key,t in tiles.items():
        if t['split']!='train':continue
        z=data[key];rotations={}
        for k in range(4):
            forward=(quarter_turn_points(z['forward_target'],k)+.5)/4-.5
            reverse=(z['reverse_target'][k]+.5)/4-.5
            rotations[k]=dict(forward=forward.astype(np.float32),reverse=reverse.astype(np.float32),
                forward_visible=z['forward_visible'].copy(),reverse_visible=z['reverse_visible'][k].copy())
        prepared[key]=dict(images=np.stack([cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA)for im in z['images']]),
                           rotations=rotations)
    return prepared


def sample_training(tiles,prepared,rng,negative_pair=False):
    from train_balanced_localization_v3 import appearance
    groups={}
    for t in tiles.values():
        if t['split']=='train':groups.setdefault(t['component'],[]).append(t['tile_id'])
    names=sorted(groups);component=names[int(rng.integers(len(names)))];ids=groups[component]
    key=ids[int(rng.integers(len(ids)))];other=key
    la,lb=rng.integers(0,6,2);rotation=int(rng.integers(4))
    if negative_pair:
        eligible=[g for g in names if g!=component and not support_overlap(tiles[key]['support'],tiles[groups[g][0]]['support'])]
        if not eligible:raise ValueError('Nonoverlapping training components required')
        query_component=eligible[int(rng.integers(len(eligible)))];query_ids=groups[query_component]
        other=query_ids[int(rng.integers(len(query_ids)))];n=len(proposal_grid_native())
        target=dict(forward=np.zeros((n,2),np.float32),reverse=np.zeros((n,2),np.float32),
                    forward_visible=np.zeros(n,bool),reverse_visible=np.zeros(n,bool))
    else:target=prepared[key]['rotations'][rotation]
    reference=prepared[key]['images'][2*la];query=np.rot90(prepared[other]['images'][2*lb+1],rotation).copy()
    images=np.stack([appearance(reference,rng),appearance(query,rng)])[:,None]
    target={k:v.copy()for k,v in target.items()}
    meta=dict(tile_a=key,tile_b=other,component_a=component,component_b=tiles[other]['component'],
              light_a=int(la),light_b=int(lb),query_quarter_turn=rotation,verified_nonoverlap=bool(negative_pair))
    return images,target,meta


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='data/north-render-v31/training-v38')
    p.add_argument('--render-root',default='runs/north-v37-development');p.add_argument('--data-root',default='data/north-render-v31/development-v35')
    p.add_argument('--manifest',default='reports/north_training_data_v38.json');a=p.parse_args()
    prepare(a.output,a.render_root,a.data_root,a.manifest)
