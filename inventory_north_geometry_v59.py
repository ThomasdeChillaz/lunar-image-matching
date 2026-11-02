"""Inventory existing training BVH labels; no image/depth or matcher calls."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inventory(manifest_path,output):
 manifest=Path(manifest_path);out=Path(output)
 if out.exists():raise FileExistsError('Preserve previous label inventory')
 m=json.loads(manifest.read_text());raw=Path(m['renderer_summary_path']).parent
 if m['schema_version']!=38 or m['lighting_count']!=6:raise ValueError('Original audited north training bundle required')
 rows=[]
 for tile in m['tiles']:
  if tile['split']!='train':continue
  name=f'supervision-forward-0-{tile["query_view"]}.npz';path=raw/tile['component']/name
  if sha(path)!=tile['exact_supervision_sha256'][name]:raise ValueError('Audited exact training labels changed')
  with np.load(path,allow_pickle=False)as z:valid,inside,visible,target=[z[k].copy()for k in ('source_valid','in_frame','visible','target_points')]
  if any(a.shape!=(676,)or a.dtype!=bool for a in (valid,inside,visible))or target.shape!=(676,2):raise ValueError('Exact BVH mask schema differs')
  if np.any(visible&~inside)or np.any(inside&~valid):raise ValueError('Geometric visibility hierarchy differs')
  hidden=valid&inside&~visible;outside=valid&~inside
  if int(visible.sum()+hidden.sum()+outside.sum()+(~valid).sum())!=676:raise ValueError('Geometry partition incomplete')
  rows.append(dict(tile_id=tile['tile_id'],component=tile['component'],visible=int(visible.sum()),in_frame_nonvisible=int(hidden.sum()),outside_fov_or_behind=int(outside.sum()),invalid_source=int((~valid).sum()),safe_visible_local_targets=int((visible&(target>=42).all(1)&(target<=470).all(1)).sum()),labels_sha256=sha(path)))
 if len(rows)!=48 or len({r['component']for r in rows})!=16:raise ValueError('Complete original training split required')
 result=dict(scope='Exact-mask inventory on48training geometry banks/16terrain components. Samples repeat physical terrain across viewpoints and are not independent terrain tests. In-frame nonvisibility is an upperbound on occlusion; missing first hits cannot be confirmed as occluders from these legacy masks alone.',manifest_sha256=sha(manifest),source_sha256=sha(__file__),training_geometries=48,training_components=16,total_point_samples=32448,totals={k:sum(r[k]for r in rows)for k in ('visible','in_frame_nonvisible','outside_fov_or_behind','invalid_source','safe_visible_local_targets')},components_with_in_frame_nonvisibility=len({r['component']for r in rows if r['in_frame_nonvisible']>0}),rows=rows,image_or_depth_arrays_loaded=0,neural_forwards=0,optimizer_updates=0,model_promoted=False)
 with out.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items()if k!='rows'},indent=2));return result


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default='reports/north_training_data_v38.json');p.add_argument('--output',default='reports/north_training_geometry_inventory_v59.json');a=p.parse_args();inventory(a.manifest,a.output)
