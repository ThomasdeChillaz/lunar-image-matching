"""Draft pilot inference/scoring bundle; no fitting/calibration/final split."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from north_render_plan_v31 import plan,support_overlap

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
LIGHT_PAIRS=((0,0),(0,3),(0,4),(1,5))


def make_cases(tiles):
 if not tiles or len({t['tile_id']for t in tiles})!=len(tiles):raise ValueError('Nonempty unique eligible pilot geometries required')
 if any(t['split']!='pilot_camera_control'for t in tiles):raise ValueError('Pilot-only scoring split required')
 cases=[];negatives=[]
 for t in sorted(tiles,key=lambda x:x['tile_id']):
  components=sorted({x['component']for x in tiles if x['family']==t['family']})
  if len(components)<2:raise ValueError('Two distinct terrain endpoints per family required')
  other=components[(components.index(t['component'])+1)%len(components)]
  candidates=[x for x in tiles if x['component']==other and x['family']==t['family']]
  q=min(candidates,key=lambda x:(abs(x['pair_index']-t['pair_index']),x['pair_index'],x['tile_id']))
  if support_overlap(t['support'],q['support']):raise ValueError('Nonoverlap endpoints share source support')
  for a,b in LIGHT_PAIRS:
   c=dict(pair_id=t['tile_id']+f'/lights-{a}-{b}',tile_id=t['tile_id'],component=t['component'],
    split='pilot_camera_control',family=t['family'],query_view=t['pair_index'],light_a=a,light_b=b,positive=True)
   cases.append(c);negatives.append(dict(c,pair_id=c['pair_id']+'/nonoverlap-'+other,positive=False,query_tile_id=q['tile_id'],query_component=other))
 return cases,negatives


def prepare(render_root,audit_path,output,manifest_path):
 from PIL import Image
 raw=Path(render_root);root=Path(output);manifest_path=Path(manifest_path);s=read(raw/'summary.json');a=read(audit_path)
 if not a['passed']or a['oracle_success']<.95 or a['summary_sha256']!=sha(raw/'summary.json')or not s['complete']:raise ValueError('Actual independently audited varied-camera pilot required')
 if a['source_sha256']!=sha('verify_north_camera_v59.py'):raise ValueError('Geometry audit source changed')
 if root.exists()or manifest_path.exists():raise FileExistsError('Preserve existing bundle/manifest')
 sites={x['component']:x for x in plan()['sites']if x['split']=='renderer_cost_pilot'}
 if {r['component']for r in s['pairs']}!=set(sites):raise ValueError('Exactly the four excluded pilot supports required')
 root.mkdir(parents=True);tiles=[];images={(r['component'],r['pair_id'],r['role'],r['light']):r for r in s['images']}
 for row in s['pairs']:
  if not row['eligible']:continue
  component=row['component'];tile_id=component+'/'+row['pair_id'];geometry={};interleaved=[]
  for role in ('reference','query'):
   g=row['views'][role];path=raw/g['path']
   if sha(path)!=g['sha256']:raise ValueError('Audited pilot geometry changed')
   with np.load(path,allow_pickle=False)as z:geometry[role]={n:z[n].copy()for n in ('axial_depth','cam2world','intrinsics','body_origin','enu_rows')}
  for light in range(6):
   for role in ('reference','query'):
    image=images[(component,row['pair_id'],role,light)];path=raw/image['image']
    if sha(path)!=image['sha256']:raise ValueError('Audited pilot image changed')
    with Image.open(path)as im:pixels=np.asarray(im.convert('L')).copy()
    if pixels.shape!=(512,512)or pixels.dtype!=np.uint8:raise ValueError('Native grayscale image differs')
    interleaved.append(pixels)
  forward=next(x for x in row['labels']if x['source']=='reference');reverse=next(x for x in row['labels']if x['source']=='query')
  labels=[]
  for record in (forward,reverse):
   path=raw/record['path']
   if sha(path)!=record['sha256']:raise ValueError('Audited exact3D labels changed')
   with np.load(path,allow_pickle=False)as z:labels.append({n:z[n].copy()for n in ('target_points','visible','geometrically_occluded','visibility_uncertain')})
  ref,query=geometry['reference'],geometry['query'];cache=root/(hashlib.sha256(tile_id.encode()).hexdigest()+'.npz')
  np.savez_compressed(cache,images=np.stack(interleaved),depth_a=ref['axial_depth'],depth_b=query['axial_depth'],
   forward_target=labels[0]['target_points'],forward_visible=labels[0]['visible'],reverse_target=labels[1]['target_points'],reverse_visible=labels[1]['visible'],forward_geometrically_occluded=labels[0]['geometrically_occluded'],reverse_geometrically_occluded=labels[1]['geometrically_occluded'],forward_visibility_uncertain=labels[0]['visibility_uncertain'],reverse_visibility_uncertain=labels[1]['visibility_uncertain'])
  tiles.append(dict(tile_id=tile_id,component=component,split='pilot_camera_control',family=row['family'],pair_index=row['pair_index'],
   native_cache=str(cache),native_cache_sha256=sha(cache),support={k:sites[component][k]for k in ('x','y')},
   body_origin=ref['body_origin'].tolist(),enu_rows=ref['enu_rows'].tolist(),
   cameras=[{n:g[n].tolist()for n in ('intrinsics','cam2world')}for g in (ref,query)]))
 cases,negatives=make_cases(tiles)
 result=dict(schema_version=59,source_sha256=sha(__file__),tiles=tiles,cases=cases,negative_cases=negatives,
  geometry_audit_path=str(audit_path),geometry_audit_sha256=sha(audit_path),renderer_summary_path=str(raw/'summary.json'),renderer_summary_sha256=sha(raw/'summary.json'),
  preprocessing='Pillow RGB-to-L uint8 native512, same convention as northv38.',
  negative_pairing='Fixed cyclic different terrain, same motion family, closest pair ordinal; both terrain endpoints retained.',
  calibration_or_final_pixels_loaded=0,fitting_allowed=False,
  scope='Four old pilot supports and varied camera motions, inference/scoring only. Not a training set, calibration set, final test or new independent geographic evidence.')
 with manifest_path.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 return result


def load_bundle(path):
 m=read(path)
 if m['schema_version']!=59 or m['fitting_allowed']or m['calibration_or_final_pixels_loaded']or m['source_sha256']!=sha(__file__):raise ValueError('Frozen pilot-only varied-camera bundle required')
 if sha(m['geometry_audit_path'])!=m['geometry_audit_sha256']or not read(m['geometry_audit_path'])['passed']or sha(m['renderer_summary_path'])!=m['renderer_summary_sha256']:raise ValueError('Audited pilot evidence changed')
 expected=make_cases(m['tiles'])
 if (m['cases'],m['negative_cases'])!=expected:raise ValueError('Frozen full pilot case/endpoint lists changed')
 tiles={t['tile_id']:t for t in m['tiles']};data={}
 for key,t in tiles.items():
  if sha(t['native_cache'])!=t['native_cache_sha256']:raise ValueError('Prepared pilot cache changed')
  with np.load(t['native_cache'],allow_pickle=False)as z:data[key]={k:z[k].copy()for k in z.files}
  z=data[key]
  if z['images'].shape!=(12,512,512)or z['images'].dtype!=np.uint8 or z['depth_a'].shape!=(512,512)or z['depth_b'].shape!=(512,512)or z['forward_target'].shape!=(676,2):raise ValueError('Native pilot cache schema differs')
 return m,tiles,data


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--render-root',required=True);p.add_argument('--audit',required=True);p.add_argument('--output',required=True);p.add_argument('--manifest',required=True);a=p.parse_args();prepare(a.render_root,a.audit,a.output,a.manifest)
