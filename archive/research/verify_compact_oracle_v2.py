"""Post-study development-only oracle geometry control, without matcher inference."""
import argparse
from pathlib import Path
import json,cv2,numpy as np
from compact_research import load_tile
from lunar_geometry_v2 import correspondences,estimate_pose,pose_error
from dataset import sha256

def run(output):
    manifest='reports/compact_dataset_v2.json';d=json.loads(Path(manifest).read_text());cache={};seen=set();rows=[]
    for tile in d['tiles']:
     if tile['split']!='model_selection' or tile['mode'] in seen:continue
     seen.add(tile['mode']);data=load_tile(tile,cache);a,b=tile['cameras']
     _,pixels,world,valid=correspondences(data['depth_a'],data['depth_b'],np.array(a['intrinsics']),np.array(a['cam2world']),np.array(b['intrinsics']),np.array(b['cam2world']),stride=16)
     cv2.setRNGSeed(20261030);pose=estimate_pose(world,pixels,np.array(b['intrinsics']))
     err=None if pose is None else pose_error(pose,np.array(b['cam2world']))
     ranges=data['depth_a'][np.isfinite(data['depth_a'])&(data['depth_a']>0)];scale=float(np.median(ranges))
     passed=err is not None and err['rotation_deg']<2 and err['translation_m']<.01*scale
     rows.append(dict(tile_id=tile['tile_id'],mode=tile['mode'],split=tile['split'],exact_visible_correspondences=len(world),pose_error=err,reference_range_m=scale,passed=passed))
    assert len(rows)==3 and all(r['passed'] for r in rows),rows
    p=Path(output);assert not p.exists();p.write_text(json.dumps(dict(passed=True,scope='Post-evaluation software diagnostic on metadata-first development tile per acquisition mode. Exact camera-projected query points, not matcher predictions; no final metric, algorithm or threshold changed.',manifest_sha256=sha256(manifest),source_sha256={x:sha256(x) for x in ['lunar_geometry_v2.py','compact_research.py','verify_compact_oracle_v2.py']},rows=rows),indent=2)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    run(parser.parse_args().output)
