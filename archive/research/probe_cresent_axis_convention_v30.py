"""One declared alternative after a retained failed nominal-rim consistency probe."""
from pathlib import Path
import sys,os,json,math,mmap,statistics,hashlib
repo=Path.cwd()
from cresent_geometry_v20 import annotations
from lunar_pds_height_v30 import parse_label
from probe_cresent_lola_rims_v30 import rim_points,bilinear_radius
root=Path('data/lola-reference-v30');old=json.loads(Path('reports/cresent_lola_rim_probe_v30.json').read_text());contract=json.loads(Path('reports/cresent_lola_rim_probe_v30_protocol.json').read_text());annotation=annotations(Path('data/cresent-metadata-v21/controls/ground_truth_selenographic_crater_coordinates-0.txt').read_text(),'world');tile=parse_label((root/'LDEM_512_45N_90N_000_090.LBL').read_text())
p=Path('reports/cresent_axis_alternative_v30_protocol.json')
protocol=dict(scope='Exploratory diagnosis after the retained360-angle released-axes probe observed312m median difference. Compare one fixed alternative: treat released a,b as full diameters for terrain sampling only. Do not rewrite released conic/pose labels or assert physical axis semantics.',same_40_crater_ids=contract['crater_ids'],source_protocol_sha256=hashlib.sha256(Path('reports/cresent_lola_rim_probe_v30_protocol.json').read_bytes()).hexdigest(),axis_sampling_scale=.5,points=360,no_fitted_offset=True,optical_pixels_decoded=0,matcher_inferences=0,fresh_test_gate=False)
if p.exists():raise FileExistsError('Preserve previous alternative contract')
p.write_text(json.dumps(protocol,indent=2)+'\n');records=[]
with (root/tile.filename).open('rb')as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)as buffer:
 for name in contract['crater_ids']:
  world=annotation[name].copy();world[3]*=.5;world[4]*=.5;radius,points=rim_points(world)
  prediction=statistics.median([bilinear_radius(tile,buffer,*point)for point in points])
  records.append(dict(crater_id=name,released_radius_m=radius,approximate_half_axes_rim_median_m=prediction,difference_m=prediction-radius))
errors=[r['difference_m']for r in records]
result=dict(protocol=protocol,records=records,median_signed_difference_m=statistics.median(errors),median_absolute_difference_m=statistics.median(map(abs,errors)),maximum_absolute_difference_m=max(map(abs,errors)),source_optical_pixels_decoded=0,matcher_inferences=0,actual_axis_semantics_confirmed=False,renderer_registration_verified=False,fresh_test_gate_passed=False)
Path('reports/cresent_axis_alternative_v30_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in('protocol','records')},indent=2))
