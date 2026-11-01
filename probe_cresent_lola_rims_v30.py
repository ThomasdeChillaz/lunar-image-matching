"""Development metadata-only crater rim/map consistency; no camera pixels/depth."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
import mmap
from pathlib import Path
import statistics
import struct

from cresent_geometry_v20 import annotations
from lunar_pds_height_v30 import parse_label


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def rim_points(world,samples=360):
    x,y,z,a,b,phi=world;radius=math.sqrt(x*x+y*y+z*z);rho=math.hypot(x,y)
    if radius==0 or rho==0 or min(a,b)<=0:raise ValueError('Nonpolar finite crater required')
    east=(-y/rho,x/rho,0.);up=(x/radius,y/radius,z/radius)
    north=(up[1]*east[2]-up[2]*east[1],up[2]*east[0]-up[0]*east[2],up[0]*east[1]-up[1]*east[0])
    c,s=math.cos(phi),math.sin(phi)
    out=[]
    for i in range(samples):
        t=2*math.pi*i/samples;ee=a*math.cos(t)*c-b*math.sin(t)*s;nn=a*math.cos(t)*s+b*math.sin(t)*c
        p=[v+ee*e+nn*n for v,e,n in zip((x,y,z),east,north)]
        out.append((math.degrees(math.atan2(p[2],math.hypot(p[0],p[1]))),math.degrees(math.atan2(p[1],p[0]))%360))
    return radius,out


def bilinear_radius(tile,buffer,latitude,longitude):
    row,col=tile.pixel(latitude,longitude);r,c=math.floor(row),math.floor(col)
    if not(0<=r<tile.rows-1 and 0<=c<tile.columns-1):raise ValueError('Interpolation crosses explicit tile edge')
    dy,dx=row-r,col-c
    values=[struct.unpack_from('<h',buffer,tile.offset(rr,cc))[0]*tile.scale_m+tile.reference_radius_m
            for rr,cc in ((r,c),(r,c+1),(r+1,c),(r+1,c+1))]
    return ((1-dy)*((1-dx)*values[0]+dx*values[1])+dy*((1-dx)*values[2]+dx*values[3]))


def run(map_root,output):
    root,out=Path(map_root),Path(output)
    if out.exists() or out.with_name(out.stem+'_protocol.json').exists():raise FileExistsError('Preserve prior probes and protocols')
    complete=read(root/'COMPLETE.json');protocol=read(root/'DOWNLOAD_PROTOCOL.json')
    tile=parse_label((root/'LDEM_512_45N_90N_000_090.LBL').read_text());image=root/tile.filename
    if not complete['complete'] or image.stat().st_size!=tile.bytes or sha(image)!=complete['sha256']:
        raise ValueError('Complete acquired map bytes must verify before sampling')
    fixture=read('pds_height_fixtures_v30.json')
    if sha(root/'LDEM_512_45N_90N_000_090.LBL')!=protocol['source_label_sha256']:
        raise ValueError('Acquired official label changed')
    source=Path('data/cresent-metadata-v21/controls/ground_truth_selenographic_crater_coordinates-0.txt')
    annotation=annotations(source.read_text(),'world');chosen=[];excluded={}
    for name,world in annotation.items():
        if not all(math.isfinite(v) for v in world) or not (500<=min(world[3:5]) and max(world[3:5])<=30000):
            excluded['finite_axes_500_to_30000m']=excluded.get('finite_axes_500_to_30000m',0)+1;continue
        radius,points=rim_points(world)
        try:
            for latitude,longitude in points:
                r,c=tile.pixel(latitude,longitude)
                if not (0<=r<tile.rows-1 and 0<=c<tile.columns-1):raise ValueError('edge')
        except ValueError:
            excluded['rim_outside_one_tile']=excluded.get('rim_outside_one_tile',0)+1;continue
        chosen.append(dict(crater_id=name,world=world,radius=radius,points=points))
        if len(chosen)==40:break
    if len(chosen)!=40:raise ValueError('Fewer than40metadata-eligible craters; no smaller substituted study')
    contract=dict(scope='Metadata-control frame0 only. Compare released median-rim radii to a declared360-angle map sampling approximation; not true camera depth or optical pixel registration.',
        map_sha256=complete['sha256'],label_sha256=protocol['source_label_sha256'],source_annotation_sha256=sha(source),
        selection='First40 published rows with both semiaxes500..30000m, finite parameters, all360projected rim positions inside this explicit tile interpolation support. No map value used for selection.',
        crater_ids=[v['crater_id'] for v in chosen],samples_per_crater=360,excluded_during_selection=excluded,
        coordinate_frame=tile.frame,frame_transform_fitted=False,
        predetermined_pass_threshold=None,source_optical_pixels_decoded=0,matcher_inferences=0,
        created_utc=datetime.now(timezone.utc).isoformat())
    pp=out.with_name(out.stem+'_protocol.json');pp.write_text(json.dumps(contract,indent=2)+'\n')
    records=[]
    with image.open('rb')as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)as buffer:
        for sample in fixture['official_labels']['LDEM_512_45N_90N_000_090.LBL']['samples']:
            start=sample['trace']['start'];raw=buffer[start:start+4]
            if hashlib.sha256(raw).hexdigest()!=sample['trace']['sha256']:raise ValueError('Downloaded map differs from earlier range probe')
        for c in chosen:
            values=[bilinear_radius(tile,buffer,*point) for point in c['points']]
            estimate=statistics.median(values)
            records.append(dict(crater_id=c['crater_id'],released_median_rim_radius_m=c['radius'],
                approximate_map_median_rim_radius_m=estimate,difference_m=estimate-c['radius'],
                sampled_map_minimum_m=min(values),sampled_map_maximum_m=max(values)))
    errors=[r['difference_m'] for r in records]
    result=dict(protocol_sha256=sha(pp),source_sha256=sha(__file__),map_parser_sha256=sha('lunar_pds_height_v30.py'),
        map_sha256=complete['sha256'],map_bytes=complete['bytes'],records=records,
        craters=len(records),rim_radius_queries=40*360,
        median_signed_difference_m=statistics.median(errors),median_absolute_difference_m=statistics.median(map(abs,errors)),
        maximum_absolute_difference_m=max(map(abs,errors)),
        optical_pixels_decoded=0,matcher_inferences=0,camera_depth_created=False,
        actual_renderer_registration_verified=False,fresh_test_gate_passed=False,
        limits=['One already exposed metadata-control frame;40craters are dependent map queries, not independent terrain tests.',
            'The authors exact rim sampling algorithm, DEM product/version and interpolation have not been independently recovered.',
            'Agreement is evidence about radius units/map indexing and nominal terrain, not a guarantee of optical registration, rendered visibility or map error bounds.',
            'No transform, threshold, camera pose or matcher is fitted to these outcomes. This does not repair the original LunarGeo freshness shortfall.'])
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--map-root',default='data/lola-reference-v30');p.add_argument('--output',default='reports/cresent_lola_rim_probe_v30.json');a=p.parse_args();run(a.map_root,a.output)
