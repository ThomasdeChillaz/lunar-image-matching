"""Metadata-only comparison to this project's historical lunar pixel inputs.

This audits the named project ledgers, not unknown external model pretraining.
It does not read new terrain heights, images or model outputs.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

from cresent_geometry_v20 import annotations
from lunar_pds_height_v30 import parse_label
from north_render_plan_v31 import RADIUS,inverse_north_stereo
from north_terrain_pilot_v32 import pilot_plan,sha
from probe_cresent_lola_rims_v30 import rim_points


def projected(latitude,longitude):
    latitude,longitude=math.radians(latitude),math.radians(longitude)
    rho=2*RADIUS*math.tan((math.pi/2-latitude)/2)
    return rho*math.sin(longitude),-rho*math.cos(longitude)


def inside_support(x,y,site,padding=0.):
    return abs(x-site['x'])<=10240+padding and abs(y-site['y'])<=10240+padding


def audit(output):
    from rasterio.warp import transform
    plan,_=pilot_plan();out=Path(output)
    if out.exists():raise FileExistsError('Preserve earlier exposure audits')
    sources={};history=[]
    for name,key in [('moon_manifest.json','tiles'),('research_manifest_v1.json','tiles'),
                     ('tycho_real_manifest.json','assets')]:
        m=json.loads(Path(name).read_text());sources[name]=sha(name)
        bounds=[t['bounds_lonlat_deg']for t in m[key]]
        if not bounds or any(len(b)!=4 or not all(math.isfinite(v)for v in b)for b in bounds):
            raise ValueError('Incomplete historical geographic ledger')
        history.append(dict(ledger=name,records=len(bounds),minimum_latitude_deg=min(b[1]for b in bounds),
                            maximum_latitude_deg=max(b[3]for b in bounds)))
    name='reports/compact_dataset_v2.json';m=json.loads(Path(name).read_text());sources[name]=sha(name)
    latitudes=[]
    for tile in m['tiles']:
        low,high=tile['world_bounds']
        if len(low)!=3 or len(high)!=3 or high[2]>=0:raise ValueError('Expected audited southern body-space supports')
        # Bounding box with negative Z: maximum latitude uses its highest Z
        # and largest possible distance from the polar axis, including edges.
        rho=math.hypot(max(abs(low[0]),abs(high[0])),max(abs(low[1]),abs(high[1])))
        latitudes.append(math.degrees(math.atan2(high[2],rho)))
    history.append(dict(ledger=name,records=len(latitudes),maximum_latitude_deg=max(latitudes),
                        bound_type='Conservative complete body-space support boxes, not camera centres.'))
    name='expanded_solar_manifest.json';m=json.loads(Path(name).read_text());sources[name]=sha(name)
    optical=[]
    for asset in m['assets']:
        left,bottom,right,top=asset['bounds_m']
        # Edges densified in map coordinates, independent of source image pixels.
        coordinates=[]
        for i in range(21):
            t=i/20
            coordinates.extend([(left+t*(right-left),bottom),(left+t*(right-left),top),
                                (left,bottom+t*(top-bottom)),(right,bottom+t*(top-bottom))])
        x,y=zip(*coordinates)
        lon,lat=transform(asset['crs_wkt'],f'+proj=longlat +R={RADIUS} +no_defs',x,y)
        optical.extend(lat)
    history.append(dict(ledger=name,records=len(m['assets']),minimum_latitude_deg=min(optical),
                        maximum_latitude_deg=max(optical),bound_type='21samples on each crop edge;1degree conservative guard applied.'))
    north_minimum=min(math.degrees(inverse_north_stereo(s['x']+dx,s['y']+dy)[0])
                      for s in plan['sites']for dx,dy in itertools.product((-10240,10240),repeat=2))
    historical_maximum=max(h['maximum_latitude_deg']for h in history)
    if historical_maximum+1>=north_minimum:raise ValueError('Historical latitude separation cannot establish nonoverlap')
    # Previously exposed northern height probes did not fit a matcher. Account
    # for them separately rather than silently omitting metadata inspection.
    name='data/cresent-metadata-v21/controls/ground_truth_selenographic_crater_coordinates-0.txt'
    a=annotations(Path(name).read_text(),'world');sources[name]=sha(name)
    name='reports/cresent_lola_rim_probe_v30_protocol.json';contract=json.loads(Path(name).read_text());sources[name]=sha(name)
    probes=[]
    for cid in contract['crater_ids']:
        for scale in (1.,.5):
            world=a[cid].copy();world[3]*=scale;world[4]*=scale
            _,points=rim_points(world)
            probes.extend(projected(lat,lon)for lat,lon in points)
    name='pds_height_fixtures_v30.json';fixture=json.loads(Path(name).read_text());sources[name]=sha(name)
    for entry in fixture['official_labels'].values():
        tile=parse_label(entry['label'])
        for sample in entry['samples']:
            latitude=tile.north-(sample['row']+.5)/tile.pixels_per_degree
            for col in range(sample['column'],sample['column']+sample['width']):
                longitude=tile.west+(col+.5)/tile.pixels_per_degree
                probes.append(projected(latitude,longitude))
    intersections=[]
    for site in plan['sites']:
        count=sum(inside_support(x,y,site,padding=1000.)for x,y in probes)
        if count:intersections.append(dict(component=site['component'],split=site['split'],points=count))
    result=dict(passed=not intersections,scope='Own-project named historical pixel ledgers and known height-probe supports only; metadata check, not a model/generalization result.',
        historical_inputs=history,candidate_full_support_minimum_latitude_deg=north_minimum,
        historical_pixel_maximum_latitude_deg=historical_maximum,latitude_guard_deg=1.,
        minimum_guarded_latitude_gap_deg=north_minimum-historical_maximum-1.,
        prior_height_probe_locations=len(probes),probe_guard_m=1000.,probe_support_intersections=intersections,
        own_project_historical_pixel_supports_disjoint=True,external_model_pretraining_overlap='unknown; not certified by this audit',
        original_released_lunargeo_fresh_gate_passed=False,final_model_inference_performed=False,
        source_sha256=sha(__file__),metadata_sha256=sources,
        limitations=['Named input ledgers cover recorded project inputs, not all lunar imagery ever seen by external models.',
                     'Same DEM, renderer and material cause shared dependence;44supports are not independent missions.',
                     'Known prior height probes are inspected separately; no undisclosed raster-wide exposure audit is claimed.'])
    out.write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise ValueError('Prior metadata probe intersects candidate supports; preserve and limit freshness claim')
    print(json.dumps({k:v for k,v in result.items()if k not in ['metadata_sha256','historical_inputs']}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/north_historical_exposure_v34.json')
    audit(p.parse_args().output)
