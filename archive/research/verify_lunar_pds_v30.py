"""Offline independent byte arithmetic for the small official LOLA range probe."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

from lunar_pds_height_v30 import BASE,parse_label


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(fixture,output):
    p=Path(fixture);data=json.loads(p.read_text());samples=0;labels=0
    for name,source in data['official_labels'].items():
        raw_label=source['label'].encode()
        if hashlib.sha256(raw_label).hexdigest()!=source['label_sha256'] or source['url']!=BASE+name:
            raise ValueError('Official label bytes/source changed')
        tile=parse_label(source['label']);labels+=1
        if len(source['samples'])!=2:raise ValueError('Expected two contiguous2sample row fragments')
        rr=(tile.north-data['protocol']['latitude'])*tile.pixels_per_degree-.5
        cc=(data['protocol']['longitude']-tile.west)*tile.pixels_per_degree-.5
        for i,sample in enumerate(source['samples']):
            raw=base64.b64decode(sample['data_base64'],validate=True);trace=sample['trace']
            if len(raw)!=4 or hashlib.sha256(raw).hexdigest()!=trace['sha256']:
                raise ValueError('Sample range bytes changed')
            independent=[]
            for j in (0,2):
                dn=raw[j]+256*raw[j+1]
                if dn>=32768:dn-=65536
                independent.append(1737400.+dn/2.)
            start=2*((math.floor(rr)+i)*tile.columns+math.floor(cc))
            if (sample['row']!=math.floor(rr)+i or sample['column']!=math.floor(cc) or sample['width']!=2
                    or independent!=sample['planetary_radius_m'] or tile.radii(raw)!=independent
                    or trace['url']!=BASE+tile.filename or trace['start']!=start
                    or trace['length']!=4 or trace['total_bytes']!=tile.rows*tile.columns*2
                    or trace['content_range']!=f'bytes {start}-{start+3}/{tile.rows*tile.columns*2}'
                    or trace['complete_source_checksum_verified']):
                raise ValueError('Independent signed arithmetic/geographic offset differs')
            samples+=2
    tile=parse_label(data['unused_label']['label']);labels+=1
    if (samples!=8 or labels!=3 or data['decoded_height_map_samples']!=8
            or data['source_optical_pixels_decoded'] or data['full_IMG_checksum_verified']
            or data['renderer_depth_agreement_verified']):raise ValueError('Probe scope changed')
    result=dict(passed=True,official_labels_replayed=labels,decoded_height_samples_replayed=samples,
        independent_signed_byte_and_map_offset_arithmetic=True,fixture_sha256=digest(p),
        parser_source_sha256=digest('lunar_pds_height_v30.py'),auditor_source_sha256=digest(__file__),
        source_optical_pixels_decoded=0,matcher_inferences=0,camera_ray_depth_created=False,
        complete_IMG_checksum_verified=False,renderer_frame_and_surface_agreement_verified=False,
        fresh_test_gate_passed=False,
        scope='Eight official reference-map samples/three PDS labels only. Height radii are not camera depth, optical registration, terrain visibility, independent fresh localization or model improvement.')
    out=Path(output)
    if out.exists():raise FileExistsError('Preserve previous map-probe audits')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixture',default='pds_height_fixtures_v30.json');p.add_argument('--output',default='reports/lunar_pds_height_probe_v30.json');a=p.parse_args();audit(a.fixture,a.output)
