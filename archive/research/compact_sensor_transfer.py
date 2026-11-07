"""Fresh POLAR lighting configurations; nominal fixed-camera diagnostics only."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np

from compact_confidence import accepts
from compact_research import matcher_for,predict
from dataset import sha256
from geometry import endpoint_error,evaluation_points
from lunargeo_dataset import write_new
from polar_reference import sensor_image
from remote_archive import HTTPRangeReader


def acquire(output='reports/compact_sensor_protocol_v2.json',root='data/polar/transfer-v2'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    previous=json.loads(Path('data/polar/provenance.json').read_text())
    cases=[];sources=[]
    for scene in previous:
        terrain=scene['terrain'];folder=root/terrain;folder.mkdir(exist_ok=True)
        reader=HTTPRangeReader(scene['archive_url'],scene['archive_size'],block_size=4*1024*1024,cache_dir=root/'.ranges')
        photos={}
        for member in scene['members']:
            match=re.search(r'/Sun_(30|180)/Cam([LR])_\d+_1024.png$',member['source_member'])
            if match:
                path=Path('data/polar')/terrain/member['local_name']
                if sha256(path)!=member['sha256']:raise ValueError('Previously acquired sensor file changed')
                photos[(int(match[1]),match[2])]=str(path)
        with zipfile.ZipFile(reader) as archive:
            selected=[i for i in archive.infolist() if re.search(r'/PosA_1500_Loff/Sun_(270|350)/Cam[LR]_\d+_1024.png$',i.filename)]
            if len(selected)!=4:raise ValueError('Expected4 original photos for fresh270/350 lighting configurations')
            for member in selected:
                match=re.search(r'/Sun_(270|350)/Cam([LR])_\d+_1024.png$',member.filename)
                content=archive.read(member);path=folder/Path(member.filename).name
                digest=hashlib.sha256(content).hexdigest()
                if path.exists() and sha256(path)!=digest:raise ValueError('Cached fresh sensor bytes differ')
                if not path.exists():path.write_bytes(content)
                photos[(int(match[1]),match[2])]=str(path)
                sources.append(dict(path=str(path),sha256=digest,source_member=member.filename,crc32=member.CRC,
                    archive_url=scene['archive_url'],terrain=terrain))
        for a,b in itertools.combinations((30,180,270,350),2):
            if (a,b)==(30,180):continue
            for camera in ('L','R'):
                cases.append(dict(pair_id=f'{terrain}-{camera}-{a}-{b}',terrain=terrain,camera=camera,
                    reference=photos[(a,camera)],query=photos[(b,camera)],sun_a_deg=a,sun_b_deg=b,
                    component='shared-POLAR-laboratory-rocks-and-rig',track='nominal_sensor_transfer'))
    doc=dict(schema_version=2,cases=cases,sources=sources,
        files_sha256={p:sha256(p) for c in cases for p in (c['reference'],c['query'])},
        source_sha256=sha256(__file__),selected_before_new_inference=True,
        known_terrain='Terrains01,06,11,12 were inspected earlier. Only new lighting configurations are fresh; reused rocks/rig do not establish independent terrain.',
        reference='Nominal identity for fixed camera. Calibration/drift pixel uncertainty unknown; do not fit confidence to these labels or claim certified accuracy.',
        preprocessing='Existing frozen POLAR center1216 crop, fixed12bit-to8bit, INTER_AREA128',
        training_permitted=False,calibration_permitted=False)
    write_new(output,doc);print('Fresh POLAR lighting cases',len(cases),flush=True);return doc


def run(sensor_protocol,terrain_protocol,publication_commit,output,method,model=None,source=None,
        device='cpu',dependency_dir=None):
    import torch,cv2
    torch.set_num_threads(2);cv2.setNumThreads(1)
    from compact_research import validate_freeze
    doc=validate_freeze(terrain_protocol,publication_commit)
    sensor=json.loads(Path(sensor_protocol).read_text())
    if doc['sensor_protocol_sha256']!=sha256(sensor_protocol):raise ValueError('Sensor protocol is not publicly frozen')
    for path,digest in sensor['files_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Sensor source changed')
    key=model or method
    policy=json.loads(Path(doc['policy_index'][key]).read_text())
    matcher=matcher_for(method,model,source,device,dependency_dir)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);rows=[]
    write_new(output/'frozen_inputs.json',dict(sensor_protocol=sensor_protocol,
        sensor_protocol_sha256=sha256(sensor_protocol),terrain_protocol=terrain_protocol,
        terrain_protocol_sha256=sha256(terrain_protocol),publication_commit=publication_commit,
        method=method,model=model,policy_sha256=sha256(doc['policy_index'][key]),
        source_sha256=sha256(__file__)))
    from synthetic import texture
    warm=texture(20261030,128);matcher(warm,warm)
    for case in sensor['cases']:
        a,ma=sensor_image(case['reference']);b,mb=sensor_image(case['query'])
        found,_,features,_,ms,original=predict(matcher,a,b,ma,mb,method)
        accepted=accepts(policy,{'features':features.tolist()})
        identity=np.array([[1.,0,0],[0,1.,0]])
        error=endpoint_error(found.matrix,identity,evaluation_points(a.shape,identity))
        rows.append(dict(case,method=method,features=features.tolist(),nominal_error_px=error,
            nominal_error_native_px=None if error is None else float(error*1216/128),
            native_pixels_per_benchmark_pixel=1216/128,
            raw_transform=None if found.matrix is None else found.matrix.tolist(),
            nominal_within3px=error is not None and error<=3,accepted=accepted,
            deployed_transform=found.matrix.tolist() if accepted else None,
            original_frozen_accepted=original,bidirectional_ms=ms,
            warning='Nominal fixed-camera agreement; actual reference uncertainty unknown'))
    write_new(output/'results.json',rows)
    write_new(output/'summary.json',dict(method=method,cases=len(rows),
        raw_nominal_within3px=sum(r['nominal_within3px'] for r in rows),
        accepted_nominal_within3px=sum(r['accepted'] and r['nominal_within3px'] for r in rows),
        accepted_nominal_outside3px=sum(r['accepted'] and not r['nominal_within3px'] for r in rows),
        original_gate_accepted_nominal_within3px=sum(bool(r['original_frozen_accepted']) and r['nominal_within3px'] for r in rows),
        original_gate_accepted_nominal_outside3px=sum(bool(r['original_frozen_accepted']) and not r['nominal_within3px'] for r in rows),
        verified_navigation_accuracy=None))
    write_new(output/'COMPLETE.json',dict(results_sha256=sha256(output/'results.json'),
        summary_sha256=sha256(output/'summary.json'),frozen_inputs_sha256=sha256(output/'frozen_inputs.json')))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('acquire');a.add_argument('--output',default='reports/compact_sensor_protocol_v2.json');a.add_argument('--root',default='data/polar/transfer-v2')
    r=sub.add_parser('run');r.add_argument('--sensor-protocol',required=True);r.add_argument('--terrain-protocol',required=True)
    r.add_argument('--publication-commit',required=True);r.add_argument('--output',required=True);r.add_argument('--method',required=True)
    r.add_argument('--model');r.add_argument('--source');r.add_argument('--device',default='cpu');r.add_argument('--dependency-dir')
    args=vars(p.parse_args());action=args.pop('action')
    if action=='acquire':acquire(**args)
    else:run(**args)
