"""Replay source metadata controls, retaining the failed broad annotation check."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

from cresent_geometry_v20 import annotations,audit_frame
from cresent_resolved_camera_v21 import resolved_camera_check
from dataset import sha256
from lunargeo_dataset import write_new


def same(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(float,int)) and math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-8)
    return a==b


def audit(root,output=None):
    root=Path(root);manifest=json.loads((root/'METADATA_MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(root/name)!=digest:raise ValueError('Metadata fixture changed/unsafe: '+name)
    for name,digest in manifest['source_sha256'].items():
        if sha256(name)!=digest:raise ValueError('Metadata interpretation source changed')
    raw=(root/'ground_truth_flight_quaternion.txt').read_bytes()
    if hashlib.md5(raw).hexdigest()!='f6430184dacdf1d25e41e799d6066545':raise ValueError('Published pose-text MD5 differs')
    poses=[]
    for line in raw.decode().splitlines():
        values=line.split()
        if len(values)!=8 or values[0]!='quaternion':raise ValueError('Pose record format differs')
        poses.append([float(v) for v in values[1:]])
    if len(poses)!=15283:raise ValueError('Published pose row count differs')
    header=(root/'cresent-frame0-png-ihdr.bin').read_bytes()
    if len(header)!=33 or header[:8]!=b'\x89PNG\r\n\x1a\n' or header[12:16]!=b'IHDR':raise ValueError('PNG header differs')
    if struct.unpack('>II',header[16:24])!=(1024,1024) or zlib.crc32(header[12:29])&0xffffffff!=int.from_bytes(header[29:33],'big'):
        raise ValueError('PNG metadata/CRC differs')
    def data(frame):
        prefix=root/'controls'
        world=annotations((prefix/f'ground_truth_selenographic_crater_coordinates-{frame}.txt').read_text(),'world')
        ellipses=annotations((prefix/f'ground_truth_projected_ellipses-{frame}.txt').read_text(),'ellipse')
        return world,ellipses
    broad=json.loads((root/'broad_annotation_validation_failed_v20.json').read_text())
    broad_rows=[]
    for saved in broad['frames']:
        frame=saved['frame'];world,ellipses=data(frame);actual=dict(frame=frame,**audit_frame(poses[frame],world,ellipses))
        if not same(saved,actual):raise ValueError('Original broad check does not replay: '+str(frame))
        broad_rows.append(actual)
    if broad['passed'] or all(r['passed'] for r in broad_rows):raise ValueError('Original failure was hidden or relabelled')
    narrow=json.loads((root/'resolved_camera_validation_v21.json').read_text());narrow_rows=[]
    for saved in narrow['frames']:
        frame=saved['frame'];world,ellipses=data(frame);actual=dict(frame=frame,**resolved_camera_check(poses[frame],world,ellipses))
        if not same(saved,actual):raise ValueError('Fixed resolved-camera check does not replay: '+str(frame))
        narrow_rows.append(actual)
    validated=all(r['camera_convention_on_resolved_annotations_passed'] for r in narrow_rows)
    if not validated or len(narrow_rows)!=5:raise ValueError('Declared different-frame camera check failed/incomplete')
    result=dict(replay_passed=True,pose_records=len(poses),original_broad_annotation_gate_passed=False,
        resolved_camera_convention_validated=True,different_resolved_control_frames=5,
        resolved_ellipses=sum(r['measurement']['eligible_ellipses'] for r in narrow_rows),
        reference_depth_geometry_gate_passed=False,fresh_matching_benchmark_completed=False,image_pixels_decoded=0,matcher_inferences=0,
        manifest_sha256=sha256(root/'METADATA_MANIFEST.json'),source_sha256=sha256(__file__),
        scope='Source annotation algebra/pose-field reproducibility only. Not image matching, optical reference precision, actual renderer-depth agreement or fresh navigation accuracy.')
    if output:write_new(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='data/cresent-metadata-v21');parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(audit(args.root,args.output),indent=2))
