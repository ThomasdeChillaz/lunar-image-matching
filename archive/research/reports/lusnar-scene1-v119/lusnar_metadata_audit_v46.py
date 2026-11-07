"""Validate recovered development-only CSV metadata; no optical/depth pixels.

Pose CSV identifies rover/sensor state, not verified camera extrinsics. Archive
prefix is incomplete; previously CRC-verified complete members are checked bySHA.
"""
from pathlib import Path
from bisect import bisect_left
from datetime import datetime,timezone
import csv,hashlib,json,math,re,statistics

def ns(value):
 if not re.fullmatch(r'[0-9]+',value):raise ValueError('Exact integer nanosecond timestamp required; no float conversion')
 return int(value)
def rows(text):
 data=list(csv.reader(text.splitlines()))
 if not data or not data[0][0].startswith('#timestamp'):raise ValueError('Expected original CSV timestamp header')
 return data[0],data[1:]
def ordered(t):
 if not t or any(b<=a for a,b in zip(t,t[1:])):raise ValueError('Unique increasing timestamps required')
def poses(text):
 header,body=rows(text)
 if len(header)!=17 or not all(key in header[index]for index,key in ((1,'p_RS_R_x'),(4,'q_RS_w'),(5,'q_RS_x'),(7,'q_RS_z'))):raise ValueError('Expected original17column position/wxyz/state CSV header')
 parsed=[]
 for r in body:
  if len(r)!=17:raise ValueError('Unexpected pose CSV columns')
  t=ns(r[0]);values=list(map(float,r[1:]))
  if not all(map(math.isfinite,values)):raise ValueError('Finite rover pose/state required')
  norm=math.sqrt(sum(x*x for x in values[3:7]))
  if abs(norm-1)>1e-4:raise ValueError('Quaternion print precision inconsistent with unit rotation')
  parsed.append(dict(timestamp_ns=t,position=values[:3],quaternion_wxyz=values[3:7],quaternion_norm=norm))
 ordered([r['timestamp_ns']for r in parsed]);return parsed

def image_times(text,suffix):
 _,body=rows(text);result=[]
 for r in body:
  if len(r)!=2:raise ValueError('Expected two-column timestamp/filename CSV')
  t=ns(r[0]);filename=r[1].strip()
  if filename!=str(t)+suffix:raise ValueError('Filename must equal exact timestamp and declared file extension')
  result.append(t)
 ordered(result);return result

def associate(image_ns,pose_ns,tolerance_ns=1000000):
 ordered(image_ns);ordered(pose_ns)
 if type(tolerance_ns)is not int or tolerance_ns<0:raise ValueError('Nonnegative integer timestamp tolerance required')
 ids=[];offsets=[]
 for t in image_ns:
  insertion=bisect_left(pose_ns,t);options=[i for i in(insertion-1,insertion)if 0<=i<len(pose_ns)]
  ranked=sorted((abs(pose_ns[i]-t),i)for i in options)
  if len(ranked)>1 and ranked[0][0]==ranked[1][0]:raise ValueError('Ambiguous nearest pose timestamp')
  distance,i=ranked[0]
  if distance>tolerance_ns:raise ValueError('Missing synchronized pose within declared tolerance')
  ids.append(i);offsets.append(pose_ns[i]-t)
 if len(set(ids))!=len(ids):raise ValueError('Pose reuse; unique frame associations required')
 return ids,offsets

def run(output):
 out=Path(output)
 if out.exists():raise FileExistsError('Preserve metadata diagnostics')
 w=Path(__file__).parent;root=w/'lusnar-metadata/development_metadata';provenance=w/'lusnar-metadata/scene1_recovered_metadata_provenance.json';p=json.loads(provenance.read_text())
 expected={m['member']:m for m in p['members']};texts={};hashes={}
 for name in('Moon_1/gt.txt','Moon_1/image0/color_timestamp.txt','Moon_1/image0/depth_timestamp.txt'):
  data=(root/name).read_bytes();h=hashlib.sha256(data).hexdigest();record=expected[name]
  if h!=record['sha256']or len(data)!=record['bytes']or not record['member_crc_verified']:raise ValueError('Recovered complete metadata member changed')
  hashes[name]=h;texts[name]=data.decode()
 pose=poses(texts['Moon_1/gt.txt']);rgb=image_times(texts['Moon_1/image0/color_timestamp.txt'],'.png');depth=image_times(texts['Moon_1/image0/depth_timestamp.txt'],'.pfm')
 if rgb!=depth:raise ValueError('Exact color/depth timestamp alignment absent')
 ids,offsets=associate(rgb,[p['timestamp_ns']for p in pose]);dt=[b-a for a,b in zip(rgb,rgb[1:])]
 result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),metadata_checks_passed=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),provenance_sha256=hashlib.sha256(provenance.read_bytes()).hexdigest(),member_sha256=hashes,development_scene=1,reserved_transfer_scenes=[7,8,9],reserved_scene_pixels_inspected=0,optical_or_depth_pixels_decoded=0,pose_rows=len(pose),color_timestamp_rows=len(rgb),depth_timestamp_rows=len(depth),unique_pose_associations=len(set(ids)),all_timestamps_parsed_without_floats=True,pose_quaternion_order_from_actual_header='wxyz',raw_pose_field_prefix='p_RS_R/q_RS: sensor or rover body; camera extrinsics not independently established',actual_depth_filename_extension='.pfm',color_depth_timestamps_identical=True,association_tolerance_ns=1000000,maximum_absolute_nearest_pose_offset_ns=max(map(abs,offsets)),median_color_frame_period_ns=statistics.median(dt),maximum_quaternion_norm_deviation=max(abs(r['quaternion_norm']-1)for r in pose),full_archive_checksum_verified=False,depth_units_and_axial_or_ray_convention_verified=False,pixel_centers_intrinsics_verified=False,camera_extrinsics_verified=False,scope='Actual recovered scene1 metadata format/synchronization check only. Does not validate camera poses, image projection, depth, reference accuracy, full archive, reserve readiness or transfer matching.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
