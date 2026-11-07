"""Cross-runtime schedule verification; always render the stored pose matrices.

Python3.12 changed float summation, which can perturb a regenerated rotation
basis by a few ulps. Only the3x3 basis allows1e-12 absolute replay tolerance.
Positions, targets, random draws and every other protocol value stay exact.
"""
import math

EXTRA_FIELDS={'acquired_summary_sha256','camera_math_source_sha256','mesh_sha256',
 'metadata_controls_passed','publication_pending','actual_geometry_controls_pending'}


def verify_schedule(frozen,expected):
 if set(frozen)-set(expected)-EXTRA_FIELDS or not set(expected)<=set(frozen):raise ValueError('Camera schedule fields differ')
 for key,value in expected.items():
  if key!='pairs'and frozen[key]!=value:raise ValueError('Camera schedule field differs:'+key)
 if len(frozen['pairs'])!=len(expected['pairs']):raise ValueError('Camera pair count differs')
 largest=0.
 for actual,wanted in zip(frozen['pairs'],expected['pairs']):
  if set(actual)!=set(wanted):raise ValueError('Camera pair fields differ')
  for key,value in wanted.items():
   if key not in ('reference','query')and actual[key]!=value:raise ValueError('Physical camera draw differs:'+key)
  for role in ('reference','query'):
   a,b=actual[role],wanted[role]
   if set(a)!=set(b)or any(a[k]!=v for k,v in b.items()if k!='cam2world'):raise ValueError('Camera position/target/roll or calibration differs')
   x,y=a['cam2world'],b['cam2world']
   if len(x)!=4 or any(len(row)!=4 for row in x)or x[3]!=y[3]or any(x[i][3]!=y[i][3]for i in range(3)):
    raise ValueError('Camera translation/homogeneous row differs')
   for i in range(3):
    for j in range(3):
     delta=abs(x[i][j]-y[i][j])
     if not math.isfinite(delta)or delta>1e-12:raise ValueError('Camera rotation basis differs beyond replay tolerance')
     largest=max(largest,delta)
 return dict(passed=True,rotation_basis_tolerance=1e-12,maximum_rotation_basis_delta=largest,
  frozen_pose_matrices_used_for_rendering=True,all_other_camera_draws_exact=True)
