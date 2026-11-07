import unittest
import numpy as np
from verify_north_camera_v59 import project,pose_errors,check_labels


def fixture():
 yy,xx=np.mgrid[:26,:26];p=np.column_stack([49.5+16*xx.ravel(),49.5+16*yy.ravel()]);k=np.array([[100.,0,255.5],[0,100.,255.5],[0,0,1.]])
 a=np.eye(4);a[2,3]=-10;b=a.copy();b[0,3]=.5
 world=np.column_stack([(p-255.5)/10,np.zeros(len(p))]);q,z=project(world,k,b)
 labels=dict(source_points=p,target_points=q,world=world,source_valid=np.ones(676,bool),in_frame=np.ones(676,bool),visible=np.ones(676,bool),geometrically_occluded=np.zeros(676,bool),visibility_uncertain=np.zeros(676,bool),target_hit_distance_m=np.linalg.norm(world-b[:3,3],axis=1))
 return labels,a,b,k


class CameraAuditControls(unittest.TestCase):
 def test_exact_plane_projection_and_visibility(self):
  labels,a,b,k=fixture();self.assertEqual(check_labels(labels,a,b,k),1.)
 def test_corrupted_target_coordinates_are_detected(self):
  labels,a,b,k=fixture();labels['target_points'][0,0]+=.01
  with self.assertRaisesRegex(ValueError,'target projection'):check_labels(labels,a,b,k)
 def test_visibility_cannot_override_out_of_frame(self):
  labels,a,b,k=fixture();labels['in_frame'][0]=False
  with self.assertRaisesRegex(ValueError,'Visibility'):check_labels(labels,a,b,k)
 def test_changed_source_grid_is_detected(self):
  labels,a,b,k=fixture();labels['source_points'][0]+=.1
  with self.assertRaisesRegex(ValueError,'proposal grid'):check_labels(labels,a,b,k)
 def test_source_ray_inconsistency_is_detected(self):
  labels,a,b,k=fixture();labels['world'][0,0]+=.1
  labels['target_hit_distance_m']=np.linalg.norm(labels['world']-b[:3,3],axis=1)
  with self.assertRaisesRegex(ValueError,'source ray projection'):check_labels(labels,a,b,k)
 def test_separate_camera_calibration_in_label_audit(self):
  labels,a,b,k=fixture();kb=k.copy();kb[0,0]=kb[1,1]=200
  labels['target_points'],_=project(labels['world'],kb,b)
  inside=((labels['target_points']>=0)&(labels['target_points']<=511)).all(1)
  labels['in_frame']=inside;labels['visible']=inside;labels['target_hit_distance_m'][~inside]=np.nan
  self.assertEqual(check_labels(labels,a,b,k,kb),float(inside.mean()))
  with self.assertRaises(ValueError):check_labels(labels,a,b,k)
 def test_nonfinite_labels_are_detected(self):
  labels,a,b,k=fixture();labels['world'][0,0]=np.nan
  with self.assertRaisesRegex(ValueError,'Nonfinite'):check_labels(labels,a,b,k)
 def test_analytical_pose_error_and_missing_pose(self):
  a=np.eye(4);b=np.eye(4);b[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];b[:3,3]=[3,4,0]
  r=pose_errors(a,b);self.assertAlmostEqual(r['rotation_deg'],90);self.assertEqual(r['translation_m'],5);self.assertIsNone(pose_errors(None,b))


if __name__=='__main__':unittest.main()
