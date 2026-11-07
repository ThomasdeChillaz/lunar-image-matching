import math,unittest
from north_camera_variation_v59 import schedule,rolled_pose,protocol


def close(a,b):return abs(a-b)<1e-10


class CameraControls(unittest.TestCase):
 def test_optical_roll_preserves_forward_and_camera_center(self):
  a=rolled_pose([100,200,1000],[0,0,0],0);b=rolled_pose([100,200,1000],[0,0,0],47)
  self.assertEqual([r[2:]for r in a],[r[2:]for r in b])
  cols=[[b[i][j]for i in range(3)]for j in range(3)]
  for i in range(3):
   for j in range(3):self.assertTrue(close(sum(x*y for x,y in zip(cols[i],cols[j])),float(i==j)))
  x,y,z=cols;det=x[0]*(y[1]*z[2]-y[2]*z[1])-y[0]*(x[1]*z[2]-x[2]*z[1])+z[0]*(x[1]*y[2]-x[2]*y[1]);self.assertTrue(close(det,1))
 def test_component_specific_reproducible_draws_and_disjoint_roll_families(self):
  a=schedule(dict(component='a',split='train'),1200);self.assertEqual(a,schedule(dict(component='a',split='train'),1200))
  b=schedule(dict(component='b',split='train'),1200);self.assertNotEqual(a[0]['query']['cam2world'],b[0]['query']['cam2world'])
  for p in a:
   roll=abs(p['relative_roll_parameter_degrees'])
   self.assertTrue(35<=roll<=55 if p['family']=='unseen_roll'else roll<=20)
   self.assertTrue(close(p['query']['roll_degrees']-p['reference']['roll_degrees'],p['relative_roll_parameter_degrees']))
 def test_oblique_family_declares_separate_query_calibration(self):
  for p in schedule(dict(component='a',split='renderer_cost_pilot'),100):
   if p['family']!='occlusion_stress':continue
   self.assertEqual(p['reference']['fov_degrees'],60);self.assertEqual(p['query']['fov_degrees'],10)
   self.assertGreater(p['query']['intrinsics'][0][0],2*p['reference']['intrinsics'][0][0])
   forward=[p['query']['cam2world'][i][2]for i in range(3)];zenith=math.degrees(math.acos(-forward[2]));self.assertTrue(65<=zenith<=75)
 def test_forbids_reserve_and_missing_clearance(self):
  for split in ('final_test','confidence_calibration','unknown'):
   with self.assertRaises(ValueError):schedule(dict(component='a',split=split),0)
  with self.assertRaises(ValueError):protocol([dict(component='a',split='train')],{})
 def test_origin_projects_to_principal_point_when_target_is_origin(self):
  for roll in (-55,0,18,50):
   p=rolled_pose([100,-200,1200],[0,0,0],roll);delta=[-p[i][3]for i in range(3)];camera=[sum(delta[i]*p[i][j]for i in range(3))for j in range(3)]
   self.assertTrue(close(camera[0],0)and close(camera[1],0)and camera[2]>0)


if __name__=='__main__':unittest.main()
