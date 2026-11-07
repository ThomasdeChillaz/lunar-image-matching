import unittest
import numpy as np
from north_camera_correspondences_v59 import mesh_correspondences
from north_camera_visibility_v59 import first_mesh_hit,visibility_check


class PlaneBVH:
 def __init__(self,occlude=False):self.occlude=occlude
 def ray_cast(self,origin,ray,maximum):
  origin,ray=np.asarray(origin),np.asarray(ray);height=-5. if self.occlude and origin[0]>.25 else 0.
  distance=(height-origin[2])/ray[2]
  if distance<=0:return None,None,None,None
  return origin+distance*ray,None,0,float(distance)


class CameraVisibilityControls(unittest.TestCase):
 def test_separate_intrinsics_are_used_for_target_projection(self):
  a=np.eye(4);a[2,3]=-10;b=a.copy();b[0,3]=.5;ka=np.array([[100.,0,255.5],[0,100.,255.5],[0,0,1.]]);kb=ka.copy();kb[0,0]=kb[1,1]=200
  p=np.array([[255.5,255.5],[265.5,265.5]]);z=mesh_correspondences(PlaneBVH(),a,b,ka,kb,p)
  np.testing.assert_allclose(z['target_points'],[[245.5,255.5],[265.5,275.5]],atol=1e-10);self.assertTrue(z['visible'].all())
 def test_closer_first_hit_is_geometric_occlusion(self):
  a=np.eye(4);a[2,3]=-10;b=a.copy();b[0,3]=.5;k=np.array([[100.,0,255.5],[0,100.,255.5],[0,0,1.]])
  z=mesh_correspondences(PlaneBVH(True),a,b,k,k,np.array([[255.5,255.5]]));self.assertTrue(z['in_frame'][0]);self.assertFalse(z['visible'][0])
  self.assertTrue(z['geometrically_occluded'][0]);self.assertFalse(z['visibility_uncertain'][0])
 def test_missing_target_intersection_is_not_called_occlusion(self):
  class MissingTarget(PlaneBVH):
   def ray_cast(self,origin,ray,maximum):
    return (None,None,None,None)if origin[0]>.25 else super().ray_cast(origin,ray,maximum)
  a=np.eye(4);a[2,3]=-10;b=a.copy();b[0,3]=.5;k=np.array([[100.,0,255.5],[0,100.,255.5],[0,0,1.]])
  z=mesh_correspondences(MissingTarget(),a,b,k,k,np.array([[255.5,255.5]]))
  self.assertFalse(z['geometrically_occluded'][0]);self.assertTrue(z['visibility_uncertain'][0])
 def test_independent_full_mesh_uses_nearest_surface_across_chunks(self):
  vertices=np.array([[-2,-2,0],[2,-2,0],[0,2,0],[-2,-2,5],[2,-2,5],[0,2,5]],float);faces=np.array([[0,1,2],[3,4,5]])
  self.assertEqual(first_mesh_hit([0,0,10],[0,0,-1],vertices,faces,1),5)
  self.assertFalse(visibility_check([0,0,10],[0,0,0],vertices,faces)['visible']);self.assertTrue(visibility_check([0,0,10],[0,0,5],vertices,faces)['visible'])
 def test_invalid_nonunit_ray_is_rejected(self):
  with self.assertRaises(ValueError):first_mesh_hit([0,0,10],[0,0,-2],np.zeros((3,3)),np.array([[0,1,2]]))


if __name__=='__main__':unittest.main()
