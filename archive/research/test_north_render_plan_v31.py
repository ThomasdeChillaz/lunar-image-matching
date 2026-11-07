"""Analytic prospective site/camera/frame checks; no source pixels or renderer."""
import math
import unittest

from north_render_plan_v31 import (RADIUS,body_point,camera_intrinsics,camera_to_world,
    enu_frame,inverse_north_stereo,local_point,plan,support_overlap)


class NorthernPlanControls(unittest.TestCase):
    def test_complete_separate_supports_and_stratified_counts(self):
        p=plan();self.assertEqual(p['component_counts'],dict(confidence_calibration=4,final_test=12,model_selection=8,renderer_cost_pilot=4,training=16))
        sites=p['sites'];self.assertEqual(len({s['component']for s in sites}),44)
        for i,a in enumerate(sites):
            self.assertTrue(all(not support_overlap(a,b)for b in sites[i+1:]))
            r,c,h,w=a['native_window'];self.assertTrue(0<=r and 0<=c and r+h<=43280 and c+w<=43280)
        for sector in ['projected-strip-'+str(i)for i in range(4)]:
            selected=[s for s in sites if s['sector']==sector]
            self.assertEqual(sum(s['split']=='final_test'for s in selected),3)
        self.assertEqual(p,plan());self.assertEqual(len(p['mesh_native_offset_indices']),897)

    def test_independent_stereographic_cardinal_directions(self):
        expected=math.pi/2-2*math.atan(100000/(2*RADIUS))
        for x,y,lon in [(0,-100000,0),(100000,0,math.pi/2),(0,100000,math.pi),(-100000,0,-math.pi/2)]:
            lat,l=inverse_north_stereo(x,y);self.assertAlmostEqual(lat,expected);self.assertAlmostEqual(l,lon)
        self.assertEqual(inverse_north_stereo(0,0),(math.pi/2,0.))
        with self.assertRaises(ValueError):inverse_north_stereo(float('nan'),0)

    def test_body_radius_and_local_up(self):
        point=body_point(100000,50000,2000);self.assertAlmostEqual(math.sqrt(sum(v*v for v in point)),RADIUS+2000,places=7)
        origin,frame=enu_frame(100000,50000,2000)
        self.assertTrue(all(abs(x)<1e-9 for x in local_point(100000,50000,2000,origin,frame)))
        q=local_point(100000,50000,2003,origin,frame)
        self.assertAlmostEqual(q[0],0.,places=8);self.assertAlmostEqual(q[1],0.,places=8);self.assertAlmostEqual(q[2],3.,places=8)
        for i,a in enumerate(frame):
            for j,b in enumerate(frame):self.assertAlmostEqual(sum(x*y for x,y in zip(a,b)),float(i==j))

    def test_camera_front_right_down_and_half_pixel(self):
        p=camera_to_world([0.,0.,10.],[0.,0.,0.]);self.assertEqual(p,[[1.,0.,0.,0.],[0.,-1.,0.,0.],[0.,0.,-1.,10.],[0.,0.,0.,1.]])
        k=camera_intrinsics(512);self.assertEqual(k[0][2],255.5);self.assertAlmostEqual(k[0][0],256*math.sqrt(3))
        for position,target in [([0,0,0],[0,0,0]),([0,0,0],[0,1,0])]:
            with self.assertRaises(ValueError):camera_to_world(position,target)


class IndependentTriangleControls(unittest.TestCase):
    def test_direct_floor_and_solid_box_hits(self):
        import numpy as np
        from verify_blender_camera_control_v31 import trace
        origin=np.array([0.,0.,10.]);targets=np.array([[-2.,-2.,0.],[1.5,.4,2.]])
        directions=targets-origin;ranges=np.linalg.norm(directions,axis=1);directions/=ranges[:,None]
        np.testing.assert_allclose(trace(origin,directions),ranges,atol=1e-12)
        self.assertTrue(np.isinf(trace(origin,np.array([[0.,0.,1.]])))[0])

    def test_behind_box_first_hit_not_back_face(self):
        import numpy as np
        from verify_blender_camera_control_v31 import trace
        origin=np.array([1.5,.4,10.]);down=np.array([[0.,0.,-1.]])
        self.assertEqual(float(trace(origin,down)[0]),8.)
