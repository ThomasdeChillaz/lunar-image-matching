"""Development-only split guards and fractional coordinate/visibility controls."""
import numpy as np
import pytest

from north_development_plan_v35 import (development_plan,prerequisite_gate,
    proposal_grid_native,quarter_turn_points,world_rays)
from north_bvh_supervision_v35 import mesh_correspondences
from north_render_plan_v31 import camera_intrinsics,camera_to_world
from cache_north_development_v35 import tile_ranges
from test_north_range_cache_v33 import header


def test_only_original_development_supports_can_be_cached():
    p,sites=development_plan()
    assert sum(s['split']=='training'for s in sites)==16
    assert sum(s['split']=='model_selection'for s in sites)==8
    assert {s['component']for s in sites}.isdisjoint({s['component']for s in p['sites']if s['split']in ('final_test','confidence_calibration','renderer_cost_pilot')})
    assert len(tile_ranges(header(),sites))>100
    changed=[dict(s)for s in sites];changed[0]['native_window']=[0,0,2049,2049]
    with pytest.raises(ValueError):tile_ranges(header(),changed)
    with pytest.raises(ValueError):tile_ranges(header(),sites+[next(s for s in p['sites']if s['split']=='final_test')])


def test_prerequisite_is_the_completed_actual_pilot():
    assert set(prerequisite_gate())=={'reports/north_terrain_pilot_audit_v34.json','reports/north_historical_exposure_v34.json'}


def test_rotated_coordinates_agree_with_actual_image_array():
    image=np.arange(512*512).reshape(512,512);p=np.array([[0,0],[25,60],[511,511]])
    for k in range(4):
        q=quarter_turn_points(p,k).astype(int)
        np.testing.assert_array_equal(np.rot90(image,k)[q[:,1],q[:,0]],image[p[:,1],p[:,0]])
        fractional=proposal_grid_native()
        np.testing.assert_array_equal(quarter_turn_points(quarter_turn_points(fractional,k),-k),fractional)
    assert np.all(proposal_grid_native()%1==.5)


class PlaneBVH:
    def ray_cast(self,origin,direction,maximum):
        origin,direction=np.asarray(origin),np.asarray(direction)
        if direction[2]>=0:return None,None,None,None
        distance=-origin[2]/direction[2]
        return origin+distance*direction,np.array([0,0,1]),0,distance


def test_fractional_mesh_labels_preserve_exact_known_plane_translation():
    a=np.asarray(camera_to_world([0.,0.,10.],[0.,0.,0.]));b=a.copy();b[0,3]=1.
    k=np.asarray(camera_intrinsics());p=proposal_grid_native()
    labels=mesh_correspondences(PlaneBVH(),a,b,k,p)
    assert labels['source_valid'].all()
    np.testing.assert_allclose(labels['world'][:,2],0,atol=1e-12)
    np.testing.assert_allclose(labels['target_points'],p+[-k[0,0]/10.,0.],atol=1e-12)
    np.testing.assert_array_equal(labels['visible'],labels['in_frame'])


def test_missing_source_intersection_is_not_a_visible_correspondence():
    pose=np.eye(4);pose[2,3]=10;k=np.asarray(camera_intrinsics())
    labels=mesh_correspondences(PlaneBVH(),pose,pose,k,proposal_grid_native())
    assert not labels['source_valid'].any()and not labels['visible'].any()


def test_pixel_center_ray_is_camera_forward():
    pose=np.asarray(camera_to_world([0.,0.,10.],[0.,0.,0.]));k=np.asarray(camera_intrinsics())
    np.testing.assert_allclose(world_rays([[255.5,255.5]],k,pose),[[0.,0.,-1.]])
