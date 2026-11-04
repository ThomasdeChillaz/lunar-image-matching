"""Independent numerical controls for source meshing and ray intersections."""
import numpy as np
import pytest

from north_render_plan_v31 import body_point, plan
from north_terrain_pilot_v32 import grid_faces, local_vertices, pilot_plan
from verify_north_pilots_v32 import triangle_hits,local_surface_check


def test_source_assignments_are_only_four_predeclared_pilots():
    p,sites=pilot_plan()
    assert len(sites)==4 and all(s['split']=='renderer_cost_pilot'for s in sites)
    assert p['source_height_pixels_decoded']==0


def test_terrain_nodes_use_sphere_and_preserve_radius():
    site=next(s for s in plan()['sites']if s['component']=='np31-pilot-0')
    height=np.full((2049,2049),-2500.,np.float32);height[1024,1025]=-2475.
    offsets=[-1,0,1];v,origin,rows=local_vertices(height,site,offsets)
    recovered=v.astype(float)@rows+origin
    for r,dy in enumerate(offsets):
        for c,dx in enumerate(offsets):
            expected=body_point(site['x']+dx*10,site['y']-dy*10,float(height[1024+dy,1024+dx]))
            np.testing.assert_allclose(recovered[r,c],expected,atol=2e-6,rtol=0)
    # Scalar and vector spherical conversions can differ by float64 roundoff
    # before subtracting the lunar-radius body origin. Check all three local
    # coordinates within that numerical scale, not exact floating-point zero.
    roundoff_m=32*np.finfo(np.float64).eps*np.linalg.norm(origin)
    np.testing.assert_allclose(v[1,1],0.,atol=roundoff_m,rtol=0)
    # Spherical curvature must not be accidentally replaced by a flat DEM.
    assert v[0,0,2]<0


def test_missing_height_cannot_be_inpainted_or_skipped():
    site=plan()['sites'][0];height=np.zeros((2049,2049),np.float32);height[0,0]=np.nan
    with pytest.raises(ValueError):local_vertices(height,site,[-1,0,1])


def test_grid_faces_cover_every_cell_and_never_wrap_rows():
    f=grid_faces(3)
    assert f.shape==(8,3) and f.min()==0 and f.max()==8
    assert set(map(tuple,f[:4]))=={(0,1,3),(1,2,4),(3,4,6),(4,5,7)}


def test_independent_triangle_ray_uses_nearest_surface_and_rejects_behind():
    triangles=np.array([[[-2,-2,0],[2,-2,0],[0,2,0]],[[-2,-2,2],[2,-2,2],[0,2,2]]],float)
    assert triangle_hits(np.array([0.,0.,10.]),np.array([0.,0.,-1.]),triangles)==8
    assert np.isinf(triangle_hits(np.array([0.,0.,10.]),np.array([0.,0.,1.]),triangles))


def test_independent_local_surface_query_uses_correct_source_cell():
    site=next(s for s in plan()['sites']if s['component']=='np31-pilot-0')
    offsets=np.array([-2,-1,0,1,2]);v,origin,rows=local_vertices(np.zeros((2049,2049),np.float32),site,offsets)
    points=np.array([v[2,2],.2*v[1,1]+.3*v[1,2]+.5*v[2,1]])
    pose=np.eye(4);pose[:3,3]=[0,0,100]
    mesh=dict(vertices=v.reshape(-1,3),body_origin=origin,enu_rows=rows)
    errors=local_surface_check(points.astype(float),pose,mesh,site,offsets)
    assert np.max(errors)<1e-5
