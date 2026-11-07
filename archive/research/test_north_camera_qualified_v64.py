import ast
from pathlib import Path
import numpy as np
import pytest
from verify_north_camera_qualified_v64 import qualify_projected_pixels


def test_pixel_tolerance_stays_strict_and_any_failed_pixel_quarantines_view():
    p=np.zeros((100,2));z=np.ones(100)
    assert qualify_projected_pixels(p,p,z)['qualified']
    q=p.copy();q[73,0]=.020001
    r=qualify_projected_pixels(q,p,z)
    assert not r['qualified']and r['pixels_over_tolerance']==1 and r['tolerance_native_px']==.02
    q[73,0]=.019999;assert qualify_projected_pixels(q,p,z)['qualified']


def test_nonfinite_and_behind_camera_are_not_silently_dropped():
    p=np.zeros((2,2));z=np.ones(2);z[1]=-1
    assert not qualify_projected_pixels(p,p,z)['qualified']
    p[1,0]=np.nan;r=qualify_projected_pixels(p,np.zeros_like(p),np.ones(2))
    assert not r['qualified']and r['maximum_pixel_error']is None


def test_geometry_oracle_and_visibility_math_are_identical_to_original():
    old=ast.parse(Path('verify_north_camera_v59.py').read_text())
    new=ast.parse(Path(__file__).with_name('verify_north_camera_qualified_v64.py').read_text())
    def node(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
    for name in ('project','pose_errors','check_labels'):
        assert ast.dump(node(old,name),include_attributes=False)==ast.dump(node(new,name),include_attributes=False)
    assert 'Oracle poses are geometry QA'in Path('verify_north_camera_v59.py').read_text()


def test_malformed_pixel_schema_rejected():
    with pytest.raises(ValueError):qualify_projected_pixels(np.zeros((3,2)),np.zeros((2,2)),np.ones(2))
