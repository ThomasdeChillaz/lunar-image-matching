"""Independent analytic camera/conic controls, with no downloaded imagery."""
import math
import pytest
from cresent_geometry_v20 import annotations,project_conic,rotation_wxyz


def test_frontal_ellipse_has_analytic_pinhole_size_and_location():
    centre,shape,depth=project_conic([0,0,10,2,1,0],[0,0,20],[[1,0,0],[0,-1,0],[0,0,-1]],focal=100,principal=(50,50))
    assert centre==[50.,50.] and depth==10
    assert shape[0][0]==pytest.approx(400) and shape[1][1]==pytest.approx(100)


def test_oblique_ellipse_centre_is_not_projection_of_its_world_centre():
    angle=math.pi/6;c,s=math.cos(angle),math.sin(angle)
    rotation=[[c,0,-s],[0,-1,0],[-s,0,-c]]
    centre,_,_=project_conic([0,0,10,2,1,0],[0,0,20],rotation,focal=100,principal=(50,50))
    # Independently eliminate the ellipse-plane coordinates analytically.
    expected=50+100*s*c*(100+4)/(c*c*100-s*s*4)
    assert centre[0]==pytest.approx(expected)
    point_centre=50+100*math.tan(angle)
    assert abs(centre[0]-point_centre)>3


def test_published_omitted_rotation_header_cannot_shift_id_or_angle():
    text='selenographic crater coordinates: X, Y, Z, a_metres, b_metres, id\n1, 2, 10, 2, 1, .25, crater-1\n'
    assert annotations(text,'world')=={'crater-1':[1,2,10,2,1,.25]}
    with pytest.raises(ValueError):annotations(text+'1, 2, 10, 2, 1, .25, crater-1\n','world')
    with pytest.raises(ValueError):rotation_wxyz([2,0,0,0])


def test_resolved_camera_check_keeps_unresolved_separate_from_visibility():
    from cresent_geometry_v20 import FOCAL
    from cresent_resolved_camera_v21 import resolved_camera_check
    world={'large':[0,0,10,2,1,0],'tiny':[0,0,10,.001,.0005,0]}
    labels={'large':[512,512,FOCAL*.2,FOCAL*.1,0],'tiny':[512,512,FOCAL*.0001,FOCAL*.00005,0]}
    result=resolved_camera_check([0,0,20,0,1,0,0],world,labels)
    assert result['geometric_resolution_exclusions']=={'unresolved_ellipse':1}
    assert result['measurement']['eligible_ellipses']==1
    assert result['camera_convention_on_resolved_annotations_passed'] is False
    assert set(world)==set(labels)=={'large','tiny'}


def test_optional_pinned_source_metadata_replays_broad_failure_and_narrow_check():
    from pathlib import Path
    root=Path('data/cresent-metadata-v21')
    if not root.exists():pytest.skip('Optional CC BY-NC4 source metadata fixtures not installed')
    from verify_cresent_metadata_v21 import audit
    proof=audit(root)
    assert proof['replay_passed'] and proof['resolved_ellipses']==6940
    assert not proof['original_broad_annotation_gate_passed']
    assert not proof['reference_depth_geometry_gate_passed'] and not proof['fresh_matching_benchmark_completed']
    assert proof['image_pixels_decoded']==proof['matcher_inferences']==0
