from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

spec = spec_from_file_location("independent_arithmetic_v137_test", Path(__file__).with_name("replay_saved.py"))
replay = module_from_spec(spec); spec.loader.exec_module(replay)


def geometry():
    k = np.array([[100., 0., 15.5], [0., 100., 15.5], [0., 0., 1.]])
    a = np.eye(4); b = np.eye(4); b[0, 3] = 5.
    return [np.full((32, 32), 100.), np.full((32, 32), 100.), k, a, k, b]


def test_independent_axial_projection_crop_and_missing_query_coverage():
    fields = geometry()
    p = np.array([[8., 8.], [24., 16.], [31., 31.]])
    target, visible = replay.visible_targets(p, fields)
    np.testing.assert_array_equal(visible, [True, True, False])
    np.testing.assert_allclose(target, [[3., 8.], [19., 16.]], atol=1e-12)
    # An unknown bilinear neighbour cannot provide visibility support.
    fields[1][16, 19] = np.nan
    _, visible = replay.visible_targets(p, fields)
    np.testing.assert_array_equal(visible, [True, False, False])


def test_independent_pose_and_correspondence_arithmetic_retains_failures():
    fields = geometry()
    row = dict(reference_xy=[[8., 8.], [24., 16.]], query_xy=[[3., 8.], [19., 16.]], pose=fields[-1].tolist())
    score = replay.arithmetic(row, True, fields)
    assert score["correct"] and score["strict"]
    assert score["correspondence"]["1"] == dict(correct=2, visible=2, returned=2)
    assert score["pose_error"]["translation_m"] == 0.
    row["pose"] = None
    score = replay.arithmetic(row, True, fields)
    assert not score["correct"] and not score["pose_returned"]
    row["pose"] = np.eye(4).tolist()
    score = replay.arithmetic(row, False)
    assert score["raw_nonoverlap"] and not score["correct"]
