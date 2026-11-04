import numpy as np

from solar_aligned_scoring_v91 import align_visible_projection, load_aligned_scoring
from solar_evaluation_v70 import score_arrays


def test_arbitrary_projection_scatter_and_unchanged_metric_denominator(tmp_path):
    # Two visible positions with one unsupported depth and one outside image,
    # interleaved to catch compact-ordinal versus native-point ordering errors.
    depth = np.full((12, 12), 100.0)
    k = np.eye(3)
    pose = np.eye(4)
    points = np.array([[3., 3.], [6., 6.], [8., 8.], [11., 11.]])
    query_depth = depth.copy()
    query_depth[6, 6] = np.nan
    labels = tmp_path / "labels.npz"
    geometry = tmp_path / "geometry.npz"
    np.savez(labels, query_cam2world=pose, reference_xy=np.array([[0., 0.]]),
             query_xy=np.array([[0., 0.]]), visible=np.array([True]))
    np.savez(geometry, reference_depth=depth, query_depth=query_depth,
             reference_intrinsics=k, reference_cam2world=pose,
             query_intrinsics=k, query_cam2world=pose)
    case = dict(positive=True, scoring_labels=labels, scoring_geometry=geometry)
    truth, method = load_aligned_scoring(case, points)
    assert method == "common_axial_depth_projection_at_returned_points"
    np.testing.assert_array_equal(truth["visible"], [True, False, True, False])
    np.testing.assert_array_equal(truth["query_xy"][truth["visible"]], points[[0, 2]])
    assert np.isnan(truth["query_xy"][~truth["visible"]]).all()
    metrics = score_arrays(points, points, pose, truth, 100.0)
    assert metrics["correspondence"]["1"] == dict(correct=2, co_visible=2, returned=4)


def test_all_or_no_visible_rows_and_wrong_shapes():
    import pytest

    xy = np.array([[3., 4.], [5., 6.]])
    np.testing.assert_array_equal(align_visible_projection(xy, np.array([True, True])), xy)
    assert np.isnan(align_visible_projection(np.empty((0, 2)), np.array([False, False]))).all()
    assert align_visible_projection(np.empty((0, 2)), np.empty(0, dtype=bool)).shape == (0, 2)
    with pytest.raises(ValueError):
        align_visible_projection(xy, np.array([True, False]))
