import cv2
import numpy as np
import pytest

from reference_frame import native, solve
from smoke_v126 import generated


@pytest.mark.parametrize("multidepth", [False, True])
def test_frame_control_recovers_approximate_camera_handoff(multidepth):
    cv2.setNumThreads(1)
    args, valid, truth = generated(multidepth)
    ref, depth, k, old_pose, query, qk, _ = args
    pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = 2000.
    expected = pose@np.linalg.inv(old_pose)@truth
    # A nonzero, geometry-only perturbation exercises the fractional raster
    # handoff, rather than initializing the patches with the exact query camera.
    proposed = expected.copy()
    proposed[:3, 3] += [.04, -.03, .02]
    rows = native.measure(ref, query, depth, k, pose, qk, proposed[None],
                          reference_valid=np.ones_like(valid))
    for row in rows.values():
        result = solve(row["reference_xy"], row["query_xy"], depth, k, pose, qk)
        assert result["pose"] is not None
        assert np.linalg.norm(np.asarray(result["pose"])[:3, 3]-expected[:3, 3]) < 2.


def test_reference_frame_result_is_invariant_to_body_frame_rotation_and_translation():
    k = np.array([[440., 0., 255.5], [0., 440., 255.5], [0., 0., 1.]])
    depth = np.full((512, 512), 1100.)
    y, x = np.mgrid[72:440:73, 72:440:73]
    p = np.c_[x.ravel(), y.ravel()]
    q = p.astype(float)-[8.1, -.3]
    base = solve(p, q, depth, k, np.eye(4), k)
    assert base["pose"] is not None
    for vector in ([np.pi, 0., 0.], [.5, -.7, .9], [-2., 1., .8]):
        frame = np.eye(4)
        frame[:3, :3] = cv2.Rodrigues(np.asarray(vector))[0]
        frame[:3, 3] = [1700000., -23000., 900000.]
        changed = solve(p, q, depth, k, frame, k)
        assert changed["pose"] is not None
        np.testing.assert_allclose(changed["pose"], frame@np.asarray(base["pose"]), atol=2e-7)


def test_invalid_and_insufficient_correspondences_still_return_null():
    k = np.eye(3)
    result = solve([[1., 1.]]*5, [[1., 1.]]*5, np.ones((512, 512)), k, np.eye(4), k)
    assert result["pose"] is None and result["estimator"]["inliers"] == 0
