import cv2
import numpy as np

from audit_lusnar_temporal_v119 import assess, track
from lusnar_temporal_geometry_v119 import CONVENTIONS


def test_observed_flow_recovers_a_generated_shift_without_pose_or_depth():
    rng = np.random.default_rng(61119)
    a = cv2.GaussianBlur(rng.integers(0, 255, (160, 160), dtype=np.uint8), (3, 3), 0)
    b = cv2.warpAffine(a, np.array([[1., 0., 7.], [0., 1., -4.]]), (160, 160))
    mask = np.ones(a.shape, bool)
    source, target = track(a, b, mask, mask)
    assert len(source) > 20
    np.testing.assert_allclose(np.median(target-source, axis=0), [7., -4.], atol=.05)
    points, _ = track(np.zeros_like(a), b, mask, mask)
    assert points.shape == (0, 2)


def evidence(names, best, ambiguous=False):
    rows = []
    for g in range(5):
        for _ in range(6):
            scores = {name: dict(median_pixel_error=1. if name == best else 5.,
                                 median_absolute_log_depth_error=.02) for name in names}
            if ambiguous:
                scores[names[1]]["median_pixel_error"] = 1.05
            rows.append(dict(scope="convention_selection" if g in (0, 2, 4) else "convention_confirmation",
                             group=g, status="scored", hypotheses=scores))
    return rows


def test_ambiguous_geometry_is_never_promoted_or_given_an_oracle_pass():
    names = [c.name for c in CONVENTIONS]
    result = assess(evidence(names, names[0], ambiguous=True))
    assert result["status"] == "equivalent_or_ambiguous_conventions"
    assert result["qualified_common_hypotheses"] == names[:2]
    assert result["exact_pose_oracle_passed"] is False
    result = assess(evidence(names, names[0]))
    assert result["status"] == "unique_convention"
    assert result["qualified_common_hypotheses"] == names[:1]
    assert result["exact_pose_oracle_passed"] is False
