from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reports/solar-parallax-v126"))

from pipeline import common_images, run
from smoke_v126 import generated


def test_every_method_sees_identical_native_encoding_and_keeps_constant_images_safe():
    a = np.arange(512 * 512, dtype=float).reshape(512, 512)
    x, y, record = common_images(a, 5*a+300, np.ones(a.shape, bool))
    assert x.shape == y.shape == a.shape and x.dtype == y.dtype == np.uint8
    assert np.array_equal(x, y)
    blank, _, _ = common_images(np.ones_like(a), np.ones_like(a), np.ones(a.shape, bool))
    assert not blank.any()


def test_native_motion_is_recovered_by_corrected_measurement_and_direct():
    cv2.setNumThreads(1)
    args, valid, truth = generated(False)
    reference, depth, k, pose, query, qk, _ = args
    image, _, _ = common_images(reference, query, np.ones_like(valid))
    result = run(reference, query, depth, k, pose, qk, pose[None], [image],
                 reference_valid=np.ones_like(valid))
    for name in ("pooled_uniform", "pooled_parallax", "direct"):
        recovered = result["methods"][name]["pose"]
        assert recovered is not None
        assert np.linalg.norm(np.asarray(recovered)[:3, 3]-truth[:3, 3]) < .5
