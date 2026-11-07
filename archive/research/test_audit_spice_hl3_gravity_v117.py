from decimal import Decimal

import numpy as np

from audit_spice_hl3_gravity_v117 import marker_rotation, optical_gravity


def test_marker_rotation_interpolates_antipodal_quaternions_as_the_same_pose():
    frame = {"marker_bracket_rows": [0, 1], "timestamp_s": "1.05"}
    row = {"Time": "1.00", "OrientationX": 0, "OrientationY": 0,
           "OrientationZ": 0.5 ** 0.5, "OrientationW": 0.5 ** 0.5}
    other = {name: -value for name, value in row.items() if name != "Time"}
    other["Time"] = "1.10"
    expected = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    np.testing.assert_allclose(marker_rotation([row, other], frame), expected, atol=1e-14)


def test_gravity_reverses_proper_acceleration_and_preserves_missing_windows():
    times = list(map(Decimal, ["0.950", "1.010", "1.040", "1.051"]))
    forces = [np.array([0, 0, 1])] * 3 + [np.array([1, 0, 0])]
    imu_to_optical = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    direction, count = optical_gravity(Decimal("1.000"), times, forces, imu_to_optical)
    assert count == 3
    np.testing.assert_array_equal(direction, [0, 1, 0])
    assert optical_gravity(Decimal("2.000"), times, forces, imu_to_optical) == (None, 0)
