# Independent gravity check for the SPICE-HL3 physical pose reference

Released inertial data corroborates the camera-frame discrepancy in our
physical laboratory diagnostic. The nominal composition puts gravity almost
entirely along the image-right direction in the image plane. The accelerometer
puts it almost along image-down. These measurements use the already exposed
Trajectory C captures and do not involve feature correspondences or a camera
pose estimate from images.

The source describes world gravity along negative Y, a 25° downward camera
mount and a tracker attached to that mount. The left camera uses the optical
basis of X right, Y down and Z forward. We transformed the normalized
quasi-static accelerometer vectors into that basis with the released
`T_c_I` and reversed their sign to obtain gravity. We used the frozen
marker-to-camera composition separately for the nominal direction.
[Dataset and frame conventions](https://arxiv.org/html/2506.22956v1#S7.SS4) ·
[Optical basis](https://docs.stereolabs.com/docs/development/zed-sdk/modules/positional-tracking/coordinate-frames) ·
[Accelerometer convention](https://github.com/ros-infrastructure/rep/blob/master/rep-0145.rst)

| Median across eligible camera times | Fast traversal | Slow traversal |
| --- | ---: | ---: |
| Captures with an eligible IMU window | 24/24 | 10/24 |
| IMU gravity roll from image-down | 4.44° | 2.87° |
| IMU gravity downward pitch | 28.18° | 28.93° |
| Nominal gravity roll from image-down | 88.87° | 88.60° |
| Nominal gravity downward pitch | 34.70° | 34.06° |
| Nominal gravity direction error against IMU | 70.10° | 71.06° |
| Error with the published marker rotation reversed | 96.24° | 95.41° |
| Error after a diagnostic quarter-turn about optical Z | 8.17° | 6.52° |

We used a ±50 ms window with at least three accelerometer readings, each
within 9.3–10.3 m/s² magnitude and 0.2 rad/s angular rate. Fourteen slow
captures lacked three archived IMU records even before these magnitude gates.
The marker orientation was interpolated from the original frozen clock
brackets. The earlier production transform and the independent interpolation
agree on all 34 nominal vectors to within 9 × 10⁻¹⁴ per component.

The quarter-turn is an exploratory axis diagnostic motivated by the previously
observed near-90° nominal roll. It leaves a residual tilt discrepancy. Gravity
does not constrain yaw about the gravity vector or determine the camera lever
arm. The audit supports retaining the 24 published pose rows as nominal
diagnostics and withholding a physical matcher ranking. A full correction
requires an independent optical calibration witness, such as confirmed
camera-to-marker axes or a geometry-verified stereo reference.
[Archived physical result](SPICE_HL3_GEOMETRY_RESULT_V101.md)

The two released IMU CSV members were fetched twice from the published ZIP
using verified byte ranges. Each retrieval used about 0.61 MB. The independent
retrieval regenerated the audit JSON byte for byte. No image data or
calibration and final lunar terrain was accessed.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python audit_spice_hl3_gravity_v117.py \
  --assets runs/spice-hl3-c-assets-v101 --imu runs/spice-hl3-c-imu-audit \
  --fetch-imu --output runs/spice-hl3-gravity-audit-v117.json
```

[Released dataset](https://doi.org/10.5281/zenodo.13970078) ·
[Archive member receipt](spice_hl3_imu_transport_v117.json) ·
[Per-capture measurements](spice_hl3_gravity_audit_v117.json)
