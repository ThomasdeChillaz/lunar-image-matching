# Recorded camera frames for the SPICE-HL3 physical pose audit

The original Trajectory C recordings confirm the optical and inertial
portion of the frame chain used in the gravity audit. They do not establish
the mocap-marker to camera-optical transform needed to score the lab poses.

The static TF record names the left camera body as parent and its optical
frame as child. It has zero translation and quaternion `(0.5, −0.5, 0.5, −0.5)` in
xyzw order. The IMU messages use `leo01/zed2_imu_link`. Composing the recorded
body and IMU transforms agrees with the released `T_c_I` to 0.51° and
0.23 mm. The slow range contains both recorded static TF messages and no
marker-to-optical static edge. The fast range covers one of two static
messages. These prefixes do not cover the complete dynamic TF history.
[Released recordings](https://zenodo.org/records/13970078) ·
[ZED frame definitions](https://docs.stereolabs.com/docs/integrations/ros-2/zed-stereo-node)

At one exactly simultaneous slow recording timestamp, the grayscale left
and right `CameraInfo` messages identify the optical frames and 640 × 360
images. Their projection gives focal length 263.9241 px and baseline
0.1198271 m. The focal length is 0.626% larger than the standalone
calibration JSON. This example establishes recorded stereo geometry for
that timestamp. A stereo validation must bind projection matrices to
the selected captures before treating its reconstructed scale as known.
[ROS CameraInfo fields](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/CameraInfo.msg)

The author's example ORB-SLAM configuration describes the released marker
matrix as camera-to-marker. The data descriptor describes its rigid
calibration in the global-to-camera direction. Reversing the matrix alone
made the gravity agreement worse in the earlier audit. The evidence
supports a dedicated calibration check before replacing the reference or
ranking matchers. A held-out stereo or surveyed-target check should
characterize the marker transform, clock uncertainty and depth.
[Author configuration](https://github.com/drodriguezSRL/ORBSLAM3_implementation/blob/main/config/spice-hl3.yaml) ·
[Data descriptor](https://doi.org/10.1038/s41597-026-06668-8) ·
[Gravity audit](SPICE_HL3_GRAVITY_AUDIT_V117.md)

The audit independently replayed four HTTP byte ranges totaling 6,295,552
bytes. The receipt preserves range and message hashes and the verified YAML
and MCAP chunk checks. The full archive digest was outside this range audit.
The procedure inspected metadata and calibration messages without decoding
image samples.

```sh
python3 -B reports/replay_spice_hl3_ros_frame_witness_v117.py \
  | diff -u reports/spice_hl3_ros_frame_witness_v117.json -
```

[Recorded frame receipt](spice_hl3_ros_frame_witness_v117.json) ·
[MCAP format](https://mcap.dev/spec)
