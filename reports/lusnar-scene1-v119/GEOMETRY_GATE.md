# LuSNAR Scene 1 format and geometry preparation

This package prepares five complete stereo/depth triplets from the existing local Scene 1 ZIP prefix. Scene 1 is exposed development. Scenes 7–9 remain reserved. The frozen manifest authorizes extraction and structural header inspection only.

## Reproduce extraction

Run these commands from this directory with Python 3.9 or newer. Use new output names when replaying.

```sh
python3 prepare_lusnar_scene1_v117.py freeze extraction-freeze.json
python3 prepare_lusnar_scene1_v117.py extract extraction-freeze.json extracted
```

The script expects the original archive prefix and inventory in the sibling `lusnar-metadata` directory and the previous metadata parser one level up. It matches the exact frozen selection against that inventory and the verified metadata. It verifies the local header fields, bounded member byte ranges, stored/Deflate structure and CRC before writing each member. The receipt hashes every extracted member. The archive is incomplete and its overall checksum is unavailable.

All ten stereo PNG headers indicate 1024 × 1024, 8-bit RGBA, without interlacing. The five PFM headers indicate 1024 × 1024, single-channel little-endian float32 storage with unit header scale. PFM byte format alone leaves metric units, vertical row order and axial versus ray depth unresolved. The RGB and depth samples have not been decoded.

## Prospective geometry gate

The published rig uses 1024 × 1024 cameras, nominal focal length 610.17784 pixels and a 0.31-metre stereo baseline. It specifies rover axes forward, right, down and optical axes right, down, forward, with both cameras at the same pitch. Verify the mapping against Scene 1 before using it for camera pose scoring. The original CSV uses `p_RS_R` and `q_RS` fields, and its header establishes `wxyz` order but leaves the transform direction and translation origin to be verified. [LuSNAR paper, §III-C](https://arxiv.org/html/2407.06512v3#S3.SS3)

1. Freeze a separate run with the extraction receipt, exact source and runtime hashes, the eligibility rule, and fixed decision margins. Keep an initial cap of 300 calendar seconds. Denominators include every frozen case and report per-case ambiguity. No scene 7–9 file may be opened.
2. Inspect the depth values and verify orientation and metric scale using stereo observations. At off-axis static pixels with native left/right measurements, compare horizontal disparity and vertical residual. For a rectified pair, positive left-minus-right disparity is `focal_px × baseline_m / axial_m`. Compare the axial and ray hypotheses at a fixed grid of physical unit factors and report coverage as well as residuals. With ray depth, convert to axial by dividing by the norm of the unnormalized camera ray. Report supporting image regions and skip a choice when coverage or discrimination is insufficient. Ground-truth depth is never derived from brightness. The independent stereo observation only verifies how the released depth should be interpreted.
3. Freeze a small set of separately extracted Scene 1 temporal neighbors with compatible viewpoint before accessing their samples. The five quartile frames are spread across the path and do not by themselves provide reliable temporal overlap. Use classical tracks and independently stored later depth to test the pose-direction, camera lever-arm and pixel-center alternatives. Compare inverse, horizontal/vertical residuals, depth agreement and coverage. A project-and-recover pose test cannot distinguish a consistent but wrong convention.
4. After convention checks, build exact visible correspondences and apply common AP3P with the three-native-pixel threshold. Require the existing ≥95% pose-control gate across eligible development pairs, report denominator and failure modes, and verify original-pixel conversion. For uncropped 1024→512 resampling, use `p_512 = (p_1024 + 0.5) / 2 - 0.5` and halve focal length. Freeze Scene 1 decisions before any reserved-scene acquisition and interpretation.

Scene 1 can establish whether the published rig and stored depth form a usable development geometry pipeline. A held-out rover-viewpoint transfer test still requires stable access to scenes 7–9, a frozen case schedule, and a successful geometry check. The simulation release alone does not establish controlled illumination change or real lunar accuracy. [Official project](https://luoxubo.github.io/LuSNAR-dataset/) · [Official dataset files](https://huggingface.co/datasets/xuboluo2001/LuSNAR/tree/main)
