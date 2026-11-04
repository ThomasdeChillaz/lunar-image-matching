# Physical lunar-analog geometry transfer preparation

The SPICE-HL3 LunaLab release contains camera images and externally tracked rover motion. Trajectory C is a sun-facing traversal recorded in an indoor lunar analog. This preparation selects an independently tracked geometry diagnostic. It complements the simulated lunar benchmark and leaves the reserved LuSNAR, POLAR, orbital, calibration and final evaluations open. [Data descriptor](https://doi.org/10.1038/s41597-026-06668-8) · [Author archive and CC BY 4.0 license](https://doi.org/10.5281/zenodo.13970078)

## Source records and eligibility

| Track | Actual source audit | Disposition |
|---|---|---|
| Trajectory A, repeated illumination, FLIR | Each selected exposure/condition folder contains 100 image records. The released A motion CSV contains 99 records. Of those, 73 orientation vector norms fall outside 0.98–1.02, and the camera timestamps do not align with the CSV under the released −0.340 s correction. | Excluded from pose scoring. |
| Trajectory C, fast SPAD | All 426 motion records have unit quaternions within the audit tolerance. The 17,681 optical metadata rows carry only 41 distinct timestamps at whole-second granularity. | Excluded from frame-level pose scoring. |
| Trajectory C, fast and slow left ZED2 | The calibrated rectified images are 640 × 360. Both streams supply native grayscale images, camera timestamps and independent motion data. 96 fast and 263 slow images have an exact published camera timestamp and a motion bracket no longer than 0.100 s. | Prepared for a separate geometry diagnostic. |

The ZED2 audit sorts motion-capture timestamps before bracketing and retains original CSV row identities. Exact decimal camera time joins use the published image basename and archived camera metadata, with the released zero ZED2 offset. Frame ordinals differ between the two camera records and are excluded as a join key. Camera pose is composed using the released marker-to-camera calibration. [Calibration convention](https://orbilu.uni.lu/bitstream/10993/60694/1/Joint%20Spatial-Temporal%20Calibration%20for%20Camera%20and%20Global%20Pose%20Sensor.pdf#page=3)

## Fixed physical cohort

The metadata-only selection yielded 24 within-traversal pairs with 48 distinct camera frames. It selects six pairs per marker-motion band in each traversal, at 0.10–0.22 m and 0.22–0.44 m. Both traversals cover one lab site. The selection uses temporal coverage and motion metadata before any matcher output or image-array decoding. Motion labels are for eligibility and scoring, separate from image matching.

Exact HTTP range reads retrieved only the 48 selected native PNG members. Original ZIP CRCs, PNG dimensions and grayscale headers were checked. The selected optical acquisition transferred 7,493,539 bytes. The prepared manifest binds the file hashes, calibration, original camera metadata, motion data and exact case identities. No optical matcher or pose inference has run.

The eventual diagnostic uses native 640 × 360 camera pixels and relative pose or epipolar correspondence error. Its results will be reported separately from the 512-pixel absolute map-relative lunar benchmark. This physical cohort tests geometry transfer on sun-facing traversals. Trajectory A would need separately qualified frame-to-pose labels to support a changed-light pose test.

## Reproduce the selection

Run these commands from the repository root with a fresh local directory.

```sh
python spice_hl3_released_archive_v100.py --metadata-output runs/spice-hl3-c-reproduction
python prepare_spice_geometry_cases.py runs/spice-hl3-c-reproduction runs/spice-hl3-c-reproduction/trajectory_C_metadata_cases_v2.json
python extract_spice_geometry_frames.py runs/spice-hl3-c-reproduction
```

A second metadata download reproduced the full selected-member inventory, all six calibration and motion metadata files, and the case schedule byte for byte. The case schedule SHA-256 is `8e5f38aa087a147dc58979c5f2abee2bcf6580bb2bf31a831cab6929e69034f8`. The acquired prepared manifest SHA-256 is `d4d9d151be40a80e8fbc628138717141a6279605b7125fd03512731f2454a435`. Running the diagnostic is a separate source-published and compute-gated step.
