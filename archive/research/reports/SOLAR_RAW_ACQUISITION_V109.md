# Repeatable terrain-scattered light in a dark lunar simulation

The earlier low-light pose experiment showed that shared 8-bit dithering could create convincing false alignments. Turning the dither off eliminated all three false poses in the selected dark cases. None of those dark poses was recovered. The present study separates renderer sampling noise from terrain-scattered signal before fitting a camera model or running another matcher. [Original pose experiment](SOLAR_CLEAN_RADIANCE_RESULTS_V106.md)

The prospective study kept the exact eight native camera views, terrain meshes, archived Sun vectors and constant diffuse material. It produced eight legacy weak-uniform-world controls and twenty black-world renders. The black-world condition removes uniform background light and measures direct and terrain-scattered light separately in float32 scene-linear EXRs. Independent seeds produce two renders per view at 64 samples per pixel, plus two at 256 for each view in the sparse-tail case. The timing pilot established a conservative 744-second admission estimate for the 28 renders. The study finished in 420.63 seconds under the frozen 900-second cap. [Timing protocol](solar-raw-acquisition-v109-timing.json) · [Study protocol](solar-raw-acquisition-v109-study.json)

The timing controls verified zero radiance with the Sun off, positive reflected signal with the direct ray blocked, and proportional response when the Sun energy was halved. All eight legacy RGB8 captures reproduced the archived pixels exactly. An independent OpenCV replay checked every EXR, all four analytic controls, archived camera geometry and both seed measurements. Maximum archive reprojection residual was 0.000079 native pixels. [Timing receipt](solar_raw_acquisition_timing_receipt_v109.json) · [Study receipt](solar_raw_acquisition_study_receipt_v109.json) · [Independent replay](solar_raw_acquisition_replay_v109.json)

At a 32-pixel block scale, repeatable spatial RMS is the square root of nonnegative cross-seed spatial covariance. The single-render seed residual is the RMS difference between independent renders divided by square root of two. Values are relative scene-linear units.

| Scene | Samples | Direct RMS, reference / query | Scattered-light RMS, reference / query | Scattered signal / seed residual, reference / query |
|---|---:|---:|---:|---:|
| u013 bright | 64 | 0.0116 / 0.0113 | 1.51e-5 / 2.52e-5 | 7.58 / 11.65 |
| u025 near-uniform | 64 | 0 / 0 | 0 / 0 | 0 / 0 |
| u027 near-uniform | 64 | 0 / 0 | 5.06e-9 / 1.65e-8 | 0.12 / 0.34 |
| u061 sparse tail | 64 | 0 / 0 | 2.08e-5 / 1.84e-5 | 5.20 / 5.60 |
| u061 sparse tail | 256 | 0 / 0 | 2.10e-5 / 1.82e-5 | 12.08 / 11.04 |

The two high-sample u061 means agree with the independent 64-sample ensemble to a block RMS difference of 3.13e-6 in the reference and 2.92e-6 in the query. Fine spatial support is less stable. The share of pixels with positive scattered radiance in both seeds rose from 1.37% to 15.71% in the reference and from 1.18% to 13.87% in the query. Positive support has no intensity threshold and should not be interpreted as camera detectability.

This is a repeatable within-camera signal in one simulated case. A second camera seeing similar aggregate contrast does not establish that the contrast is at the same terrain locations. The finite mesh can omit surrounding scatterers. The model also uses constant diffuse albedo and lacks a calibrated conversion to photoelectrons. Neither a camera noise floor nor pose recovery was measured. ShadowCam is a real example of secondary illumination imaging and the associated saturation tradeoff, with a separately calibrated instrument response. [ShadowCam calibration](https://doi.org/10.5140/JASS.2023.40.4.173)

Extract the released input and output bundle into a clean checkout with the Git history in place. The ZIP contains the original relative paths and a per-file SHA-256 manifest. Its SHA-256 is `365f416450c8beff4a93db1ed076c6f2c8827f64203d3997a2f20b1ab83f67dc`. The replay writes new result files and does not re-render the scenes. [Download the v109 bundle](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-raw-radiance-v109)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  python replay_raw_acquisition_v109.py \
    --output runs/solar_raw_acquisition_independent_repeat.json \
    --table runs/solar_raw_acquisition_independent_repeat.md
```
