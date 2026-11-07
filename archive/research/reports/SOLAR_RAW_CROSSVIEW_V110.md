# Coarse low-light structure repeats across the u061 views

The v109 renderer audit found that averaging u061 over 32 × 32 pixels suppressed much of the random path-sampling variation within each camera. This follow-up asks whether those coarse patterns also agree between cameras at the same projected terrain locations.

The protocol was frozen before decoding the sixteen scene-linear float32 black-space Combined EXRs. u061 uses two independent renders per camera at 256 samples per pixel. u013, u025 and u027 each use two independent renders per camera at 64 samples per pixel. All four cases are previously exposed model-selection positives rendered at the physical query illumination. The prior geometry diagnostic supplies conservative common support from the exact camera and depth archives, including reciprocal depth checks and one reference sample per aligned query pixel.

For each fixed nonoverlapping reference block, the mean uses terrain pixels shared by the aligned and four ±32-pixel query shifts. Blocks with fewer than half their native pixels in common support are excluded. Retained blocks receive equal weight. The predeclared comparison asks whether aligned Pearson correlation exceeds each shifted correlation for all four pairings of the two reference and two query renders. Those four pairings reuse two renders per camera and provide a stability check, without a terrain-independence or significance claim.

| Model-selection case | Samples per pixel | Supported 32-pixel blocks | Aligned seed-pair correlation | Seed-average aligned / strongest shift | Predeclared comparison |
|---|---:|---:|---:|---:|---|
| u061 | 256 | 169 | 0.9856 to 0.9871 | 0.9929 / 0.7948 | All four pairs above every shift |
| u013, bright control | 64 | 205 | approximately 1.0000 for all four | 1.0000 / 0.7439 | All four pairs above every shift |
| u025, near-uniform control | 64 | 174 | three undefined, one −0.0058 | −0.0058 / not defined for all shifts | Indeterminate |
| u027, near-uniform control | 64 | 202 | −0.0406 to 0.1308 | 0.0433 / 0.2031 | No consistent alignment advantage |

In u061, the weakest of the four 32-pixel improvements over its strongest shift was 0.188. The aligned correlation remained high on 16-pixel blocks, from 0.949 to 0.956, and fell to 0.051–0.052 at native pixels. At 32 pixels, seed-average spatial RMS was about `1.53e-5` relative scene-linear luminance in both cameras, approximately 770 times smaller in amplitude than the bright control. High correlation establishes a coarse same-light pattern in the simulated pair. It does not establish photon detectability, fine correspondence quality, or pose recovery. The controls likewise do not establish a general darkness threshold.

The measurement took 2.92 seconds and the separate arithmetic replay took 3.72 seconds. The replay recomputed the common support with independently composed relative-camera geometry, grouped blocks by sorting rather than binning, and used population covariance matrices to reproduce all 240 cross-view seed-pair cells. The validated EXR decoder was shared. No renderer, matcher or optimizer ran.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| Source | `55f920a07f245bc29c69526b47672c7f539008e567ade4a5677b1edcb8abc2ae` |
| Frozen protocol | `478d90ff7f50ff69d1f030c4875b3a7378b186bf61e4a1471c0bdf7990b5b932` |
| Result | `0a04c3f90a63c566ac9db9e3dd6ca0e256d5888a276c57ce9d29d56d9ef84498` |
| Independent replay | `b032e43d0a48ce4a15b1642a4000fcfc35338740341d14762443055282b18641` |

Extract the radiance and cross-view geometry ZIPs from the v109 research release into a clean repository checkout. The cross-view geometry ZIP has SHA-256 `537fffdc5fb63bd7a5db2a5f6e61d55e00b738f44c9c98ed106fe04b74381c84`. Run from the repository root with the validated Python environment active. [Download the archived EXRs and geometry](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-raw-radiance-v109)

```sh
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
protocol=reports/solar-raw-crossview-protocol-v110.json
prior_replay=reports/solar_raw_acquisition_replay_v109.json
checksum=478d90ff7f50ff69d1f030c4875b3a7378b186bf61e4a1471c0bdf7990b5b932
python audit_raw_crossview_v110.py measure \
  --prior-replay "$prior_replay" --protocol "$protocol" --protocol-sha256 "$checksum" \
  --output runs/solar_raw_crossview_repeat_v110.json
python audit_raw_crossview_v110.py replay \
  --prior-replay "$prior_replay" --protocol "$protocol" --protocol-sha256 "$checksum" \
  --result runs/solar_raw_crossview_repeat_v110.json --output runs/solar_raw_crossview_repeat_replay_v110.json
```
