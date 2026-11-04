# Exposure planning from competing lunar pose hypotheses

This prototype chooses a fixed query ROI and frame integration time from the current pose and Sun hypotheses. It asks whether to protect a bright feature from saturation or spend more time resolving a dim feature when the two features separate different hypotheses. It runs on a deliberately constructed fixture with no archived or acquired imagery.

## Inputs and observation contract

The policy accepts a reference-side relative irradiance atlas of shape `H × N × 7 × P`, where `H` is the set of surviving pose and Sun hypotheses, `N` is the list of candidate ROIs at common query detector coordinates, and `P` is the same set of detector offsets in every ROI. Use a singleton final dimension for one irradiance value per ROI. The seven terrain perturbations use the existing active sampler order `(0, −5, +5, −10, +10, −20, +20) m`. A separate Boolean mask identifies reference coverage for each hypothesis, ROI, variant and pixel. Known low irradiance can be zero. Unknown coverage supplies zero discrimination evidence and appears in diagnostics.

An adapter for the existing active sampler would define one common set of detector ROIs, then use each hypothesized camera and Sun to project a prediction from the same reference map into each ROI. Reference-side transfer and atlas coverage must be checked for each terrain perturbation. Candidate ROIs must use the same query detector coordinates under every hypothesis so a chosen action identifies one physical measurement. Existing geometry perturbations and lookup masks can supply the order and support, while calibrated reference irradiance values would require a separate adapter. Reference gradient is a separate input for the gradient control, taken from a frozen reference view or projection. No image from a frame being planned or true query pose, depth, or Sun is an input. Any real acquisition or posterior update would happen outside the planner after a frame is selected. The existing matcher and active patch selection are separate entry points.

Each action represents one whole-frame integration, with one same-size ROI counted as its observation. The prototype assumes this is a valid acquisition pattern. All policies receive the same number of frames, integration ticks, available candidate patches and sampled pixels per frame. A discrete feasibility check prevents the joint policy from leaving an impossible remaining budget. The planner works open-loop from the weights given at entry. A sequential implementation would carry forward actual exposures and the remaining frame and tick budget when updating those weights.

## Sensor and objective

For relative irradiance `I` and integration `t`, the expected photoelectrons are `t × (scale × I + dark_rate)`. A Poisson draw saturates at the declared full well, followed by independent zero-mean read noise. The code evaluates the exact first and second moments of the clipped Poisson using truncated CDF expressions and adds read variance. This keeps the work independent of the numerical full-well size. Absolute gain, read noise and full well are synthetic fixture settings. The model omits ADC effects, exposure overhead, smear, optical transfer and correlated pixels.

A pair of hypotheses receives evidence only when both have complete seven-variant predictions for a detector pixel. The signal is the squared gap between their intervals of mean clipped charge, divided by the sum of their maximum variances. Overlapping intervals contribute zero. The policy sums this conservative separation over ROI pixels and independent frames, then uses a bounded, prior-weighted surrogate with diminishing returns for already separated pairs. This is an analytic discrimination score and cannot be read as classification accuracy or a pose probability.

The controls are equal-duration spatial coverage, equal-duration strongest reference gradients, and equal-duration hypothesis-aware ROI selection. The joint policy chooses both ROI and duration. Completely indistinguishable hypotheses fall back to spatial coverage and balanced time. The output distinguishes unresolved pairs with complete predictions from those with missing coverage, and reports the highest supported saturation probability for each selected action.

## Procedural result

All policies used three frames, nine integration ticks and four pixels per frame.

| Policy | Selected ROI and ticks | Expected separation |
|---|---|---:|
| Fixed spatial coverage | 0 at 3, 1 at 3, 2 at 3 | 0.21999 |
| Fixed strongest gradient | 2 at 3, 3 at 3, 4 at 3 | 0 |
| Fixed hypothesis-aware | 1 at 3, 0 at 3, 2 at 3 | 0.21999 |
| Joint ROI and duration | 1 at 5, 0 at 1, 2 at 3 | 0.33228 |

The fixture deliberately places separate bright and dim discriminators at ROIs 0 and 1. The joint schedule spends one tick on the bright ROI, reducing its maximum supported saturation probability from 1.00 to 0.27, and five ticks on the dim ROI. Additional controls passed for identical hypotheses, missing coverage, overlapping height-variant intervals, saturated alternatives, exact finite-sum moments at a tiny well, and stable behavior at 100,000 electrons. These results verify the analytic mechanism on constructed data. Actual atlas errors, useful ROI scales, inference cost, and pose recovery need separate tests.

## Reproduction

Run from this directory in a Python environment with NumPy and SciPy 1.13.1, the version pinned by the project’s optional dependencies.

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python check_pose_conditioned_exposure_v110.py fixture-replay.json
cmp solar_pose_conditioned_exposure_fixture_v110.json fixture-replay.json
```
