# Native repair completed: no pose improvement over Direct

All **160 method/case cells** completed in **189.48 seconds**, including
16 positives and 16 certified cross-terrain nonoverlaps for five methods.
Saved-score replay passes. Prediction was sealed before opening query
geometry. The original v126 zero-measurement result remains unchanged.

| Method | Correct poses /16 | Retained matches, all positives | Matches within 2 native pixels | Raw nonoverlap poses /16 |
|---|---:|---:|---:|---:|
| Original measurement with common encoding | 0 | 0 | 0 | 0 |
| Boundary halo | 0 | 0 | 0 | 0 |
| Masked pooled measurement, uniform selection | 0 | 14 | 6 | 0 |
| Masked pooled measurement, parallax selection | 0 | 17 | 6 | 0 |
| Full Direct reference library + matched-light reference | **4** | 1,868 | 1,817 | 0 |

The repair fixes part of the measurement failure, but **does not beat
Direct or recover a pose**. Seventeen retained matches are spread over
16 pairs; no pair supplies the common estimator's six reliable matches.
Parallax selection adds three retained matches without adding two-pixel
matches or correct poses. Neither patch-selection benefit nor calibrated
acceptance is established.

Direct's four successes are the bright condition's four independent-noise
pairings. All methods fail on the faint and two near-uniform conditions.
These four camera conditions reuse **three terrain components**; sixteen
render pairings are not sixteen independent geographies. All images use
supplied matching illumination. This is a diagnostic development subset,
separate from the historical 48-positive modern comparison.

## What still fails

The old fine pixel score and reverse support checks explain why v126 kept
zero measurements. Masking, pooled fine scores and comparison context
restore some measurements, with correlation/reciprocity settings unchanged.
The corrected parallax arm still has 39 weak forward peaks, 41 uniform
templates and 19 nonreciprocal results across its positive search attempts.

A **post-hoc geometry-only** audit finds seven to nine selected anchors
within two pixels of an allowed local-grid position on the faint/bright
pairings. Actual matching recovers far fewer. The near-uniform u025 case
has no coarse seed; u027 has zero or one recoverable selected location.
This geometric potential is not observed matching or a guaranteed pose
bound. It motivates testing a globally shared camera using aggregated
image evidence before native correspondence measurement.
[Audit code](audit_native_window_v130.py) · [Exact capture counts](capture-audit.json)

Median measured positive inference time is 4.40 seconds for repaired
parallax selection and 1.29 seconds for Direct on this CPU workload.
Actual reference-cache misses and hits are included in frozen case order;
query matching always reruns. These are reference-resident development
measurements with shared patch setup, not a standalone deployment speed
claim. No modern accuracy–efficiency advantage is established.

The paired descriptive three-component bootstrap difference versus Direct
is −25 percentage points, with a very broad interval [−78.33, 0]. Repeated
negative endpoints are weighted together. Zero raw nonoverlap poses
does not establish a 1% acceptance risk. Public results remain uncalibrated;
calibration and final terrain were not opened.

## Reproduce

From the repository root after extracting the development replay bundle:

```sh
python reports/solar-local-v130/replay_saved.py
python reports/solar-local-v130/audit_native_window_v130.py
```

[Exact summary](results.json) · [Frozen protocol](protocol.json) ·
[Preserved original failure](../solar-parallax-v126/RESULTS.md) ·
[Next shared-camera control](../solar-local-v131/METHOD.md)
