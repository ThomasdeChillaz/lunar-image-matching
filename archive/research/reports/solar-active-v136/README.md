# Sequential patch acquisition: software prepared, lunar comparison pending

The previous joint-correlation pilot improved a faint camera proposal to 10.02 m
error but retained only five reliable image matches. Its final recovery still
tied Direct. This implementation asks whether **additional image measurements
chosen to distinguish camera hypotheses** can produce a useful final pose.
It adds no network parameters and changes no acceptance thresholds.

**No v136 lunar inference or training has run.** The original comparison envelope
has zero seconds left. The request for a separate 40-minute CPU allowance remains
pending; this code, unit tests and small generated-image control grant no new
scientific execution authority. Calibration and final terrain remain closed.
The fourteen original research targets are unchanged.

## Implemented mechanism

1. Run the unchanged v135 36-anchor image-correlation fit. Preserve its result as
   `joint_no_extra`, including failures. With zero extra budget, endpoints, camera
   and estimator exactly match the frozen implementation.
2. Create 256 deterministic reference anchor candidates. Exclude missing reference
   geometry, unavailable predicted support and locations within 12 pixels of an
   existing anchor. Build a separate reference-only 3D raster for the extra pool.
3. Around the current predicted camera, form seven bounded local hypotheses along
   three weak directions of the acquired points' projection Jacobian. This search
   cloud is not a calibrated covariance or independent camera validation.
4. Predict nine cheap reference-preview intensity changes with a first-order
   gradient/displacement model. Compare each hypothesis's prediction envelope
   over reference-point shifts of 0, +/-5, +/-10 and +/-20m along an explicitly
   supplied reference-frame height direction. Unknown hypothesis pairs contribute
   zero support; scores keep the full pair-weight denominator.
5. Select one candidate, acquire its actual 64-pixel query correlation field, and
   refit the six-dimensional camera using all acquired fields. Rebuild the local
   camera hypotheses and their tempered evidence weights before the next choice.
   A failed query search still consumes its slot.
6. Select camera-conditioned image-grid modes, fit fractional peaks, apply the
   original appearance/ambiguity/independent reverse checks and finish with the
   unchanged six-match, three-native-pixel AP3P/refit. Public poses remain null;
   raw hypotheses require the explicit research-output function argument.

Up to 32 additional forward fields are requested. The final reverse check uses
only selected/acquired fields. Full native query correlations are never computed
for unselected extra candidates. Global query filtering and the original 36 fields
remain shared preprocessing. Reference preview costs and hypothesis maintenance
must be included in eventual timing; they are not free.

## Causal controls and limits

Four policies share one eligible pool, the same spatial slots, extra search cap,
query pipeline, camera update and final solver:

| Policy | Candidate ranking within its common spatial slot |
|---|---|
| Uniform | Deterministic distance to cell centre |
| Appearance | Energy of the nine normalized **reference** preview gradients |
| Geometry | Robust separation of projected-pixel envelopes, without intensity prediction |
| Active | Robust separation of predicted intensity-change envelopes |

The appearance control is reference-texture ranking, not an exhaustive
query-matching score bank. A future stronger coarse-appearance control is a
separate comparison; the original full-study requirement remains open.

The extra raster uses reference 3D support without the historical fitted-plane
intersection mask. All four extra-acquisition policies receive that same raster.
Thus an improvement over `joint_no_extra` alone would combine more measurements
with a different additional-template path. **Only gains over the equal-budget
acquisition controls can support a selection contribution.**

The image prediction is a linearized local approximation, not a physical
relighting/rendering model. Height shifts perturb point projections, not the
full uncertain terrain or its occluders. No solar-shadow or camera-visibility
ground truth is generated from brightness. Camera-conditioned endpoints are
not independent landmarks. No lunar pose gain, empirical risk, full-runtime
advantage, unknown-Sun robustness or novelty is established by software tests.

Active image search itself is established prior art. Davison's
[Active Search for Real-Time Vision](https://www.doc.ic.ac.uk/~ajd/Publications/davison_iccv2005.pdf)
uses prior uncertainty and information measures to guide processing and compare
measurement utility with its cost. This code instead tests a particular masked,
height-envelope separation proxy around the lunar matcher. Its score is not
Shannon mutual information in bits, and this focused check does not establish
novelty against the full literature.

## Verification and replay

Ten focused tests cover sequential hypothesis updates, the acquisition boundary,
unknown height coverage, unrenormalized missing evidence, equal spatial/search
budgets, failed searches, flat previews, invalid inputs, rigid camera clouds,
zero-extra equivalence and native generated image-to-pose controls.

The small end-to-end smoke uses two generated native 512-pixel scenes and **two extra
patches per policy**, with an independent saved-arithmetic replay of all ten
method/scene cells. It opens no lunar images, accesses no calibration/final data
and is not the full 32-patch cost/admission pilot.

```sh
python -m pytest -q reports/solar-active-v136/test_active_v136.py
python reports/solar-active-v136/smoke.py --output new-software-smoke.json
python reports/solar-active-v136/replay_smoke.py new-software-smoke.json --output new-software-replay.json
```

Use `requirements-lock.txt`, two CPU threads and the repository root as the
working directory. The saved `procedural-smoke.json` and replay are software
controls only. The historical JSON-artifact verifier targets another schema;
the explicit replay above verifies this control's independent geometry.

## Next scientific admission

The [v135 proposal](../solar-joint-v135/NEXT_EXPERIMENT.md) remains unexecuted.
The required geometry-only control adds one arm: the prospective comparison now
needs fresh Direct, joint without extras and all four acquisition policies,
alongside the unchanged historical controls. Preserve all 16 positives and 16
nonoverlaps. The forecast must fit the proposed allowance with the complete
scope; do not silently remove controls if it fails.

Before any lunar execution, record explicit supplementary resource authorization,
publish source/case/preprocessing/metric hashes, bind the exhausted original
journal, and pass a charged full-budget generated timing gate. The supplementary
controller must enforce the approved independent cap without editing or refunding
the original ledger. No supplementary allowance or launcher is created here.
No email, paper submission or model promotion is part of this preparation.
