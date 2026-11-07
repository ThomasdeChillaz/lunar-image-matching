# Northern data complete; three-seed fitting started

The additional northern development study has passed its data and fitting
software gates. This is progress toward a model comparison, not a learned
accuracy improvement or a passed original fresh released-LunarGeo gate.

## Completed data and pilot

All 24 development terrains / 576 images are complete. Sixteen completed
components were copied byte for byte; the separate eight-component recovery
took **726.75 calendar seconds**, within its explicit 1,800-second amendment.
The original v37 render's deadline remains failed and its files stay preserved.

The unchanged independent auditor recovers **72/72 oracle camera poses** and
checks **243,360 exact fractional-grid label points** across 360 banks.
Maximum oracle translation/rotation errors are 0.8442m / 0.01939 degrees;
these are geometry controls, not matcher results.

The **100-update train-only pilot completed in 17.22 fitting seconds**
(21.29 calendar seconds). Its 81,696-parameter model has 53,424 trainable
parameters; the fine encoder stays frozen. All six lighting settings were
sampled for both images. Independent retraining reproduced model, optimizer,
losses, gradients and random states exactly. The pilot did not evaluate
selection accuracy; training loss is not a localization result.

Evidence:

- [Bound data/pilot summary](reports/north_data_and_pilot_complete_v41.json)
- [Complete geometry audit](reports/north_development_audit_v37.json)
- [Versioned training manifest](reports/north_training_data_v38.json)
- [Measured pilot summary](reports/north_training_cost_pilot_v38.json)
- [Exact pilot replay](reports/north_training_pilot_audit_v38.json)

## Completed common baseline reference

Both pipelines used the same 96 positive and 96 nonoverlap selection pairs,
reference axial geometry, native image content and fixed pose estimator.
Query depth and true query pose enter scoring only. All **384 saved metric
rows replay successfully**.

| Pipeline | Correct /96 positives | Strict | Wrong positive proposal | No positive pose | Nonoverlap proposals /96 |
|---|---:|---:|---:|---:|---:|
| Frozen compact + surface | 13 | 2 | 28 | 55 | 11 |
| SIFT-first + surface | 13 | 2 | 28 | 55 | 11 |

Correct raw pose recovery is **13.54%**. This is a weak baseline, far below the
original 70% correctly accepted target; the two metrics are also distinct.
Identical totals do not establish identical point correspondences. Neither
pipeline has a calibrated acceptance rule in this study. The nonoverlap
proposals are explicitly unsafe raw research outputs, not accepted navigation.

These are different data from the old 72-pair exposed study. Do not interpret
13/96 versus the historical 54/72 as a matched model degradation experiment,
or mix these counts into one benchmark. The new lighting and terrain study
tests a substantially different distribution.

[Full baseline summary](reports/north_full_baselines_v39.json),
[saved-metric replay](reports/north_full_baseline_audit_v39.json).
The loaded-model median inference times over the complete common positive/
negative case mixture were 2.053s and 2.120s respectively. They are measurements
for these two pipelines only, not an efficiency advantage against modern models.

## Active fitting and remaining claims

The publicly frozen v39 architectural control has started with seeds 20261101,
20261102 and 20261103, 2,000 updates each. All 96 positive selection pairs are
evaluated at steps 0/500/1,000/1,500/2,000. Initialization remains eligible;
every seed, endpoint and failed attempt is retained. Each selected checkpoint
gets all 96 nonoverlap diagnostics and actual known-transform/blank controls.
The later saved-result audit does not claim a complete 6,000-update optimizer
retraining; exact optimizer replay is established only for the 100-update pilot.

The production fitting code passed 307 repository tests with four optional
skips and a fresh 100-row procedural software smoke before launch. One
scientific CPU job runs at a time with two threads. A separate metadata-only
camera-design diagnostic is being checked locally; it does not alter this run.

Keep four calibration and twelve final supports unused until the complete
checkpoint, confidence and case freeze. Fixed camera templates, idealized solar
geometry, common renderer/DEM dependence and unknown external pretraining overlap
remain limitations. Fresh evaluation, calibrated acceptance, modern profiles,
unseen-camera transfer, uncertainty, a paper and compact-v3 release remain
unfinished. No new model improvement, SOTA result or orbital accuracy is claimed.

[Training protocol](NORTH_EXPANDED_CONTROL_V39.md),
[preserved recovery amendment](NORTH_RENDER_RECOVERY_V40.md).
