# Expanded-data architectural control

This is a **prepared, gated study**, not a measured model improvement.
The corrected v37 render is running; the v38 train-only pilot is queued.
The original architecture, loss and optimizer stay fixed so this next control
tests the effect of expanded terrain and illumination data.

## Gates before full fitting

1. Complete all 24 development terrains and 576 images under the original
   render deadline. Pass the unchanged 72-geometry oracle gate, including
   243,360 fractional-point labels and the corrected solar-vector checks.
2. Prepare the six-light v38 bundle. Finish the train-only 100-update pilot
   and independently replay its model, optimizer, losses, gradients and RNG.
3. Complete the fixed 16-case cost bank. Keep a timeout or failure explicit.
   Run full repository tests and a fresh controlled software smoke.
4. Publish and verify this source before inference/fitting. Run both frozen
   compact/surface and SIFT-first/surface baselines on all 96 positive and
   96 nonoverlap selection cases. Replay all 384 saved metric rows.
5. Use measured pilot and inference costs to check the full cohort fits a
   fixed six-hour calendar allowance. This conservative estimate is an
   admission check, not a measured runtime or efficiency claim.

The program refuses full fitting if any gate is missing or changed. A failed
gate requires diagnosis and preserved evidence, not a silent weaker substitute.

## Fixed experiment

- Architecture: original coarse-adaptation SearchWindowModel, 81,696 total
  parameters, 53,424 trainable. The native fine encoder stays frozen.
- Original v18 bidirectional reachable-window objective and 0.5 genuine-anchor
  term. AdamW context rate 1e-4, coarse rate 1e-5, decay 1e-4, gradient clip 5.
  The failed v23 dual-softmax objective is not reused.
- Train only on the 16 training terrain groups / 48 geometries. Component-balanced
  sampling, all six lighting settings on either image, four quarter turns,
  and a distinct nonoverlapping training component every fourth update.
  Exposure/blur/noise/quantization are sensor augmentations, not sunlight models.
- Run seeds 20261101, 20261102 and 20261103, 2,000 updates each. Evaluate at
  steps 0, 500, 1,000, 1,500 and 2,000. Publish every seed and endpoint.
- Choose using all 96 eligible positive selection pairs / eight terrain groups:
  correct raw pose count, then correct two-pixel correspondence count, then
  earliest checkpoint. Failed poses remain in the denominator; initialization
  remains eligible. Initial native points and poses must exactly match the full
  frozen baseline on the same new cases.
- Independently check the selected model's known-transform and blank controls.
  A model failing them is ineligible for adoption; keep its failed outcome.
- Evaluate all 96 nonoverlap selection pairs at each selected checkpoint.
  These are **raw proposals**, not calibrated acceptance or a 1% risk result.
- Reference axial depth/pose and query intrinsics are inference inputs.
  Query depth and true query pose enter scoring only. The 3-native-pixel AP3P
  estimator, pose criteria and reference-range normalization stay fixed.

The original backbone is checked unchanged. Frozen fine-image features may be
cached for selection, with a 64-map bound (about 2.2GB), while coarse scores, poses and warped features are
recomputed. These cached selection times are unsuitable for speed claims.
The separate baseline comparison measures actual inference with alternating
method order and fresh caches.

## Interruption and verification

One scientific job at a time, two CPU threads, existing local resources only.
Every optimizer update saves an atomic model/optimizer/data-RNG boundary.
Checkpoint rows save individually and resume without replacing earlier cases.
The cohort stores one absolute deadline; resuming does not grant more time.
Every saved result and failed attempt is retained.

The v39 auditor replays all checkpoint and selected-negative metrics (1,728
rows), fixed case lists, selection, frozen fine weights, optimizer step counts
and all 6,000 data-sampler records. It does **not** claim a 6,000-update exact
optimizer retraining; only the separate 100-update pilot is exactly retrained.
These checks are distinct from the independent v37 geometry-label oracle.

```sh
python benchmark_north_control_v38.py --full --max-seconds 7200 \
  --output runs/north-v39-full-baselines --publication-commit PUBLIC_COMMIT
python verify_north_results_v39.py baseline \
  --run runs/north-v39-full-baselines \
  --output reports/north_full_baseline_audit_v39.json
python train_north_control_v39.py --publication-commit PUBLIC_COMMIT
python verify_north_results_v39.py cohort \
  --run runs/north-v39-expanded-control \
  --output reports/north_expanded_control_audit_v39.json
```

## Limits and next decision

No four calibration or twelve candidate-final support pixels are loaded.
The failed original fresh released-LunarGeo gate remains failed; northern
self-rendered data are an explicitly additional study. External pretraining
overlap remains unknown.

Terrain-held-out evaluation still uses four fixed relative camera templates.
It cannot establish unseen-relative-pose or trajectory generalization. Before
broad claims, add an independently frozen pose/roll transfer test and a simple
pose-prior control. Never pass case IDs or query-view IDs to a matcher.
Lighting uses an idealized +1.5-degree subsolar-latitude family, not dated
ephemerides. Point Sun, finite terrain halos and uniform material remain
simplifications. Simulated pose accuracy is not verified orbital navigation.

This control can reveal whether data help before further learned refinement
changes. If all trained checkpoints lose, retain the frozen baseline and
investigate the failure. A new architecture, calibrated error/coverage result,
modern-model comparison, transfer, uncertainty, paper and compact-v3 release
remain separate required work. No superiority claim or email follows from
training loss alone.
