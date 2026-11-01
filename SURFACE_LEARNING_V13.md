# Geometry-normalized training protocol

This is an exposed-development hypothesis, not a new final evaluation.
The frozen v11 surface control gives54/72 correct raw poses; the fixed SIFT
fallback gives57/72. Earlier learned v8 seeds47/45/47 do not beat this control.

Test whether rectifying training query images using the reference3D surface
improves native correspondence precision under viewpoint and lighting changes.
True query pose is permitted only for training normalization and scoring.
Inference consumes an image-estimated query pose and reference geometry only.

Three primary-seed ablations use identical visible reference points, component
sampling, light choices, fractional crop jitter and sensor perturbation draws:

1. Raw3D encoder supervision, evaluated with the surface pipeline.
2. Geometry-normalized encoder supervision, otherwise identical.
3. Geometry-normalized supervision with the existing small pair-scoring head.

The common training support intersects projected query visibility, raw crop
bounds and full rectified crop support. Integer reference rays avoid depth
interpolation ambiguities. Fractional query crop centers retain subpixel labels.
Normalization also changes resampling and effective search extent; a gain
cannot be attributed solely to the architecture. Rendered lighting variants
supply physical relighting; exposure, blur, noise and quantization do not.

Each run uses2000 AdamW updates, eight visible examples per batch, learning
rate0.0001, weight decay0.0001 and two CPU threads. The frozen coarse network
and trainable fine network together remain below100000 parameters. Negatives
and a new reliability head are deliberately deferred: this study isolates
local precision before revisiting the previously collapsed no-match head.

Freeze all72 exposed model-selection cases as the selection set before fitting;
select by unfiltered pose success, then2px-correct correspondence count, then
earlier checkpoint, including initialization. Evaluate every500 updates. This
replaces the small16-case selection bank for this new cohort only; prior
cohorts are unchanged. Run seed20261101 for each ablation and repeat the selected
configuration with20261102/20261103. Publish failures and initial-checkpoint wins.

The frozen four-orientation proposal generator and two surface-refinement
updates are unchanged. Final AP3P retains its3native-pixel threshold. No
initial pose means no recovered pose; this known coverage limit remains.
Before fitting, verify both initialized variants against every historical
v11 primary/strict status, and encoder-only native points and poses.

Do not load calibration/test pixels or claim a fresh test. The current data
still have eight selection geometries in four terrain components and shared
renderer dependence. Acceptance calibration, nonoverlap risk, transfer and
fresh-final targets remain incomplete. Do not call an unfiltered pose accepted.

Run the prepare/control gate first:

```bash
python train_surface_learning_v13.py --output runs/localization-v13-surface-development --prepare-only
```

Then run the same frozen protocol without `--prepare-only`. Outputs are
immutable; optimizer, RNG states and selected checkpoints are saved every100
updates for recovery. Completed fitting directories cannot be overwritten.
The per-run wall cap is six CPU wall-hours; this is a cap, not permission to
exceed the overall research compute budget.
