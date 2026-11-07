# Native-detail scale diagnostic

Executed 1 November2026 after the v18 cohort, selected-checkpoint replay and
MOONSt3R software control all completed. **No pose gain:** native/half/quarter
profiles each12/16, strict5/4/4, one wrong estimate each. All48 native metric/pose
rows independently replay. Source freeze `fac436adf00c9c46d7f24f40b793154830574f9f`.
See [measured findings](reports/fine_pyramid_bank_v22.md). No full72-case scale
run was launched because neither multiscale profile improves raw poses. The
protocol below is retained as the executed bounded study. One two-thread CPU
job at a time. Historical fitting sources are unchanged.

## Hypothesis and bounded test

The historical descriptor sees a25pixel field at128pixel pretraining input.
At512pixel native input the same field covers four times less terrain. V13
native fine learning produced only an inconclusive one-pair gain. Before another
training cohort, test whether shared frozen descriptors at image scales1,2,4
help recover context while retaining native-pixel candidate coordinates.

Use the existing fixed16-case exposed model-selection bank first. Publish all
three profiles, including failure: native only, native+half scale, and
native+half+quarter scale. Each starts from exactly the same coarse image-only
pose and original reference points. Compare equal-weight cosine scores inside
the same33nativepixel search, bounded strongest-peak subpixel interpolation,
the same3pixel AP3P estimator and two reference-surface refinement iterations.
No new learned parameters, query depth or true query pose enter inference.

All three profiles use the same conservative56pixel descriptor support and
72pixel surface-warp support. Include the historical v11 result separately:
the larger common border can itself change localization. Resize with area
averaging and use `(native+.5)/scale-.5` pixel centres. Lower-resolution
descriptor interpolation is normalized. Scale1-only decoding delegates exactly
to the historical routine on the retained point indices.

Run actual selected-checkpoint rotation/pose and blank controls before loading
lunar images. Source hashes, input hashes and case lists are saved before any
inference. The existing manifest and train/model-selection split are unchanged;
no calibration/test imagery is loaded. Shadows are appearance, not occlusion.

```sh
python evaluate_fine_pyramid_v22.py \
  --source-commit PUBLIC_COMMIT \
  --output runs/localization-v22-fine-pyramid-bank
```

The runner preserves every attempt and writes native correspondences, poses,
failures and all three outcomes. Its bounded wall budget is1800seconds after
the known-transform controls. Cross-case descriptor caches are permitted for
this diagnostic; the times cannot support an inference-speed claim.

## Decision after the bank

If a scale profile helps, repeat the declared profiles on all72 exposed pairs
with `--full` and independently replay all native geometry and pose metrics.
This extends a development diagnosis, not an untouched test. Do not select a
different profile using each case's truth. If only reduced support helps, report
that engineering result. If scales fail, retain the failure before changing
training. A learned extension needs a separate train-only cost/replay pilot,
matched initialization and supervision controls, and a public training freeze.
No breakthrough, accepted navigation, fresh transfer or SOTA claim follows
from a development-bank gain.
