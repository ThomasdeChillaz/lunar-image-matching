# MOONSt3R common-native baseline

Prepared 1 November2026. **Actual pretrained control PASSED; no lunar
performance has been measured.** The sequential queue completed repository
tests, fresh software smoke/replay and the actual pretrained control after
v18 fitting/audit/report. The control took116.97seconds including startup and
two forward passes: native translation/pose, blank and invalid-mask checks
pass. The loaded model has688,638,856parameters. See
`reports/native_moonst3r_control_v19.json`. No concurrent
fitting or other baseline job is allowed. Preserve every failure and timeout.

Source: `clementinegrethen/StereoLunar` at
`ed6ea257475abb8a5235dff3a4d6167214b6f697`.
Weights: `cgrethen/MOONSt3R` at
`f339a2c46f4e37ba7ed0ad3b7df51a1cb6aa8664`.
The official5.82GB checkpoint hash and safe tensor conversion are pinned in
`moonst3r_model_lock_v19.json`. Strict1017tensor loading uses a static reviewed
architecture, never evaluated checkpoint configuration text or unsafe pickle.
All executable upstream source and runtime versions are pinned. SciPy1.13.1
was installed only in the optional dependencies directory; its official wheel
digest was verified and the compact fitting environment is unchanged.

## Principal input and geometry contract

Both images retain their original square512pixel grayscale content/field of
view. Repeat grayscale into RGB and normalize to[-1,1]; no external crop or
resize. This is an explicit common-content comparison: the author image loader
normally crops a square512image to512x384. It is not an assertion that our
settings are the author's recommended lunar evaluation profile.

Use author reciprocal descriptor nearest-neighbour matching, dot distance,
stride8, block8192 and at most10iterations, with its3pixel border. No affine
or homography filter. Every matcher receives the same reference axial depth,
camera parameters and3originalpixel AP3P/RANSAC/LM pose estimator. MOONSt3R's
predicted3D is unused. True query depth and pose enter scoring only.

The checkpoint names training split files absent from the released repository.
Exact training-pair/terrain overlap is unknown, and any comparison must disclose
that limitation. The source top-level Apache2license does not replace nested
MASt3R/DUSt3R CC BY-NC-SA4terms. Weight-card license is unspecified; do not
redistribute pretrained weights or treat them as the small model's parameters.

## Execute only after the gate and public source freeze

```sh
python control_native_moonst3r_v19.py \
  --dependency-dir /path/to/optional/dependencies \
  --output runs/localization-v19-moonst3r-native-control --max-seconds 600

python evaluate_native_moonst3r_v19.py \
  --dependency-dir /path/to/optional/dependencies \
  --source-commit PUBLIC_COMMIT \
  --output runs/localization-v19-moonst3r-native-development

python verify_native_moonst3r_v19.py \
  --run runs/localization-v19-moonst3r-native-development \
  --output reports/native_moonst3r_audit_v19.json
```

The completed queue already executed the first command; do not repeat it or
overwrite its output. Selected v18 controls are running next; no baseline
inference until that job ends.
Its software gate includes actual pretrained forward passes, known native
translation/pose, blank images and invalid masks. After a successful gate,
measure cost before deciding on the full72case comparison. `--bank` selects
the fixed16exposed-case bank, never a success-based subset.

The comparison preserves per-case failures and includes every eligible case in
the denominator. It can resume only with identical sources/protocol/case lists.
A shared four-hour deadline includes startup and interrupted wall time; resume
does not reset the allowance. The overall72CPUhour pretrained budget still
applies. Report incomplete comparisons explicitly. Saved-metric replay checks
native geometry, common poses, controls, hashes and denominators; it does not
reimplement the neural model. Timings are not a paired-worker speed comparison
with another method. These are eight exposed geometries/four terrain groups,
not a new final test, calibrated accepted navigation or SOTA result.
