# Replay the native development experiments

Use a fresh checkout of tag `solar-native-v131-development`, then extract
`lunar-native-v131-development.zip` into the repository root. The archive
preserves the source and input paths in both frozen protocols. Its
`NATIVE_REPLAY_INVENTORY_V131.json` lists each file's hash and size.

Use the validated environment in `requirements-lock.txt`. The scientific
runtime was Python 3.12.14, NumPy 1.26.4, OpenCV 4.10.0 on a local x86_64 CPU.
No model downloads, new training, rendering or paid compute are needed
for the saved-score replay:

```sh
python reports/solar-local-v130/replay_saved.py
python reports/solar-local-v131/replay_saved.py
python reports/solar-local-v131/score_proposals.py
python reports/solar-local-v131/render_results.py
```

The principal replay checks all 224 final method/case cells and requires
the original five control correspondences, camera poses and estimator
evidence to equal the preceding 160-cell experiment. Proposal scoring is
post-hoc and remains separate from the principal camera outputs. Replayed
timings are stored measurements, not new runtime measurements.

The bundle also preserves the v126 zero-measurement output, refused timing
attempts, successful cost controls and preflight publication refusal. It
does not include a compute reservation journal, worker authority, model
weights, private correspondence, calibration images or final-test images.
The reported local and clean-checkout software tests are separate from
the scientific accuracy result.

MoonAnything/LunarGeo-derived geometry and reference renders retain the
CC BY 4.0 attribution to C. Grethen, Y. Shi, S. Gasparini and G. Morin and the
source provenance in [THIRD_PARTY.md](../../THIRD_PARTY.md). The archive
contains adapted development data and independent renderer-noise repeats;
these are not new independent lunar terrains or verified orbital poses.
