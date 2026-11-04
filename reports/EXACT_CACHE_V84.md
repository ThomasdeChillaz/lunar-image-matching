# Explicit cache execution version

This version reuses repeated computation. It adds **no new learned technique
or measured accuracy improvement**. Its completed pilot measures a runtime
saving on the tested workloads; see [the results](SOLAR_CACHE_PILOT_V84.md) for
the timing scope and unchanged recovery. The original implementation and
earlier pose pilots remain unchanged.

`localize_terrain.py` accepts `inference_revision: solar-v84-exact-cache` with
`pose_initialization: strict3`. A separate
`solar-v84-quantized-init-exact-cache` revision retains the existing
quantization-aware engineering control. Historical revisions remain the
default. `match_terrain.py` is unchanged. Public poses remain null until a
separately validated acceptance policy exists.

## What is reused

- Immutable per-case model identities replace repeated parameter hashing;
  full byte checks run before a result can escape.
- Selected reference surface maps, canonical patch pairs and their original
  two-patch encoder outputs are reused under exact pose/location keys.
  Requested Sun direction remains part of response scoring.
- Constant reference-side predictions are reused. Query conditioning still
  uses the current pose and Sun hypothesis.
- Active selection reuses the exact expanded-action geometry projection when
  its inputs have not changed. Query measurements, Sun choices, shadow states,
  hypothesis scores and weights refresh after every acquisition.

Caches are bounded and per case. They add memory, not learned parameters.
The model remains **95,808 parameters**, and the eight hypotheses, 128-search
principal budget, 4,096 proposal samples, final AP3P estimator and 3-native-pixel
threshold are unchanged. Both the original geometry object and derived
reference-Sun geometry are checked for replacement/mutation before returning.
No process-global function replacement is used.

## Verification

The complete repository suite passes **982 tests**, with four skips. Its 75
cache checks cover exact response math, support boundaries, mutation and
failure paths, and all eight complete 128-search solver controls. The latter
use a deterministic procedural response callback; they do not test lunar
learning quality.

A fresh native image-to-pose smoke experiment passes all eight combinations
at 32 searches with an untrained constant-response head. Independent saved
arithmetic replay passes. All eight image results also match the earlier
procedural implementation exactly except declared timings, cache counters
and execution-version fields. These are software controls, not lunar results.

Evidence: [verification](solar_exact_cache_verification_v84.json),
[independent source review](solar_exact_cache_review_v84.json),
[image equality](solar_image_equality_v84.json), and
[smoke replay](solar_controlled_smoke_replay_v84.json).

All 49 learning and 42 physical-preparation source files retain their original
bytes. The original training checkpoints and physical data are reused without
relabeling or rerendering them.

## Fresh scientific measurement

The [published schedule](solar_cache_selection_schedule_v84.json) predeclares
both modes, all five checkpoints and the same eight pilot pairs. Each fresh
pilot has a 900-second cap and a separate 300-second saved-metric replay cap.
The original cumulative budget and 12,000-second maximum full-study allowance
are unchanged. No speedup follows from cache-hit counts; complete outer
inference and case clocks must be measured. The raw internal timer ends before
the last geometry check, while the outer evaluation timer includes it.

Two immutable settings files per mode are declared before inference. They
differ only in output directory: the pilot uses a new v84 directory; any
qualifying full run uses the still-empty final evidence slots required by the
original v79 learning contract. The new inference revision is explicit in
every protocol. Existing comparison checks require scientific identities to
match across phases. No files, aliases or copied results stand in for a run.

Both new full-study projections exceed admission, so those findings and the
complete case list are retained. Independently of speed, the saved diagnosis
still requires a proposal/pose-bootstrap improvement before claiming useful
learned recovery.
