# Raw inference evidence for the first pose experiments

The evaluation worker now records the original return value from its single
backend call. The matcher, hypotheses, random state, scoring and common pose
estimator are unchanged. The 49 learning-source and 42 reference-preparation
bindings remain byte-identical. This is a reporting change before the first
new-model pose pilot, not a model improvement.

Each sealed case JSON contains `raw_inference_result` and
`raw_inference_reporting`. Existing measurements retain their acquisition order
and `response_used` flags. Incremental assignments, final hypotheses, Sun-support
flags, initialization, rounds, proposal counts and response-call counters are
preserved when the backend returned them. Numeric arrays use typed pointers to
the same hash-sealed case NPZ. Dtypes, shapes, tuples and scalar types are kept;
loading requires no pickle or object arrays.

Nonfinite values and unsupported objects are explicitly marked. An unavailable
value is never silently turned into a supported result. Reporting bounds are
16 MiB of numeric arrays, 512 KiB of compact JSON, 100,000 structure nodes and
32 levels of nesting; exceeding them fails and preserves the run instead of
truncating it. The tested 128-update, eight-hypothesis, 256-anchor trace fits
these limits. Capture is cleared between cases, including failures, and cannot
retain a previous profile's model through serialized objects.

Observer bookkeeping is inside `inference_seconds`. JSON/NPZ packing is outside
that interval and inside `case_calendar_seconds`. The existing warm/setup,
offline-preparation and worker-lifetime peak-RSS distinctions remain intact.
`raw_inference_execution_scope` distinguishes fresh execution from an exact
historical case-cache replay. Serialized byte counts are not model-memory
measurements.

Interpretation still needs care:

- A returned Sun vector can exist when `solar_supported` is false.
- Requested module flags and positive call counts establish neither a benefit
  nor a causal localization improvement.
- The frozen active selector does not log its effective policy or fallback
  reason. Both stay explicitly `unrecorded_in_frozen_solver`. Recorded search
  sequences permit matched-policy comparisons; they are not a calibrated
  information-gain posterior.

The [verification record](solar_reporting_checks_v82.json) covers 899 existing
tests, eight new trace tests and eleven related memory tests, plus all eight
fresh procedural controls and their independent arithmetic replay. The
[independent review](solar_reporter_independent_review_v82.json) checks
single-call passthrough, failure cleanup, exact typed roundtrips, bounded
storage, backend release and unchanged protected sources. These are software
checks. New-model lunar pose recovery remains unmeasured.
