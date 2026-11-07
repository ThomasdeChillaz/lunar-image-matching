# Native512 MatchAnything versus the compact surface baseline

Both methods receive the same full512px image content, reference axial geometry and3-original-pixel AP3P pose estimator. Query depth and query pose enter scoring only. Every eligible case is included. These72 pairs are eight exposed geometries/four terrain components, not a fresh final test.

| Method | Correct raw /72 | Strict /72 | Wrong raw | Median full inference (s) |
|---|---:|---:|---:|---:|
| matchanything_native | 39 | 5 | 30 | 4.396 |
| sift_first_surface | 57 | 25 | 2 | 2.981 |

MatchAnything minus the SIFT-first surface baseline: -25.00 percentage points; descriptive paired terrain-component interval [-60.00, -11.11]. This interval uses only four exposed groups after extensive development and is not confirmatory evidence.

Median paired MatchAnything/SIFT-first latency ratio: 1.48. Both models are loaded/warmed, order alternates and compact features are not reused across pairs. This ratio describes this CPU/profile and pipeline. Historical speed ratios from other workers or compact variants do not apply.

MatchAnything uses pinned released weights and unchanged author matching confidence/fine matching, with no planar affine/homography prefilter. Its native512 NPE settings are explicit; the832 profile is a separate pending control, not an assumed author-recommended lunar configuration. Raw wrong poses remain visible. No rejection rule is fitted here.

The comparison is reproducible and independently replays native geometry metrics and all144 common-estimator pose fits. It does not establish fresh geographic generalization, low-error accepted localization or orbital accuracy. MOONSt3R, remaining author profiles, transfer and calibrated acceptance remain incomplete.
