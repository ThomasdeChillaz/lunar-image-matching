# Identify the matching failure before changing the model again

This extends the active study with a saved-result analysis and a **prepared,
unexecuted** coarse-proposal diagnostic. It does not modify the running v39
cohort, any acceptance threshold, a pose criterion, or any reserve case.

## Evidence available during training

The full frozen baseline recovers 12/24 same-light poses and only 1/72
changed-light poses. At the first seed's selected 1,500-update checkpoint,
those counts are 12/24 and 2/72. The primary seed completed 2,000 updates;
its endpoint returns to 13/96 overall. Its selected checkpoint has 14/96
correct positives, 15 wrong positive proposals and 4/96 raw nonoverlap
proposals, versus 13, 28 and 11 for the frozen baseline. The primary selected
known-transform/blank controls pass. These are one-seed development results;
the repeat seeds and full independent cohort audit are unfinished.

At the later source freeze, the second seed has also completed: it selects
step 500 with 15/96 correct poses, 10 wrong positive proposals and 7/96
nonoverlap proposals; its endpoint gives 14/96. The third seed and complete
cohort audit remain unfinished. [Both completed seed receipts](reports/north_two_seeds_complete_v42.json)
retain all checkpoint readouts; no best-of-seed result is adopted.

The gain is one positive pair (1.04 percentage points), far below the ten-point
target. It is not sufficient to claim a reliable improvement, 70% correct
acceptance, a 1% error operating point, or superiority to modern models.

The saved native correspondences make the failure more specific. For the 72
changed-light pairs, two-pixel precision over **all returned points** is 2.715%
initially, 2.150% at step 500, 1.479% at step 1,000, and 2.270% at step 1,500.
Thus fewer wrong poses do not demonstrate better fine correspondence precision.
These thresholds are analysis only and do not alter checkpoint selection.
The saved-results reporter also keeps scorable-only precision separately.

The console prints loss every 100 updates, while every fourth training example
is a nonoverlap pair. Consequently every printed loss is a negative example.
For the completed primary run, changed-light positive mean loss changes only
5.3532 → 5.3009 from the first 500 to last 500 updates; negative loss is
0.1116 → 0.1130. The small printed values must not be interpreted as successful
positive matching. The full history, rather than only console lines, provides
these separate curves.

The interim snapshots are immutable:

- [First saved-result snapshot](reports/north_lighting_failure_interim_v42.json)
- [Snapshot including the completed primary seed](reports/north_lighting_failure_interim_2_v42.json)

They may include different numbers of completed repeat checkpoints because the
cohort is live. Neither includes partial per-case directories. These snapshots and the prepared diagnostic sources are published for review;
new neural execution waits for the queued full regression. The second snapshot is
not a claim that all three seeds have finished.

## Test the coarse-versus-fine explanation

Low final point precision alone cannot identify the stage that failed. The
prepared v43 diagnostic will compare frozen coarse proposals with the primary
seed's 2,000-update endpoint on all 96 positive selection pairs. This endpoint
comparison is a failure diagnosis, not retrospective adoption of that endpoint.

1. Run the same 128px image-only proposal generator for all four fixed query
   rotations. Map its half-pixel grid exactly to the original 512px coordinates.
2. Score returned proposal locations against the existing exact fractional BVH
   labels **after inference**. No query depth, pose or case ID enters matching.
3. Count genuine proposals, correct native-pixel proposals, geometrically visible
   reference points offered, and true correspondences lying inside the actual
   33×33 native-pixel refinement search. Use its rounded search center and exact
   support mask; do not approximate the square window by a Euclidean radius.
4. Retain branch counts and the deduplicated union over reference points. Report
   the maximum reachable matches in any single branch as well; a six-point
   union spread across four branches does not prove one branch can fit a pose.
5. Add a clearly separate **oracle local-refinement control**: replace only
   reachable predictions with exact query points, reject everything else, and
   run the unchanged reference-geometry/AP3P estimator. Query truth is used, so
   these are counterfactual diagnostic poses, never learned accuracy or a usable
   branch-selection rule. Keep all failures in the denominator.

Low recovery by this oracle weakens the hypothesis of local refinement alone.
Inspect proposal counts, spatial support and reference-geometry errors before
committing to another fine head. If the oracle performs well while the real
fine stage fails, learned local refinement and pair-specific reliability are
better-supported hypotheses. This is a counterfactual diagnostic, not a strict
mathematical bound on pose success: conditioning, depth sampling and the
estimator can also matter. Neither outcome promises that learning reaches it.

## Execution gates and budget

First finish all three predefined seeds and the 1,728-row/6,000-sampler-record
audit. Let the separate queued repository regression and fresh smoke finish
after the scientific controller and its child processes exit. The prepared sources are published before execution; byte-verify that public
revision and the completed regression before a diagnostic neural forward.

The separate verifier independently replays the canonical grid, matrix-based
quarter turns, rounded search windows, branch unions and all 768 oracle-pose
controls from saved predictions. It loads no neural network. The v43 program
refuses incomplete upstream evidence, changed checkpoint seals,
changed public source, and concurrent upstream processes. It uses two threads
and one fixed 1,800-second calendar/monotonic allowance. Per-branch records are
cached and checked on explicit resume within the original absolute deadline.
A frozen-model known-transform/blank/invalid-mask control must pass. A failing
endpoint control is retained as a negative finding, not hidden.

```sh
python report_north_lighting_failure_v42.py \
  --output runs/north-lighting-final-snapshot.json
python north_coarse_reachability_v43.py --publication-commit PUBLIC_COMMIT
python verify_north_coarse_reachability_v43.py
```

Six saved-metric/loss tests, six coarse-grid/oracle-mask tests and three
independent-replay/atomic-cache tests pass in isolation. The camera-design
study has three additional isolated metadata tests. No new neural probe, oracle pose estimator or optimizer ran alongside
the active cohort. Do not add those isolated counts to the previous 307-test
suite and present the sum as a new full regression. Preserve all snapshots and
the earlier failed deadline; no paid compute, human review or email.

The exact source used for the first interim snapshot is retained as a
[byte-matched historical source record](reports/north_lighting_failure_initial_source_v42.py.txt).
The later reporter adds separate training-loss profiles; it does not rewrite
the earlier snapshot or its source hash.

The fixed-camera limitation and original released-data shortfall remain. A
separately frozen unseen-camera/roll study, modern comparisons, calibration,
uncertainty and the final research package are still required for the original
goal. This diagnostic does not redefine those targets around a small gain.
