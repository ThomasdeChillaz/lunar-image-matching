# Persistent lighting and camera hypotheses during patch selection

This implementation addresses the measured v142 failure: the old pipeline
selected one camera/light combination before acquiring additional native image
measurements. The new pipeline keeps qualified alternatives alive until final
pose estimation. **No new lunar recovery result has been measured yet.**

## What changes

1. Search the unchanged 24-light catalogue and up to 2,048 reference-side cameras.
   Retain up to four lighting choices and eight total camera/light seeds, using
   reference-anchor projection distance to avoid spending every seed on nearby
   cameras. With four feasible lights, this allocates two cameras per light.
2. Apply the frozen photometric refinement and record every seed that fails its
   feasibility gate. Up to eight qualified alternatives enter acquisition.
3. Predict patch observations under each branch's shared reference illumination,
   current camera and seven local height perturbations (0, +/-5, +/-10, +/-20m).
   Select a reference anchor, acquire an actual native correlation field for each
   branch, update each camera and recompute branch weights before choosing again.
4. Keep every qualified branch alive throughout acquisition. A fixed exploration
   floor retains 10% total weight across alternatives. Missing map support and
   failed fields cannot become supporting evidence; they remain in the attempted
   measurement denominator. These are search weights, not calibrated confidence.
5. Evaluate each surviving branch with the unchanged reciprocal native image
   measurements and AP3P/RANSAC: at least six matches and a three-pixel threshold.
   Choose one complete camera/light branch, without mixing its correspondences
   with another branch's lighting or substituting a soft camera for a failed pose.

The principal new controls share a **128-field total cap**, counting one actual
branch-specific correlation attempt as one field. Eight branches receive 16
anchors each, not 128 each. If seven branches qualify, the common effective cap
is 126 for every new control, including the single-branch control, provided the
common reference candidate pool has sufficient capacity. Scarce shared coverage
reduces the allowance for all methods together. Unsupported attempts are charged;
actual counts and unused capacity remain explicit. No
hidden initial set of 36 native query fields is added to the new methods.

Reference-only warped rasters and previews are prepared for the branches; native
query correlation fields are deferred until selection. Runtime includes that
reference preparation, hypothesis management, per-branch updates, reciprocal
checks and final pose fits. Equal field budgets do not imply equal latency.

## Controls and interfaces

The planned matched study includes Direct, frozen no-extra and active+32
controls; a single-hypothesis uniform control using the new initializer and full
common field budget; and uniform, geometry and active selection with the same
persistent branches. The first comparison tests retaining branches; comparisons
between beam policies isolate selection. A gain over the old smaller patch
budget alone cannot establish a beam or selection contribution.

The reference candidate universe is the union of qualified branches' known
reference support. The single-camera control prefers its own supported candidates
within each spatial slot, and charges failed attempts after that support runs out.
Reference coverage cannot silently give one method extra attempted measurements.

`match_terrain.py` and `localize_terrain.py` retain their existing behavior. The
separate experimental command accepts the inference-only manifest format:

```sh
python reports/solar-beam-v143/localize_beam.py \
  --manifest reports/solar-blind-v138/inference-inputs.json \
  --case-id blind138-00 --policies beam_active --field-budget 128 \
  --output runs/my-beam-output.json
```

The public output remains `pose: null`, with an uncalibrated rejection status.
Add `--research-output` explicitly to inspect native reference-to-query
correspondences, camera-to-world hypotheses, selected catalogue Sun and branch
traces. Input hashes are checked; query pose/depth/Sun metadata is rejected.
Agent-run lunar inference must use the separately admitted study controller,
rather than bypassing resource admission with this library or command.

## What the software checks establish

Generated checks verify native measurements and pose recovery, a retained
lower-ranked light/camera winning after acquisition, continued branch identities,
global lighting consistency, failed-search charging, missing coverage, blank
queries and query-truth isolation. The branch-switch control deliberately places
a correct camera in a generated proposal bank. It tests the acquisition
mechanism; it does not establish the image-only initializer's lunar coverage.
Generated random-pattern light labels are not a physical sunlight simulation.

The full repository suite, fresh generated smoke, source/input closure and
public checks are recorded in the verification artifacts when complete. No lunar
image has entered a new inference run during implementation. The latest measured
[unknown-light result remains 1/4 for every method](../solar-blind-v142/README.md).

## Scientific limits and the actual next test

The selection score is a first-order structural/geometric separation proxy.
Height envelopes do not rerender uncertain terrain or certify solar visibility.
Lighting is selected from a discrete reference catalogue; there is no continuous
Sun estimate, newly trained network or measured novelty/superiority claim.
The final branch ranking uses geometric support and image evidence, and can
still choose an incorrect pose. Raw outputs are never independent landmarks.

The prepared [complete experiment](NEXT_EXPERIMENT.md) retains all four positive
and four nonoverlap cases, seals predictions before scoring and independently
replays all 56 method/case cells. These are exposed development cases on three
positive terrain components, not fresh test geography. Original accepted-pose,
modern-comparison, transfer and final-evaluation targets remain unchanged.

All previous comparison allowances are fully charged under the published fixed
cap rules. The new study therefore needs the concrete additional resource
decision described in that protocol. No new allowance or private execution
authority is created by this implementation.
