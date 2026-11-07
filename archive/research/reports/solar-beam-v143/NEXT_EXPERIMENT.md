# Proposed 30-minute persistent-beam comparison

Status: implemented and prepared; **not authorized or launched**. No new ledger
or scientific reservation is issued by this document or the preparation command.

The original comparison envelope and both earlier supplements (2,400 and 900
seconds) are fully charged. They cannot be refunded or relabelled as training.
Request a separate allowance of **up to 1,800 local CPU wall-seconds (30 minutes)**
for the full cost pilot, eight-case comparison, scoring and replay. Use two
threads per serial scientific job, OpenCV one and no paid compute.

## Frozen experiment

Keep the same eight clean native512 RGB16 cases, with four positives and four
known nonoverlaps. The inference inputs, depth conventions, pose estimator,
three-pixel threshold and primary pose criterion remain unchanged. Query pose,
depth, Sun and historical predictions are unavailable to the prediction-phase
input accessor. Score only after sealing all eight predictions.

Run all seven arms:

| Arm | Purpose |
|---|---|
| Direct | Strong simple render-and-match reference |
| Frozen no-extra | Previous compact engineering control |
| Frozen active+32 | Previous selected compact policy |
| Single-hypothesis uniform | Stronger single-camera control with the new initializer and full field budget |
| Persistent uniform | Isolate retaining multiple camera/light branches |
| Persistent geometry | Geometry selection with the same branches and budget |
| Persistent active | Structural/geometry prediction separation with the same branches and budget |

All new arms use the same up-to-eight qualified camera/light alternatives and
common effective native field cap:
`floor(min(128, common_reference_candidate_capacity) / branches) * branches`. Each
branch-specific attempted correlation is charged. No free base field bank or
per-branch128 budget is allowed. Report actual counts even when coverage runs
out. The single-camera control intentionally keeps only the best branch; it
receives the full common field allowance, not the smaller beam anchor count.
All methods share the union reference candidate pool; the single-camera control
prioritizes its supported anchors and explicitly charges unavailable attempts.

Compare persistent active with the strongest non-beam arm, then separately with
persistent uniform/geometry. Report raw/strict pose recovery, wrong positives,
nonoverlap outputs, native 1/2/4/8/12-pixel precision, Sun proposal error, full
online latency and worker memory. Native endpoints, pose and estimator decisions
for all three frozen controls must repeat the saved v140 experiment exactly.
Independently reconstruct all 56 metric cells and every branch/field charge.
Use paired terrain-group intervals and both endpoints for negative pairs.

## Cost gate and execution safeguards

Before inference, publish the source/protocol hashes and verify their public
bytes. After actual user approval, bind one new private ledger to the frozen
protocol and all prior journal hashes. One 1,800-second reservation and original
deadline cover every stage; no failure refund, reset, smaller substitute or free
retry. The same serial scientific lease is inherited by each child worker.

First run both full-scope generated controls: flat and layered geometry, 24
reference lighting patterns, 2,048 camera proposals, every method and the full
128-field cap. They verify poses using actual native measurements; these are
not physical sunlight renders. Admit lunar inference only if
`1.5 * 8 * worst_complete_pair_seconds + 45` fits the original time remaining.
If the gate fails, preserve the pilot and stop before lunar inference.

The prior >=95% exact-correspondence geometry gate remains required. Keep all
failed attempts and every eligible case. Preserve four calibration and twelve
final terrain groups; no new final-test inference, model fitting or threshold
selection. Existing clean-input and v142 output capsules supply the dependencies.

## Decision and deliverables

A benefit means more correct raw poses than the strongest matched non-beam
control, without hiding wrong poses or nonoverlap outputs. A beam gain with no
active-versus-uniform gain supports retaining alternatives, not the active
selection score. A tie or failure is published. This small diagnostic does not
replace the original >=70% accepted-pose, >=10-point gain, uncertainty, 1% error,
modern accuracy/efficiency, seed and transfer requirements.

Publish the complete results, branch-survival traces, replay code, timing,
uncertainty and failures in a downloadable capsule. Update the requirement audit
without lowering any target. No email or paper is sent automatically.

Preparation is metadata-only:

```sh
python reports/solar-beam-v143/study_beam_v143.py freeze
```

Execution additionally requires a separately recorded actual user resource
decision, exact public source bytes and an unused private authorization identity.
