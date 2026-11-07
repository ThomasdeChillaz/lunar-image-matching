# Broader initialization fails: global correspondence quality is the bottleneck

The frozen surface pipeline has **54/72 correct raw poses and three wrong poses**.
A12pixel initial RANSAC proposal followed by the unchanged3pixel surface
refinement returns **54/72 correct and18 wrong poses**. All15 formerly missing
cases now yield an estimate; **none is correct**. Strict success stays16/72.
This control is rejected as an improvement. It does not change the default
pipeline or any historical result. Returning a pose is not reliable acceptance.

The rule preserves every historical estimated result. On missing cases only,
it fits all four orientation branches with12pixel RANSAC, chooses by geometric
support/residual and runs the same two reference-surface updates. A broad pose
is never returned unless at least one unchanged3pixel update succeeds. Unit
tests verify that distinction; the final estimator independently replays on
every case. Nevertheless, a coherent fit to incorrect image matches is wrong.
The selected broad proposals have rotation errors of about25–177degrees.

The fixed16case pilot also adds three wrong estimates and zero correct ones.
The expanded72case denominator is retained. These are dependent, exposed
development cases across eight geometries/four components. No new pixels from
calibration or final testing are used; no1% risk or real navigation claim.

A separate diagnostic scores all **60 coarse/broad branches on the15 missing
cases** after prediction. Even oracle branch choice finds **0/15 correct broad
poses**. The best genuine12pixel match fraction has median
**2.83%**; 7/15 cases contain
fewer than six genuine12pixel matches in every branch. This rules out branch
selection alone as a fix for these particular broad proposals. It does not
prove every better proposal algorithm impossible. Oracle counts are diagnosis
only and never choose an inference output. Native ground truth comes from
audited cameras/depth; darkness is not an occlusion label.

The next focused hypothesis is to improve global match discrimination before
local refinement. A small attention/context experiment is being prepared with
an exactly frozen starting score matrix and pointwise/self/cross controls.
Generic attention is not new architectural science; any contribution requires
actual learning gains, seed stability, a strong compact baseline and fresh
validation. The separate v13 fine-refinement cohort remains unchanged and live.

Verification:197 repository tests/four optional skips and a fresh100row
software smoke passed after the recovery kernel was added. Actual pretrained
quarter-turn/blank controls pass. Independent audits replay72 final3pixel fits,
the16case pilot and all60 broad fits, native point/pose errors, availability-only
routing, source/input/checkpoint hashes and summaries. Timing was concurrent
with training and supports no speed claim. No paid compute, human review or
automatic outreach.

Reproduce with `evaluate_pose_recovery_v14.py --full --output runs/new-recovery`,
then `verify_pose_recovery_v14.py --run runs/new-recovery --output runs/new-audit.json`.
For branch diagnosis use `diagnose_missing_poses_v14.py --output runs/new-branches`
and `verify_missing_poses_v14.py --run runs/new-branches --output runs/new-branch-audit.json`.
MoonAnything LunarGeo: Grethen et al., doi:10.5281/zenodo.18415119, CC BY4.0.
Own code/checkpoint MIT. This is a negative development finding.
