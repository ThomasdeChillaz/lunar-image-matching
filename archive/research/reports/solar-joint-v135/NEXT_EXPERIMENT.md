# Proposed follow-up: acquire information after camera fitting

**Status: proposed, not authorized or run.** The original comparison envelope
is exhausted. This proposes a separate maximum of **40 CPU wall-minutes**, two
threads per serial job, no paid compute. Keep the original ledger and failures
unchanged; unused training allowance cannot be reassigned to comparisons.

## Mechanism

Keep the soft camera and its local competing hypotheses instead of immediately
discarding the faint case with five reliable correspondences. Select additional
reference anchors whose predicted native patch observations best distinguish
those cameras. Obtain each actual query measurement only after selecting it.
Update the camera evidence after each measurement. Test sensitivity to reference
terrain height perturbations of +/-5, +/-10 and +/-20 metres. Missing terrain or
off-image predictions supply no positive evidence. Brightness is never a
visibility label.

This is extra image acquisition, not a lower AP3P match count, looser reciprocal
threshold, fitted acceptance change or synthetic projection masquerading as a
landmark. Reference geometry, the cached predicted initializer and optional
supplied matching illumination are permitted; true query geometry is scoring-only.
No additional network weights are needed in this first mechanism test.

## Matched study

Use all original v134 cases: 16 positives and 16 certified nonoverlaps, on the same
three exposed terrain components. Preserve the nearly uniform failures and every
noise pairing. This is development evidence, not fresh geography or final testing.

Five fresh arms:

1. Direct with its original full reference library and unchanged settings.
2. Joint fit with the original 36 anchors and no extra measurements.
3. Joint fit plus 32 uniformly selected additional anchors.
4. Joint fit plus 32 appearance-ranked additional anchors.
5. Joint fit plus 32 ambiguity-reducing additional anchors.

The three acquisition arms have identical forward/reverse search caps, patch
settings and final six-match/three-pixel AP3P policy. Keep the twenty exact saved
v134 controls alongside the fresh outputs. Do not rank cached timings against
fresh times. Initializer computation must either be freshly included for every
new method or clearly excluded from all conditional-stage timing comparisons.

Report raw/strict recovery, wrong positive and nonoverlap outputs, native-pixel
precision, full stage costs and memory. Publish failure reasons and terrain-group
paired intervals. A preliminary positive signal needs at least two more correct
positives than the strongest matched completed control, no additional wrong
outputs versus Direct, and a gain over both equal-search acquisition controls.
This does not replace any of the fourteen full research requirements.

## Admission before inference

- Implement the policy and independent controls; publish source hashes, fixed
  cases, input bindings, seeds and metrics before lunar inference.
- Reserve at most 120 seconds for generated controls and cost measurement.
- Admit the full comparison only if its conservative forecast fits 1,800 seconds.
- Keep the remaining 480 seconds as a bounded recovery envelope, requiring a
  separately published diagnosis and prospective protocol before use. Failures
  remain charged; no quiet retries or deadline resets.
- A supplementary admission must record explicit user authorization and bind the
  exhausted original journal. It must not modify or bypass the original ledger.
- Calibration and final terrain remain closed. If the mechanism fails, publish
  the negative result without promoting a model or starting final inference.

After a positive result, propose a separately budgeted broader development study
before any final-test claim. No email or paper submission is part of this proposal.
