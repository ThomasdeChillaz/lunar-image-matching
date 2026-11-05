# Shared-camera photometric consensus before native measurement

The completed v130 repair retains 17 matches but recovers 0/16 poses;
the matched Direct control recovers 4/16. The post-hoc geometry audit
shows seven to nine selected anchors within the native search tolerance
on the faint/bright example pairings. Recovering a few individual peaks
does not yet supply a stable camera estimate.

This study makes one camera explain both images across fixed reference
3D locations. Start from the original eight coarse cameras, optimize six
camera variables over blur scales 16/8/4/2 pixels using robust normalized
photometric residuals, and rank every hypothesis on the same reference
points. Samples stay fixed during optimization. Image-boundary penalties
prevent the optimizer from discarding difficult points. It uses reference
depth and both images, with no query geometry or query Sun input.

The optimized camera initializes the existing native reciprocal patch
measurement. Only actual measured correspondences reach the unchanged
AP3P estimator; projected optimizer locations never become measured
landmarks. Compare both uniform and parallax selection. Recompute all
five previous controls on the identical 16 positives and 16 certified
nonoverlaps, and require exact saved-coordinate/pose repetition.

No network parameters or training are added. This is an engineering
control for global geometry consistency; novelty and a learned-model
contribution are not established. Supplied matching illumination and
three exposed terrain groups limit the result. The near-uniform cases
remain in the denominator.

A source-published 60-second generated cost/control pilot precedes the
seven-method comparison, capped at 1,500 seconds from the same original
cumulative allowance. Forecast the optimizer's maximum iteration budget,
both new native patch policies and the complete original five-arm cost.
No old deadline, refusal, charge or result changes. A candidate passes
the development rule only with at least two additional correct poses
over Direct and no increase in wrong poses or nonoverlap outputs.
