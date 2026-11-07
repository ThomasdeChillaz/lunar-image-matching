# Integrated solar matching research contract

Accepted by the user on 3 November 2026. Implement ALL three techniques and
controlled experiments as one matcher, regardless of earlier negative results.
Preserve all older measurements, failed attempts and immutable test reserves.
Novelty is a hypothesis requiring prior-art comparison and measured evidence.

## Data and resources

Native512px, fewer than100000total model parameters, two CPU threads per job,
no paid compute, no human review. Reconstruct cumulative training48h/baseline72h
consumption before admitting further training/comparisons. Record incomplete
accounting explicitly; unknown time is not zero. Renderer recovery is a separate
900s data-only attempt after original1800s failure, never a clock extension.

Complete v62data first: copy40complete pairs/480frames exactly (including failed
geometry), render8missing pairs/96new frames. Preserve unknown per-frame timing.
Require95%exact-correspondence common-estimator pose success before matching.

Use original16train/eightselection groups; four deterministic camera pairs each
(two motion/two roll), seed20261104. Sun hour angles -86.25..86.25deg step7.5,
24states at subsolar latitude+1.5deg, derived with lunar geometry. Train loaders
and teacher targets may read only even light indices. Frozen selection light
pairs: (0,0),(16,16),(0,17),(16,1),(8,17),(8,1). Preserve all eligible cases and
negative endpoints. Four calibration/twelvefinal groups remain inaccessible.
New views/light variants are not new independent terrain. Freeze sources,
schedule, eligibility, budget and source hashes before any new image inference.

Solar labels come from Sun rays, not brightness or camera visibility. Keep
camera occlusion, Sun shadow, certified clear and unknown separate. Clear rays
need coverage/terrain-envelope support; cropped or missing maps yield unknown.

## A: joint correspondence, pose and Sun inference

Frozen compact descriptor/native refinement. Deterministic256spatial anchors,
eight query candidates each. Solar state table from reference terrain, soft
image evidence from Otsu-centred sigmoid. Query Sun is ONE shared latent per
frame; initial hour-angle5deg and subsolar-latitude0.5deg within+/-1.5deg grid,
exclude below-horizon Sun. Keep eight diverse hypotheses and alternate assignment,
native refinement and pose estimation for three rounds with bounded Sun search.
Combine appearance, robust reprojection and solar likelihood; unknown adds no
support. Require32informative anchors in four regions for solar support.
4096minimal proposal samples total; existing final AP3P/RANSAC3nativepx stays.
Controls: solar term disabled with same candidates/budget, and intentionally
independent per-correspondence Sun choices. Do not report oracle masks/candidate
banks as usable image-only localization. Primary unknown-Sun and secondary
known-Sun tracks give all methods identical available metadata/reference map.

## B: learned lighting-response descriptors

Frozen original coarse encoder; train native patch encoder plus degree-five
real spherical-harmonic response head (36coefficients for each32Ddescriptor).
Condition on observed light, reference-surface geometry and trial viewing direction.
Canonicalize patches with reference geometry and trial pose identically at fit
and inference. No exact cyclic illumination-equivariance assumption. Predict
frozen descriptor features under other available TRAINING illuminations of the
same3Dpoint, plus correspondence discrimination. Held-out lights never become
teacher targets. Count all encoders/heads under100k. Compare capacity-matched
pooled-response control with identical training data and budget.

100update pilots with exact optimizer replay, then2000updates/seed20261101;
checkpoints0/500/1000/1500/2000. Select by all-case unfiltered pose success,
then2pxcorrespondence count, then earlier checkpoint. Repeat selected learned
configuration with20261102/20261103, preserving every result and checkpoint.

## C: active native patch selection

Use competing pose/Sun predictions to choose discriminating image patches;
require robustness to terrain-height perturbations+/-5,+/-10,+/-20m. Obtain
measurement, update hypotheses and choose again. Defer native feature extraction
until selected. Controls: uniform spatial, strongest appearance, geometry-only.
Budgets32/64/128queries, principal128. Time all feature/physical/search work.

## Evaluation and interfaces

Run all eight A/B/Ctoggles on identical cases/checkpoints, plus pooled response.
Baselines: frozen original CNN, strongest completed compact control, SIFT,
direct render-and-match, MatchAnything-EfficientLoFTR, LoMa-R,RoMav2,MOONSt3R
when actual pretrained controls/resource caps permit. Pin profiles; failures and
timeouts stay recorded. Raw/accepted/strict pose, native correspondence error,
wrong/nonoverlap outputs, latency/memory, lighting/view/terrain/seed strata and
paired terrain intervals; negative analysis accounts for BOTH terrain endpoints.
One fixed calibration recipe across methods, after model selection; no
threshold-selection gains attributed to new matching mechanisms.

Preserve match_terrain.py. Add localize_terrain.py with native images,
reference geometry/pose, both intrinsics, frozen config/checkpoint and optional
known Sun. No query true pose/depth in primary inference. Output native
reference-to-query correspondences, camera-to-world pose, inferred Sun,
acceptance/reason; rejected or uncalibrated public pose=null. Raw hypotheses
require explicit research output. Query truth is training/scoring-only.

Verify ray signs/self-offsets/coverage, frames/canonicalization, truth isolation,
shared Sun, blanks, insufficient/degenerate matches, checkpoint-sensitive caches,
exact pilot replay, independent saved metrics, full tests and fresh end-to-end
procedural smoke. Source/models/thresholds/cases frozen before final inference.
Preserve final reserve if development lacks meaningful recovery improvement.

Early milestone: >=10percentage-point raw pose improvement on matched development,
changed-light gains surviving map perturbation. Full targets unchanged: >=70%
correct accepted poses,>=10pp accepted gain with positive paired interval,
<=1%empirical accepted error/nonoverlap, competitive modern accuracy or efficiency.
No deployment-risk guarantee from small samples. Publish negative findings
honestly if needed; broader goal remains active. Release reproducible code/data
artifacts, brief, demo and paper draft. No automatic email or submission.
