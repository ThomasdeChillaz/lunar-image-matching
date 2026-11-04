# Saved SH cost-pilot diagnosis

The pilot exposes a proposal/pose bootstrap bottleneck. It does not yet demonstrate a learned accuracy gain. There are **four unique positive pairs**, evaluated at five checkpoints, rather than 20 independent positive tests. All checkpoints recover the same one pair, u012. The two changed-light pairs also use unseen-roll cameras, preventing a lighting-only causal conclusion.

Only saved scoring/prediction arrays and raw execution traces were read. All 40 saved-array hashes and the summary hash were verified. No images, rays, neural networks, new pose fitting or query-truth projections were used. The replay took under 3 seconds. Exact row evidence and hashes are in `diagnosis.json`.

| Pair | Saved visible anchors | Top8 candidate distance ≤1/2/4/8/12/16 native px | Native correct ≤2px | Initial live poses | Learned B measurements |
|---|---:|---|---:|---:|---:|
| u012, light 0→0 |256|3 /7 /23 /106 /148 /153|49/57 at step 0;40/99,40/99,42/95,40/99 subsequently|5/8|128|
| u026, light 0→17 |unavailable|No proposal arrays: early return|0 outputs|not run|not run|
| u060, light 16→16 |233|0 /3 /5 /18 /27 /30|3/107 visible returned;118 total outputs|0/8|0|
| u074, light 8→1 |225|0 /1 /1 /2 /4 /7|0/84 visible returned;109 total outputs|0/8|0|

The top8 distances use the best saved candidate for each visible anchor. Their denominator differs from the native-output precision denominator. A16px Euclidean disk is conservative relative to a 33×33 square search; it is not an exact refinement-support certificate. Saved coarse arrays have identical values, shapes and dtypes across all five checkpoints for the three cases that generated them.

- **u026:** the frontend returns `blank_or_uninformative_image`. The frozen guard tests whether either image has standard deviation below 1. No image was opened, and the trace does not identify which image triggered it. This failure stays in the positive denominator; no coarse recall can be inferred.
- **u060:** geometric overlap is substantial, but only 30/233 anchors have any top8 candidate within 16px;25 of 30 lie in the top-left quadrant. Twelve selected searches were within 16px; the final three ≤2px matches all occupy that quadrant and only two 4×4 grid cells. All eight beam assignment vectors are identical at every saved post-measurement update and at the end. The final 2000-iteration rescue fails.
- **u074:** only 7/225 anchors have any top8 candidate within 16px; six are top-right and one bottom-left. Descriptor candidate index 0 has none within 16px. The 128 selected searches contain zero candidates within 16px. Five unique assignment vectors after the first search differ by only 0.53% of anchors on average; the final two vectors differ at just one of 256 anchors. No native ≤2px matches survive; the final rescue fails.
- **u012:** wide candidate support enables pose initialization and response execution. It has eight distinct assignment vectors after the first search, with mean Hamming difference 12.3–12.5%; final diversity is five or six vectors. Native correspondences change with the response checkpoint, but raw pose correctness does not improve.

Every retained beam reports `initial_candidate_rank=0`; exact premeasurement assignment vectors were not saved. The earliest diversity figures above are **after the first measurement**, not initial-state reconstructions. The solver rank tag also differs from descriptor candidate index 0.

On u060/u074, all rounds retain zero live poses, response refresh calls are zero, and native prediction arrays are identical across checkpoints. Source inspection confirms that B returns ordinary appearance evidence when no trial pose exists. Thus longer response training alone cannot affect these recorded failure executions. A future, separately frozen development experiment should test genuinely diverse candidate/pose bootstrap and a pose-independent path into learned response scoring. This is a design implication, not a demonstrated improvement or novelty claim. Actual C policy/fallback remains unrecorded, so no active-selection execution claim is inferred from requested flags.

Summary SHA256: `c3e2ec236a8a5b5e516b1a8da181186193067445b166567fe193fd0779056120`. The full selection evaluation, matched ablations and uncertainty analysis remain necessary; this cost pilot selects no checkpoint.
