# Shared camera and lighting search: software prepared, clean lunar test pending

The last measured active-acquisition study remains **8/16 raw poses**, versus
7/16 for equal-budget uniform/geometry and4/16 Direct. That result used a
reference rendered at the query's matching illumination. Its one-pose selection
gain is uncertain and it is slower than Direct. No model is promoted.

This version removes the supplied matching-light input from the experimental
initializer. **No new lunar matching, pose fitting, training or rendering has run.**
The generated control below tests software wiring, not physical Sun estimation.
All 14 original research targets remain unchanged.

## Implemented algorithm

1. Accept a reference-only catalogue of up to 24 images with reference-world Sun
   labels, reference axial depth/pose/intrinsics, query image/intrinsics and an
   explicit reference coverage mask. True query pose, depth and Sun are absent
   from the inference signature. The library has no data-path parser or loader.
2. Build reference 3D points independently of brightness. Dark pixels remain
   valid observations. A common reference-photometry support mask across all
   catalogue entries prevents a light from winning by discarding inconvenient
   pixels. Unknown coverage remains unknown.
3. Share exact z-buffer projections across all lights for the frozen reference
   camera bank (at most 2,048 candidates). Rank camera/light pairs using positive-
   gain robust image loss on the same supported detector cells. Use full 3D
   geometry. Coarse ranking uses the historical fixed central-half detector
   region; downstream measurements retain native 512 content.
4. Keep at most two global catalogue lights and four cameras per light. Refine
   these eight hypotheses using the existing native photometric camera optimizer.
   One shared lighting hypothesis is selected by image evidence for the full
   match set. No per-anchor true light or query-derived reference image is added.
5. Run the unchanged no-extra and equal-budget native patch-acquisition policies
   from that same initialization. The original six-match, three-native-pixel
   final AP3P remains. Public outputs return `pose:null` until calibration; raw
   proposals require an explicit research flag.

Only reference geometry projections may be reused. Changing query pixels
recomputes all scores. Changing catalogue photometry recomputes lighting scores;
changing reference geometry or cameras invalidates projections. The new method
adds zero trainable parameters. It is a discrete template search, **not** the
complete continuous-Sun / learned-response / active A/B/C contribution. Catalogue
Sun proposals are not independent solar-visibility validation. Active image search
and physical render-and-match have substantial prior art; no novelty claim is made.

## Verification completed

Nine focused checks pass: known camera/pattern selection; score recomputation;
geometry-cache invalidation; common missing coverage; independent robust-loss
arithmetic; catalogue permutation; bounded seeds; invalid inputs/query-truth
arguments; and blank public rejection. A generated native 512 image-to-pose smoke
passes no-extra, uniform, geometry and active, with two additional searches per
policy. Pattern IDs do not simulate sunlight. Independent replay recomputes saved
pose and endpoint arithmetic without importing a matcher or pose estimator.

Use the validated environment from `requirements-lock.txt`, with two CPU threads:

```sh
python -m pytest -q reports/solar-blind-v138/test_catalogue_v138.py
python reports/solar-blind-v138/smoke_v138.py --output new-generated-smoke.json
python reports/solar-blind-v138/replay_smoke_v138.py new-generated-smoke.json --output new-generated-replay.json
```

These small software controls are not the full 24-light/2,048-camera/32-search
timing pilot. They create no scientific execution authority.

## Clean-data gate and modern comparison

The metadata/hash/header-only census verifies 104 completed dither-free RGB16
images: four reference cameras with 24 reference lights each, and eight query
exposures. The prepared diagnostic retains all four original positives and all
four nonoverlaps, on three positive terrain components. Negative endpoints span
six components. This is exposed development data, not new geography.

The larger historical modern comparison has 16 reference cameras across eight
development terrains. Only **three** of those cameras have a complete clean
catalogue; 13 need clean reference acquisition. Six of its 96 case IDs overlap the
small clean cohort. Its original counts (Direct 30/48, LoMa-R 15/48, RoMa v2 15/48)
remain unchanged but use the archived default-dither RGB8 inputs. Earlier work
already demonstrated image-coordinate dither matches on several archived dark
conditions. Do not reinterpret that table as a clean-image robustness result or
rank this new method against it without a matched rerun.

[Encoding mechanism](../SOLAR_DITHER_FINGERPRINT_V105.md) ·
[Dither-free control](../SOLAR_CLEAN_RADIANCE_RESULTS_V106.md) ·
[Exact current data census](data-preparation.json)

`inference-inputs.json` contains only permitted reference catalogue/geometry,
query images and intrinsics. `scoring-inputs.json` separately holds true query
geometry/Sun and outcome labels. File hashes are checked without decoding lunar
photometry. The four-unit clean cohort preserves its original renderer/exposure;
it is not the later black-space EXR experiment, so the 8/16 result cannot be carried
over as its baseline. Do not replace RGB16 with an RGB8 decoder silently.

## Next experiment and resource gate

The original comparison budget has 0 seconds left. The approved 40-minute supplement
has 1,920 seconds of fixed charges and 480 seconds held for recovery. The resumed
goal and existing authorization cover this next extra-patch comparison against
Direct and equal-search controls. The remaining recovery is assigned prospectively
under [the v139 controller](../solar-blind-v139/README.md), without another allowance
or refunded charge. [Diagnosis and scope](NEXT_EXPERIMENT.md).

Before new lunar inference: publish the full
controller and source/case/cost freeze, run a full-budget generated timing control,
and admit the comparison only if it fits its original deadline. The code and data
preparation here do not satisfy those execution gates. Calibration and final
terrain stay closed; no paid compute or automatic outreach.
