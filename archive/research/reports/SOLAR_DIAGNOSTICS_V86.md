# Lunar localization diagnostics — 4 November 2026

The shared-Sun matcher, learned lighting responses and active patch selection are implemented, and the SH and pooled models have each completed 2,000 updates. **No learned localization improvement is demonstrated.** Four calibration and twelve final terrain groups remain closed. The historical attention result remains 15/96 on different cases.

## What the new experiments establish

A source-published pose-free retrieval diagnostic completed in 154.77 seconds under its original 600-second cap. It uses actual pretrained encoders, one shared Sun direction and eight spatially distinct candidates per reference anchor. On a generated translation, 223/256 anchors have a candidate within 16 native pixels, exceeding the predeclared 80% software gate; 218/256 are within 1 pixel. Wrong-input-size and blank-image controls pass. These controls verify execution and matching under a known transform. They do not establish lunar camera accuracy.

Independent replay found that the original diagnostic scored its 256 sparse prediction anchors against the first 256 rows of a separate 676-anchor label inventory. **The original lunar counts of 5/1/0 are incorrect and superseded.** The correction joins labels by exact native reference coordinates, preserves every raw prediction, reproduces the scoring error and rejects missing or duplicate coordinates. The physical reference world points, normals and solar states use the correct 256-anchor grid. Scoring labels are opened after prediction; no query ground truth was found to enter inference.

Corrected candidate-window results on the single same-light pair, `u012`:

| Diagnostic arm | Candidate within 2 pixels / 256 | Candidate within 16 pixels / 256 |
|---|---:|---:|
| Frozen native descriptor control | 13 | 43 |
| 2,000-update directional response | 0 | 7 |
| 2,000-update pooled response | 0 | 8 |

Each count records whether any of eight candidates reaches the specified distance. It does not measure accepted correspondences or recovered poses. All 256 visible anchors remain in the denominator. Interpolating descriptors from a native image differs in sampling and scale from encoding canonical 33-pixel training patches. Conditioning on the reference view also does not synthesize the target viewpoint. These are possible failure mechanisms to test; their causal role has not been established. The dense diagnostic extracts features globally and therefore does not satisfy the main active policy's rule to defer native feature extraction until a patch is selected. It is not adopted as the main matcher.

## Matched camera-pose controls

The separate 32-row pilot and independent replay of its saved metrics are complete. Each method receives the same four positive pairs and four certified nonoverlap pairs; only two positive pairs change lighting. All positive failures remain in the denominator.

| Method | Raw correct poses / 4 | Changed-light correct / 2 | Raw nonoverlap poses / 4 |
|---|---:|---:|---:|
| SIFT, 2,048 features | 2 | 0 | 0 |
| Frozen compact, A/B/C off | 1 | 0 | 0 |
| Frozen compact, shared Sun + active selection, strict initialization | 1 | 0 | 0 |
| Same compact, quantization-aware initialization | 1 | 0 | 0 |

SIFT's two recovered cases both use unchanged lighting. No method returns a wrong positive pose in this pilot. These four-pair pilots do not establish calibrated acceptance, a 1% error bound, statistical significance or superiority over modern models. The shared Sun + active selection comparison changes two factors together and does not isolate either contribution. All eight learned lunar ablations remain required.

The measured worst-case admission rule projects 20,324 seconds for all 384 cases across four profiles, exceeding the unchanged 12,000-second maximum. The existing builder refuses the full stage before reserving compute. Cases, profiles, failures, the original maximum and prior charges remain unchanged. A pilot winner does not substitute for full checkpoint selection.

## Data and software verification

The image audit retains all 384 cases and 224 unique images. Of 192 positive cases, 17 trigger the current low-standard-deviation guard; all 17 contain nonconstant images. This gives a 91.15% coverage ceiling under the current software guard. It does not label physical observability or solar visibility.

The repository passes 1,063 tests, with 4 skipped. A separate fresh coordinate-join control correctly scores all 256 constructed translated anchors against a permuted 676-label inventory. That control uses coordinate arrays only and runs no neural inference. All 49 learning and 42 physical source bindings remain unchanged. Independent budget replay preserves failed jobs and their full charged caps. Remaining conservative allowances are 15,867 training seconds and 49,080 baseline seconds. There are no refunds, deadline extensions, final inference, emails or submissions.

Clean public [CI](https://github.com/ThomasdeChillaz/lunar-image-matching/actions/runs/41813736784) also passes 1,061 tests, with 6 platform skips, and the procedural benchmark. [Verification receipt](solar_ci_diagnostics_v86.json). These are software checks, not lunar accuracy measurements.

## Next scientific gate

The next test concerns how to represent an unknown viewpoint. A separate prototype outside the repository predicts a descriptor distribution conditioned on Sun and viewing direction, then marginalizes uncertainty in viewing direction before pose recovery. It has 95,116 total parameters and passes 31 toy tests of gradients and API constraints. Its teacher adapter uses the same full 128-pixel descriptor extraction definition for training and inference, preserves even-light-only supervision and retains every legacy camera and point record. The prototype is **untrained on lunar data: actual lunar feature-bank extraction, training and inference have not started**. It is outside the production matcher. Actual pretrained and known-transform qualification, an admitted cost pilot and matched pose evaluation must precede a performance claim.

The broader targets remain unchanged: at least 70% correctly accepted poses; a gain of 10 percentage points in accepted poses with a positive terrain-paired interval; empirical accepted-pose and nonoverlap errors of at most 1%, with uncertainty reported; and a measured accuracy or efficiency advantage against modern methods. No candidate supports a claim of superior performance to Ansar's team.

## Reproducible evidence

[Exact result table](solar_diagnostics_results_v86.json), [corrected label-association audit](solar_pose_free_corrected_metrics_v86.json), [matched metric replay](solar-preselection-compact-controls-cost-pilot-replay-v86.json), [full-stage refusal](solar_preselection_full_refusal_v86.json), and [complete artifact index](solar_diagnostics_artifacts_index_v86.json). [Download the 2.8 MB artifact capsule](solar_diagnostics_artifacts_v86.tar.gz), which includes all original diagnostic outputs, all 32 matched-control rows, histograms, manifests, protocols, receipts, corrected arrays and correction source. The original erroneous lunar metrics are preserved and explicitly superseded. Images and external weights are not bundled.
