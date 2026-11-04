**Repository integration note:** Dedicated modules, tests and prepared reports are integrated. The central registry/admission diff is applied alongside the native pretrained adapters. No scientific inference or asset construction has occurred.

# Historical compact and direct render comparison adapters

No lunar images, model forwards, ray queries, rendering, downloads or root-source edits were performed in this preparation task. Twelve procedural/stub controls pass. A read-only checkpoint-lineage replay verifies that all three named historical artifacts exist and match their preserved completion records.

The profiles are intentionally prepared-only/unavailable until their adapter source and actual native controls are published. No new scientific comparison or accuracy gain is claimed.

| Backend method | Exact historical identity | Inference |
|---|---|---|
| `historical_original_native512` | Original Tycho v1 CNN, checkpoint `ab555480…` | Original fully convolutional raw mutual descriptor extraction on native 512 pixels, then common PnP. It was trained/evaluated historically at 128 pixels; this is an explicitly new native-input control, requiring actual qualification. No old 2D rejection gate is transplanted. |
| `historical_frozen_surface` | Expanded compact initializer `c92f6917…` | Preserved coarse-128/native-512 matching with the original fixed two-iteration reference-surface normalization. |
| `historical_north_context` | Selected v61 attention control `ec7b9f77…` | Exact `cross_adaptation` architecture/checkpoint, preserved frozen native fine encoder and surface pipeline. Its historical result is 15/96 raw development poses; it has no calibrated acceptance result. |

`historical_compact_adapters_v73.py` exposes `validate_historical_profile`, `historical_profile_inputs`, and `make_historical_compact(profile)`. The adapter implements `.infer(InferenceInput)` and `.provenance`. Exact weight identities prevent relabeling another model as a historical control. Preserved algorithm hashes, completion seals and native-control prediction artifacts are independently checked. The runtime receives reference depth/pose and query intrinsics only, resets its feature cache for each call, and returns the evaluator’s common final PnP solution even when its internal geometric refinement selected another proposal. Old cached evaluation timings are never reused as new CPU latency.

Native-control artifacts must contain saved predictions on a predeclared analytic [8,−4] translation at native512, empty blank/invalid outputs, model/source/configuration hashes and a completed source-bound receipt. Their error is arithmetically replayed rather than trusting a `passed` boolean. The task does not fabricate those controls or execute the required forwards.

`direct_render_adapter_v73.py` exposes `validate_render_profile`, `render_profile_inputs`, and `make_direct_render(profile)`. This is the limited **reference-camera physical-relighting render-and-match** control. It matches SIFT features from each declared synthetic reference render to the query, ranks candidates without query truth, and estimates every candidate/final pose with the common PnP pipeline. It does not render candidate viewpoints.

`build_direct_render_library_v73.py` can inventory the existing timed reference-half renders after the complete96-unit ≥95% geometry gate and a published library protocol. It selects all24 predeclared lighting states for every eligible selection camera, never query-role renders. It hashes rather than decodes the images and starts no renderer. Reuse is allowed only when the primary map is the SAME full scene mesh; cropped or height-perturbed maps require a separately qualified library. Camera/geometry identities, source summaries and original preparation times remain bound. The inventory has a fixed1800-second two-thread allowance.

Primary inference searches only generic predeclared reference Sun entries. Secondary inference requires an exact supplied-Sun render in the reference ENU frame. An off-grid negative Sun raises `BaselineUnavailable`: it is an unfinished comparison, **not a correct mismatch rejection**. No nearest-Sun substitution occurs. Keep those failures in the case list and report comparison/nonoverlap coverage; a zero output count with unavailable cases is not robustness evidence.

Attributable reference rendering/centre-ray time and a conservative whole-camera-pair outer interval are retained separately from inventory and complete online image-I/O/SIFT/search/PnP time. The latter includes startup/I/O and unused query work; it is an upper bound, not a fabricated exact allocation. Missing measured outer costs make a library unavailable. Report the amortization count explicitly. Sharing the simulator/material between reference renders and simulated queries is a dependence; it does not verify optical repeat-image or orbital navigation accuracy.

## Parent integration

Combine `registry_integration_v73.patch` with the independent pretrained-adapter patch. Keep six explicit backend roles: `solar`, `sift`, `modern`, `native_pretrained`, `historical_compact`, `direct_render`. Root sources were not edited here.

The central admission must add both runtime adapter modules to its published source closure; call their profile validators before admitting scientific inference, and include `historical_profile_inputs` / `render_profile_inputs` in the global immutable input map. Preserve each profile’s nested source and artifact pins. Positive cases for both new backends require `scoring_geometry` because arbitrary native SIFT/CNN points need the shared depth-based scoring path.

Backend construction stores the adapter provenance. `.infer` delegates directly to the adapter; the existing independent common-PnP replay remains unchanged. Save `historical_control` and `render_search` diagnostics. Record unavailable status separately from algorithmic failure/rejection, including unavailable positive/negative counts and incomplete comparison flags in summaries/uncertainty tables. Do not admit an unavailable method as a completed fair comparison.

Prepared profile/protocol JSON files are templates. Refresh hashes after integration, obtain actual native control artifacts through a separately admitted bounded run, and publish the concrete frozen configuration before any lunar case. Library preparation remains blocked until full data/audit/reference assets exist. Repository checks and the common procedural smoke remain the parent’s integration responsibility.
