# Native modern comparisons: completed, scoped development evidence

All72 assigned pairs for LoMa-R and RoMa-v2 are complete and independently replayed, including their same-job original compact controls. The fixed positional split uses first36 from the original run and last36 from the declared range worker; all duplicate and partial attempts remain preserved. Both original jobs timed out at64/72, so a successful workflow alone was never treated as complete coverage.

| Method | Correct raw /72 | Strict /72 | Wrong raw | Median full inference |
|---|---:|---:|---:|---:|
| loma_r common | 42 | 14 | 28 | 54.25s |
| romav2 common | 53 | 17 | 19 | 53.30s |
| Frozen orientation CNN |45|17|12|2.64s /3.17s in paired workers|
| Frozen CNN + reference-surface rectification |54|16|3|Not measured in same worker yet|
| Same SIFT-first + surface pipeline |57|25|2|Not measured in same worker yet|

The original compact control is20.6× faster than LoMa-R and16.8× faster than RoMa-v2 by median per-case paired ratios. **Those speedups do not describe the new surface pipeline.** Its direct same-worker timing/full-score workflow is prepared separately. Model initialization, downloads and reference scoring are outside inference timing. Joint memory includes both loaded models and cannot rank isolated model memory.

These are exact settings, not every author's best configuration: LoMa-R receives the full512pixel field of view internally resized to504; RoMa-v2 uses the declared fast/common512pixel, single-direction profile. Original compact matching uses128pixel coarse proposals and512pixel frozen features. Author-recommended precise RoMa-v2 native inference is not completed, and native EfficientLoFTR/MOONSt3R comparisons remain outstanding. Source revisions, weight hashes and settings are pinned in each artifact. All models passed actual512pixel known-translation pose/point, blank and invalid-mask controls before lunar inference.

All methods receive the same reference geometry and final3native-pixel AP3P estimator. Surface matching additionally uses its image-estimated pose for explicit reference-surface normalization; query depth/true pose are excluded from inference. Its accuracy count exceeds the tested RoMa-v2 profile by one pair, which does **not** establish superiority. The per-method descriptive terrain intervals are in native_modern_summary_v10.json. Only four exposed terrain components/eight geometries generate these72 dependent pairs. No fresh-test confidence or1% risk guarantee follows. The compact CNN has lunar training while the compared large checkpoints remain generic/frozen; this does not isolate architectural advantage.

Raw outputs are not calibrated acceptance. The surface+SIFT pipeline still has two wrong proposals; native nonoverlap and confidence calibration are incomplete. Stricter surface+SIFT success25/72 is below the earlier fallback27/72. No global SOTA or independently verified orbital accuracy is claimed. Every claim is confined to these cases, settings and saved native coordinates.

Replay each method with combine_native_baselines_v8.py using the original and range artifact directories. Then run this reporter with --artifacts pointing to their four extracted ZIP folders. Own code is MIT; MoonAnything LunarGeo data are CC BY4.0 (Grethen et al., doi:10.5281/zenodo.18415119). Upstream model licenses/weights remain governed by their pinned author notices; no third-party weights are redistributed here.
