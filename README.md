# Compact lunar image matching

A reproducible study of a **28,272-parameter structural CNN**, calibrated rejection,
modern pretrained comparisons and failure under 3D/sensor transfer. Independent
project; no NASA/JPL affiliation or flight-readiness claim.

**Question:** Can a small matcher retain correct lunar alignments while rejecting
mismatches, with a useful accuracy–CPU-speed tradeoff?

[Release and downloads](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/compact-v2) · [Two-page brief](reports/compact_brief_v2.pdf) · [Measured report](reports/RESEARCH_ITERATION_6.md) · [Interpretation](reports/COMPACT_INTERPRETATION.md) · [Reproduction](COMPACT_REPRODUCTION.md)

## Measured findings

- **Primary frozen model:** 376/384 correct positive acceptances (97.92%), zero observed incorrect outputs and 0/288 nonoverlap acceptances. The other predefined seeds retain 351 and 380 correct positives; the latter accepts one nonoverlap pair. No best-of-test seed selection occurs.
- **Calibration drives most of the gain.** The old checkpoint retains 100/384 with its old gate and 368/384 with newly calibrated confidence. Both old and primary checkpoints have 384/384 raw correct proposals. All 284 correct proposals rejected by the old gate fail only its global-correlation threshold. The primary retraining gain over the recalibrated old checkpoint is small and not consistent across seeds.
- **Common comparison:** compact retains 35/36 correct positives versus MatchAnything's 32/36, with no observed incorrect acceptances for either. The paired gain interval is −2.78 to +19.44 points, so accuracy superiority remains inconclusive. The median paired CPU speedup is 101.2× for these specific profiles; the paired accuracy interval meets the declared −3-point margin. This is a conditional benchmark result, not global SOTA.
- **3D remains weak:** only 2/288 compact poses meet the secondary criterion. NASA POLAR transfer gives 23/40 acceptances within nominal geometry and one outside it. Reference drift/calibration uncertainty is unknown; orbital navigation accuracy remains unverified.

Inference uses **128px images** and a coarse-registration tolerance of **12 original 512px source pixels**. The final test contains 12 selected spatial components across three polar sectors, not 384 independent terrain tests. Trajectory IDs and Sun angles are absent; shared terrain/renderer dependence remains. Calibration has only 12 positives and 12 negatives across four components, so observed low error is not a deployment guarantee.

## Identical common-input comparison

36 positive and 36 known-nonoverlap cases across the same 12 final components.
Compact networks are lunar-trained; the large pretrained methods are not adapted
to this sample. Internal resolutions and CPU allocations differ and are recorded.

| Method | Raw correct positives | Correct accepted | Incorrect accepted | Nonoverlap accepted |
|---|---:|---:|---:|---:|
| Compact v2 / primary | 36/36 | 35/36 | 0 | 0/36 |
| Old CNN / new confidence | 36/36 | 34/36 | 0 | 0/36 |
| SIFT | 0/36 | 0/36 | 0 | 0/36 |
| LoMa-B / 504 | 28/36 | 27/36 | 1 | 0/36 |
| LoMa-R / 504 | 30/36 | 27/36 | 0 | 0/36 |
| RoMa v2 / fast512 | 34/36 | 31/36 | 0 | 0/36 |
| Original RoMa | 31/36 | 31/36 | 1 | 0/36 |
| MatchAnything–EfficientLoFTR / 4× | 36/36 | 32/36 | 0 | 0/36 |

LoMa-B and original RoMa exceed the declared 1% accepted-error target at their
frozen operating points. Those thresholds were not adjusted on this test.

![Accuracy and paired CPU runtime](reports/compact_accuracy_runtime_v2.png)

Each baseline is compared with its compact control on identical inputs in the
same CPU job. Absolute latency across different baseline allocations is not a
controlled hardware ranking. Memory, all three principal figures, complete
curves, per-region results and ablations are in the measured report. RoMa v2's
precise author-profile control exceeded its ten-minute CPU budget; the table
uses the explicitly declared fast512 profile. No model was silently substituted.

## View or reproduce

For viewing, download `compact-v2-report.zip` from the release, extract it and
open `compact_research_v2.html`. It includes an interactive success, rejected
mismatch and remaining failure. No landmark review or model download is needed.

For saved-metric replay, install the pinned environment and download the model,
data and experiment ZIPs plus their checksum sidecars. Follow
[COMPACT_REPRODUCTION.md](COMPACT_REPRODUCTION.md). The experiment capsule digest is
in [the checksum file](reports/compact_experiment_assets_v2.sha256). Saved replay
checks source/data/model hashes, geometry, confidence fitting/calibration,
rejection behavior, summary tables, intervals and paired CPU measurements.

Select the new model explicitly after installing the released assets:

```sh
python match_terrain.py --model-version compact-v2 --model-dir runs/compact-v2-expanded-20261030 --protocol reports/compact_freeze_v2.json --reference reference128.png --query query128.png --output runs/alignment-new.json
```

Coordinates map reference pixel centers to query pixel centers. Rejection returns
`transform: null`; raw proposals require `--include-proposal`. The default v1
interface and all earlier results remain preserved.

## Provenance and limits

- [Public freeze proof](reports/compact_publication_v2.json): commit `392cdf1020c925cec61d3340a7720e26cc6cdce7` preceded all final inference. The protocol binds 29 matching/data/metric source files, analysis source, three checkpoints, ten confidence policies and every case list.
- [Independent replay](reports/compact_replay_audit_v2.json) and [table/interval/CPU-control audit](reports/compact_table_audit_v2.json): 13 policies, 14 aggregate tables, 10 interval checks and 360 paired controls. All selected final baseline shards completed; redundant recovery jobs were stopped and retained as partial records.
- [Old-gate diagnosis](reports/compact_gate_diagnosis_v2.json) and [development oracle geometry check](reports/compact_depth_oracle_control_v2.json) are post-study diagnostics; neither changes the frozen outcomes.
- [Data limitations](reports/compact_data_limitations_v2.json), [execution record](reports/compact_execution_status_v2.json), [selection provenance](reports/compact_final_execution_selection_v2.json), [licenses and attribution](THIRD_PARTY.md).

Validation: **128 tests passed, four optional tests skipped**, a fresh controlled
smoke and actual public-freeze matching-command controls. No paid compute,
human landmark review or automatic outreach. The study is complete; broader
trajectory independence, real navigation accuracy and 3D improvement remain
research gaps. Consumed final cases cannot become a new tuning reserve.

<details>
<summary>Earlier benchmarks and pilot documentation (preserved)</summary>

A reproducible research project comparing classical and pretrained learned image matching on real lunar repeat observations and controlled image warps.

**Question:** How do matching methods behave when illumination changes on the same lunar terrain?

**Earlier research iterations:** [protocol and hypothesis ledger](RESEARCH_ITERATIONS.md), [validation hardening report](reports/RESEARCH_ITERATION_5.md). The development comparison completed: our standalone CNN **1,072/1,152 (93.06%)**, original RoMa **1,032 (89.58%)**, tested MatchAnything–ELoFTR4× **995 (86.37%)**, the baseline ensemble **1,083**, and the ensemble plus our CNN **1,103 (95.75%)**. These are development findings with different training/compute budgets, not current-SOTA conclusions. The [preregistered reserve protocol](reports/reserve_protocol_v1.json) fixes the standalone model, strict gate and all 10,944 cases across 12 footprint/acquisition groups before reserve access. Both reserves are now consumed by that evaluation and must never be reused as fresh data. The13-predictor ensemble remains development-only. Earlier reports preserve negative results and historical protocols.

**Rejection is part of the output:** `match_terrain.py` returns `transform: null` when its frozen rule rejects a proposal. Ungated proposals are hidden unless explicitly requested. Real-photo controls retained334/336 correct known-warps and rejected84/84 different-scene pairs without real-data calibration. However, the NASA laboratory relighting test exposed a transfer failure: the strict rule rejected all 8 fixed-camera pairs, including4 nominally correct raw fits. These distinct experiments do not establish orbital navigation accuracy.

**Audited reserve result:** raw CNN correct on **8,414/9,216 (91.30%)**; the unchanged strict gate correctly accepts **4,603/9,216 (49.95%)**, with **zero observed wrong outputs and 0/1,728 accepted nonoverlap pairs**. All 32,832 method outcomes were replayed. The price is lower coverage: 3,811 correct raw proposals are discarded. These are 12 grouped components within Tycho; they do not establish universal reliability or fresh performance of the full ensemble.

The project includes checksummed data acquisition, masked geospatial loading, NCC and SIFT baselines, pinned DISK + LightGlue inference, frozen region splits, quantitative evaluation, plots, an interactive offline viewer, a two-page PDF report, and an independent-checkpoint review tool.

**Current scope (29 October 2026):** no human landmark review is required or planned. Accuracy comparisons use known transforms and physically rendered changing-light pairs. Real repeat observations support fit rates, consistency and provisional map-agreement diagnostics only; real navigation accuracy and real-data correctness calibration are unverified. Review tools remain optional for a future validation study.

**Scientific status:** repeat-observation measurements use the publisher's map coordinates. Their accuracy is unknown. These are map-reference diagnostics; independent checkpoint validation remains open. Exact-transform controls provide a separate correctness check. See [STATUS.md](STATUS.md) and [DATASET_CARD.md](DATASET_CARD.md). No NASA/JPL affiliation or endorsement is implied.

**Research extension:** [measured report](reports/EXTENSION_REPORT.md). The full 321-pair real-image benchmark and original CNSFM common-grid comparison have executed, alongside 1,152 physical-render evaluations and 108 method measurements on 27 new real repeat pairs across five Tycho areas. A small lunar descriptor adapter improved held-out synthetic registration from 93/128 to 98/128 pairs; its development-selected rejection policy still accepted three incorrect held-out alignments. Independent real reference review, real-data rejection calibration and verified real-image improvements remain open.

The adapter does **not** currently establish a real-image improvement: on polar S3 it returns 99/231 alignments versus 158/231 for the frozen matcher. The frozen matcher remains the default; the adapter is experimental. Map disagreement among returned fits cannot establish true accuracy.

## Install and run

Python **3.12** is the verified environment. From this repository:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python reproduce.py --output runs/reproduction --test
```

The command downloads 15 source images (about 26 MB) and two pretrained weight files (about 52 MB), verifies their hashes, runs the test suite including actual pretrained inference, and evaluates all three regions. Original source rasters and weights are excluded from Git. CPU execution may take several minutes. Every run requires a fresh output directory.

For the small procedural software check:

```sh
python benchmark.py --config smoke.json --output runs/smoke-new
```

For individual stages:

```sh
python dataset.py fetch
python fetch_models.py
python evaluate.py --split test --output runs/held-out
```

Use `python reproduce.py --offline --output runs/offline` after assets and weights are cached. Missing or corrupt files produce explicit errors; learned matching never silently falls back to a classical method.

## Inspect results

The read-only [current validation report](reports/validation_hardening.html) distinguishes fresh physical-render evaluation, known warps of real orbital photos, repeat-observation diagnostics, and external laser-reference laboratory measurements. It requires no landmark review.

For the standalone gated matcher, first extract the published training capsule so `runs/contrastive-terrain-train-v1` exists, then run from this repository:

```sh
python match_terrain.py --reference reference128.png --query query128.png --reference-mask reference_valid.png --query-mask query_valid.png --output runs/my-alignment.json
```

Images and optional validity masks must be128×128 single-channel uint8. The frozen sources and checkpoint are verified before inference. A rejected result contains no usable transform; `--include-proposal` is an explicitly ungated research diagnostic. This rule has empirical evaluation results, not a deployment safety guarantee.

The published pilot artifacts use the `lunar_` prefix: [report](lunar_REPORT.md), [PDF](lunar_REPORT.pdf), [per-pair measurements](lunar_results.csv), [summary](lunar_summary.json), and [environment/source hashes](lunar_environment.json).

Open `lunar_demo.html` locally for an interactive pair/method selector and overlay slider. GitHub displays HTML source, so download the file or serve this directory locally:

```sh
python -m http.server 8000 --bind 127.0.0.1
```

Then visit `http://127.0.0.1:8000/lunar_demo.html`. For a new run, open `runs/reproduction/demo.html` instead.

![Held-out reference agreement](lunar_agreement.png)

## Optional independent checkpoint review

The updated [source-bound review packet](reports/landmark_review.html) supports zoom, named landmarks, separate reference/query uncertainty and unscorable cases. This optional validation route requires human-selected landmarks; it contains no suggested coordinates. It is not required for the current project scope. Generate a fresh packet and import a completed review:

```sh
python landmark_review.py --output runs/manual-review.html
python import_review.py review.json --output reviewed_manifest.json
```

After importing actual human review, convert only its corresponding pairs to the expanded grids using identical source image hashes and known map affines, then score saved predictions:

```bash
python rebase_review.py reviewed_manifest.json --output reviewed_expanded_manifest.json
python score_independent.py --reviewed-manifest reviewed_expanded_manifest.json --runs runs/expanded-confidence-v1 runs/cnsfm-common512-v1 runs/cnsfm-real-transfer-v1 --output runs/independent-new
```

The conversion adds an explicit conservative interpolation allowance and retains second-review uncertainty. It never uses a matcher's estimated transform. Only reviewed pairs gain independent scores; incomplete pairs remain unknown. Development labels select policies before held-out checkpoint errors are computed; insufficient labels produce an explicit unavailable policy.

Add `--second-review review-b.json` when an independent second reviewer is available. Keep a second review blind to the first reviewer’s coordinates. The importer retains disagreement and raises the uncertainty estimate. Re-evaluate on the reviewed manifest; software validation alone does not establish human expertise or navigation accuracy.

The original simpler interface remains available:

```sh
python checkpoint_review.py --output runs/checkpoint_review.html
```

Open that page locally, identify at least ten distributed stable landmarks per pair, enter reviewer and picking uncertainty, and export `checkpoints.json`. Avoid using shadow edges. Import into a new manifest and rerun:

```sh
python apply_checkpoints.py checkpoints.json --output reviewed_manifest.json
python evaluate.py --manifest reviewed_manifest.json --output runs/reviewed
```

Imported annotations are recorded as independent reviewer input; the tool does not claim they are automatically verified. A second independent review and explicit uncertainty assessment are appropriate before navigation-accuracy claims.

## Data and methods

The subset uses five LROC NAC observations from each published CNSFM scene: Chang'e 3 (S1 reserved), Apollo 11 (S2 development), Peak Near Shackleton (S3 held-out). All source observations remain within one split. No training or parameter search is performed. Three exact-transform controls per region are evaluated separately from the twelve real repeat pairs.

- **NCC:** central template, translation-only; rotation is labeled out-of-model.
- **SIFT:** ratio-filtered features and partial-affine RANSAC, up to 2000 features.
- **DISK + LightGlue:** pretrained depth-supervised DISK features (512 maximum), learned matching and the same robust geometry estimator.
- **Ablation:** each method with and without CLAHE.

Transforms map reference `(x,y)` pixel centers to query `(x,y)`. Failures stay in denominators. Endpoint errors use valid evaluation points and independent targets when provided. Timings exclude downloads/model loading, use warmup, and synchronize the selected device. Different feature budgets and a single timing repeat limit speed comparisons.

The shared map grid is an external reference, not guaranteed subpixel truth. Gain/gamma perturbations do not reproduce moving lunar shadows. The original pilot uses no physical rendering or additional DEM, and no reproduction of LIMA is claimed. The extension uses externally rendered MoonAnything data as a separate experiment.

## Reproduce the research extension

The committed manifests are frozen before matching outcomes. Entire source scenes/acquisitions stay in one split. S3 was inspected in the pilot and is therefore exploratory for new choices.

```sh
python dataset.py fetch --manifest expanded_solar_manifest.json
python evaluate_extended.py --output runs/expanded-new
python fetch_moonanything.py
python moon_dataset.py --output moon_rebuilt_manifest.json
python train_lunar_adapter.py --manifest moon_rebuilt_manifest.json --output runs/adapter-new
python evaluate_moon.py --manifest moon_rebuilt_manifest.json --adapter runs/adapter-new/adapter.json --output runs/physical-new
python verify_extension.py --extended-run runs/expanded-new --physical-run runs/physical-new --output runs/extension-audit-new.json
```

To reproduce the real appearance subset and frozen adapter transfer (an additional 4.68 GB source archive):

```bash
python fetch_moonanything.py --members render.zip
python tycho_dataset.py --output tycho_rebuilt_manifest.json
python evaluate_transfer.py --dataset tycho --manifest tycho_rebuilt_manifest.json --output runs/tycho-transfer-new
python evaluate_transfer.py --dataset cnsfm --manifest expanded_solar_manifest.json --output runs/cnsfm-transfer-new
python verify_transfer.py runs/tycho-transfer-new --output runs/tycho-transfer-audit.json
```

The committed `tycho_real_manifest.json` freezes selection before outcomes. The new 128 px common grids use shared metadata intersections of original 128 px real orthocrops; their errors are not directly comparable to the 512 px scene errors or independently known physical render errors. No additional model fitting uses real test observations. After downloading the ten full Windows artifacts into separate directories under one execution root, prepare the exact common-grid inputs locally and audit every original output:

```bash
python prepare_cnsfm_common.py --output data/cnsfm/common512 --fetch
python score_cnsfm.py --executions PATH --output runs/cnsfm-common-new
```

The staging command verifies pixels and masks against the committed detector predictions and refuses to overwrite an existing staging folder. Use the matching original manifest and predictions when auditing the published artifacts.

MoonAnything acquisition downloads the metadata and physical Hapke render subarchives (about 0.9 GB), not the outer 11 GB ZIP. Byte-range retries and persistent hash-checked range caches support interrupted transfers. The real-appearance `render.zip` is an additional optional 4.7 GB download for expanding real repeat-observation regions. All downloaded archives, selected source rasters and caches remain outside Git. The report publishes measurement tables and a small trained descriptor projection.

The physical protocol uses 48 training, 16 development and 16 test patches from distinct acquisitions in separated Tycho longitude bands. Its 128 px rendered supervision stays separate from real 512 px map-reference measurements. All-zero physical renders remain explicit evaluation cases. The detector/backbone and LightGlue are frozen during descriptor adaptation. Threshold selection uses independently labelled development outcomes; provisional map disagreement is never a correctness label.

The original highest-Sun anchor protocol covers only 21–78° of held-out Sun separation and couples light index to rotation. The report now shows correctness versus full Sun-direction change and raw shared non-dark terrain, including failures and support-threshold sensitivity. These are descriptive strata, not isolated causal effects or geometric shadow labels. Recompute them from the saved results with `python physical_lighting_analysis.py --run runs/physical-adaptation-v1 --output reports`.

The all-Sun-pair follow-up holds geometry constant and retains the original model and development policies:

```bash
python evaluate_crosslighting.py --output runs/crosslighting-new
python verify_crosslighting.py runs/crosslighting-new --output runs/crosslighting-audit.json
python report_crosslighting.py --run runs/crosslighting-new --audit runs/crosslighting-audit.json --output reports
```

This includes NCC as an applicable translation baseline, SIFT and the unchanged learned variants. Every pair of the nine lighting renders is included. Previously inspected geographic test terrain is explicitly exploratory; no new tuning uses the follow-up outcomes. Resume an interrupted run only with `--resume` and identical frozen inputs/source.

The [completed fixed-geometry report](reports/CROSS_LIGHTING_REPORT.md) covers 576 pairs and 2,880 method evaluations, including Sun separations above 90°. NCC is correct on 344/576 pairs, frozen learned matching on 303/576 and the adapter on 323/576. The original learned rejection policy performs poorly on this design (five correct fits retained, seven incorrect fits accepted); it remains experimental. Its translation limit is below the known imposed displacement. These results do not overturn the lack of independently verified real-image improvement. Physical uncertainty intervals now group overlapping footprints and shared acquisitions, rather than treating every acquisition as independent terrain.

The original CNSFM Windows probe is verified in all three scenes. The [Windows workflow](.github/workflows/cnsfm-original.yml) supports a native probe, a common-grid probe, a full common-grid comparison, and an optional full native-resolution run. Common-grid inputs are verified byte-for-byte against the detector’s image/mask arrays, and forward/reverse matching are independent executions. Shared RANSAC scoring requires six inliers; that differs from the paper’s success definition. The original binary and model weights are retrieved separately from the pinned author repository.

To regenerate the optional crater predictions, install [requirements-detector.txt](requirements-detector.txt) in a separate environment and obtain the documented source revisions:

```sh
mkdir -p data/cnsfm
git clone https://github.com/Bin501/CNSFM.git data/cnsfm/upstream
git -C data/cnsfm/upstream checkout 94cebaaa7fd3e070f8040620975554c3067651ef
git clone https://github.com/WongKinYiu/yolov9.git data/cnsfm/yolov9
git -C data/cnsfm/yolov9 checkout 5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff
python cnsfm_detector.py --yolo-source data/cnsfm/yolov9 --weights data/cnsfm/upstream/CDAYOLOV9/YOLOv9Best.pt --output crater_predictions_new.json
```

The optional YOLOv9 integration retains GPL-3.0 terms. Its generated crater coordinates are method predictions and are never imported as independent reference landmarks. Native author patch tiling is not reproduced by the single whole-image common-grid detector pass; do not compare percentages directly with the paper.

## Repository guide

The earlier [development research report](reports/RESEARCH_ITERATION_2.md) records separate translation/rotation/scale/combined results, the image-selected hybrid and conditional confidence fallback. Its975/1,152 hybrid successes and857 strict retained-correct cascade results remain development findings. The newer reserve experiment consumes both frozen reserves; none may be described as untouched for a future iteration. Reports include measured figures, full outcome capsules, source snapshots, licenses and audit/reproduction commands. Regenerate the earlier development figures with `python report_research_development.py`.

[AI_PLAN.md](AI_PLAN.md) defines the agent workflow; [MILESTONES.md](MILESTONES.md) preserves acceptance criteria; [STATUS.md](STATUS.md) distinguishes implementation from scientific validation. [THIRD_PARTY.md](THIRD_PARTY.md) records source/model notices. Original code is MIT licensed except the explicitly marked optional GPL-3.0 detector integration; third-party data and weights retain their own terms.

References: [CNSFM paper](https://doi.org/10.3390/rs17132302), [source dataset](https://github.com/Bin501/CNSFM), [LightGlue](https://arxiv.org/abs/2306.13643), [DISK](https://arxiv.org/abs/2006.13566), [Adnan Ansar's public research](https://www-robotics.jpl.nasa.gov/who-we-are/people/adnan_ansar/).


</details>
