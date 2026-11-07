# MOONSt3R same-case development pilot

This prospective eight-case pilot uses the same native 512-pixel image content, reference geometry, query intrinsics and AP3P estimator as v115. The four positives span four reference terrains and include two same-light and two changed-light pairs. Each has its certified matched nonoverlap. The selection uses only the published v103/v115 case metadata and the deterministic rule in the protocol. It does not depend on v115 outcomes.

The pilot adds a local raw accuracy observation for MOONSt3R. Read the same eight cases from the audited v115 result to compare raw correct poses, strict correct poses, wrong positive outputs, raw nonoverlap outputs and native correspondence precision. Keep all eight eligible cases, including errors. Local x86 timing and memory are separate from the ARM run. This small pilot does not establish a performance ranking, calibrated acceptance, a final-test result or an accuracy–speed tradeoff.

The locked native adapter keeps full square image content. It uses the official pretrained descriptors and reference-map AP3P. Query depth, pose and Sun are excluded from inference. Predicted MOONSt3R point maps are unused. The author loader's default crop differs from this common-input profile. The paper describes lunar adaptation in the south polar cap and the existing v73 audit places these reference terrains in the north. Exact checkpoint training membership and base-pretraining overlap are unverified. The existing local checkpoint is pinned and remains separate from released artifacts. [Author paper](https://arxiv.org/html/2510.18172) · [Official implementation](https://github.com/clementinegrethen/StereoLunar) · [Overlap audit](../moonst3r_overlap_disclosure_v73.json)

## Verify the preparation

Run from the repository root with the preserved Python 3.12.14 x86_64 environment. These commands require source and development metadata only.

```sh
PYTHONDONTWRITEBYTECODE=1 ../../work/lunar-clean-env/bin/python \
  reports/moonst3r-v121/prepare_v121.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=reports/moonst3r-v121:. \
  ../../work/lunar-clean-env/bin/python -m unittest -v test_v121_preparation
```

## Conditions for a later run

The controller and worker both require the successful first official v115 attempt, its original Actions artifact and the completed independent 480-row replay. They verify the original Actions ZIP against the REST size and SHA-256 captured by the exact v115 artifact verifier, then recheck its embedded watchdog and nested scientific ZIP. The local audited directory must be the external destination created by that verifier. It contains `artifact_audit_v115.json`, `github_v115_metadata.json`, the 480-cell inventory, the independent replay, the comparison report, and the exact nested result files.

A later immutable `reports/moonst3r-v121/dispatch.json` must be published with the protocol and sources on main. The preparation creates no dispatch and no reservation. The dispatch schema is `moonst3r_v121_local_dispatch`. It requires `execution_authorized` set to true, `protocol_sha256`, `max_calendar_seconds` equal to 1,800, and UTC `created_utc`, `latest_start_utc`, `absolute_deadline_utc`. The start lease is at most ten minutes. The absolute deadline is the latest start plus 1,800 seconds. The dispatch also requires `artifact_verifier_source_sha256` from the protocol, `v115_official_zip_sha256` from the official artifact's REST digest, and the following `v115_evidence` entries. Each entry contains `path` relative to the verifier's evidence directory and `sha256` of that file.

| Role | Path in the verified evidence directory |
|---|---|
| `official_run` | `github_v115_metadata.json` |
| `artifact_audit` | `artifact_audit_v115.json` |
| `summary` | `runs/solar-five-arm-v115/comparison/summary.json` |
| `seal` | `runs/solar-five-arm-v115/comparison/COMPLETE.json` |
| `replay` | `independent_replay_v115.json` |
| `comparison` | `matched_comparison_v115.json` |
| `nested_cell_inventory` | `nested_cell_inventory_v115.json` |
| `outer_watchdog` | `runs/solar-remote-five-arm-v115/<frozen nonce>/remote_watchdog.json` |
| `archive` | `runs/solar-remote-five-arm-v115/<frozen nonce>/scientific_results.zip` |

After the published dispatch, a separate authorized operation must charge the canonical baseline journal for `<comparison-protocol SHA-256>/moonst3r-v121` with 1,800 seconds and no refund. The controller reads that charge and checks remaining cumulative allowance. It does not reserve or modify the journal. The cap includes launch admission, repeated archive identity checks, checkpoint load and safe tensor digest, two fresh procedural control forwards, all eight inference rows and result sealing. The output path is exclusive. An incomplete attempt stays incomplete.

```sh
../../work/lunar-clean-env/bin/python reports/moonst3r-v121/run_v121.py run \
  --publication-commit PUBLIC_MAIN_COMMIT \
  --dependencies ../../work/modern-matching-deps \
  --audit-root /absolute/path/to/v115-verifier-output \
  --official-zip /absolute/path/to/original-actions-artifact.zip
```

The worker uses two CPU threads. It verifies the saved original pretrained gate and repeats the native known-transform, blank and invalid-input controls with the same resident model. The controller and worker refuse preparation-only use and refuse any partial or unaudited v115 result. This path has not been exercised with the live v115 artifacts or the model.

## Replay sealed results

After the controller writes the local completion seal, run the following separately. The v98 verifier reopens the saved raw arrays and independent source labels and recomputes the metrics without loading a model.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  ../../work/lunar-clean-env/bin/python reports/moonst3r-v121/run_v121.py replay \
  --publication-commit PUBLIC_MAIN_COMMIT \
  --output runs/moonst3r-v121/comparison/independent-replay.json
```
