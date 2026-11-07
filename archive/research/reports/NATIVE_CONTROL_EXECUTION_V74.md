Native historical control qualification v74 -- prepared code, not measurements

Files:
  control_historical_native_v74.py
  test_control_historical_native_v74.py
  historical_control_settings_prepared_v74.json
  matchanything_two_thread_settings_prepared_v74.json

No actual texture, neural forward, download, rendering or scientific run was
executed. Tests use opaque stub inputs and small coordinate/pose arrays only.
Stub monitors are explicitly marked stub_test_only and cannot produce the
actual_pretrained_forward flag or qualified native-control evidence.

After the ACTUAL complete96-unit independent >=95% geometry gate passes:
1. Fill a separate settings copy for each exact existing method:
     historical_original_native512
     historical_frozen_surface
     historical_north_context
   Supply actual completed dataset/audit paths, a finite cap <=3600seconds,
   and a new output directory; retain canonical baseline accounting. The
   template is prepared_only and cannot execute.

2. From repository root, after integrating these source files:
   python control_historical_native_v74.py prepare --settings SETTINGS.json \
     --protocol reports/METHOD-native-control-protocol-v74.json
   This reads metadata/checkpoint hashes only, validates historical lineage,
   freezes all source/artifact inputs, and sets latestStart=created+finite
   publication lease; absoluteDeadline=latestStart+executioncap.

3. Publish the EXACT source and protocol. No code here commits or pushes.
   python control_historical_native_v74.py run \
     --protocol reports/METHOD-native-control-protocol-v74.json \
     --publication-commit FULL_PUBLIC_COMMIT --python VALIDATED_PYTHON
   Worker construction occurs only after full-data/source/canonical budget
   admission. It sets two Torch threads/one OpenCV thread and disables hub
   downloads. One external process watchdog bounds cold initialization, actual
   image generation, all controls and validation; failed caps are not refunded.

4. Only after actual success:
   python control_historical_native_v74.py qualify \
     --protocol reports/METHOD-native-control-protocol-v74.json \
     --publication-commit FULL_PUBLIC_COMMIT --output-profile QUALIFIED.json
   Qualification replays analytic pose/common reference geometry and invokes
   the unchanged validate_historical_profile. No available profile is emitted
   from an incomplete/failed run or a stub.

Actual control details:
  * Seed20261101, native512 texture, fixed analytic [8,-4] affine translation,
    flat10m reference depth, K(f=400,c=255.5), reference cam2world=I; analytic
    query camera translation[-.2,.1,0]. Truth is never matcher-fitted.
  * Cold and warm known-transform calls, blank, invalid reference support.
    This adapter has no mask input: invalid_support is all-invalid axial-depth
    support. Its actual public infer guard must reject BEFORE neural work.
  * The existing _HistoricalRuntime is constructed only after immutable model
    lineage validation. A control-only instance executes the exact existing
    HistoricalCompactAdapter.infer body; it bypasses only the circular demand
    for this first not-yet-existing native-control receipt. Lunar adapters keep
    their full original validator and receive no bypass switch.
  * Actual model module forward hooks record native descriptor and, for the
    selected context model, coarse descriptor/context-head execution. Merely
    requesting a model is insufficient. The known/blank/invalid p/q and poses
    are saved. Valid but inaccurate controls are retained as failed evidence.
  * Known median <=1 native pixel, common-pose rotation<=.5deg and translation
    <=.05m, >=6inliers, and no blank/invalid p/q/pose. All map/pose arithmetic is
    replayed; rejected controls do not become qualified baselines.
  * Output schema exactly matches historical_compact_adapters_v73:
    method/native_shape/cpu_threads/configuration_sha256/artifact_sha256/
    source_sha256/actual_pretrained_forward/query_truth_used/known_transform/
    blank/invalid_support, plus actual counters, timings and common-pose data.
    Receipt has complete/passed/method/protocol_sha256/outputs_sha256, plus
    outer source/deadline/canonical-charge proof. Final native_control holds
    protocol/outputs/receipt paths and SHA256each.

Minimal MatchAnything two-thread requalification:
  The existing control_native_matchanything_v17.py already supports
    --profile native512 --threads 2 --max-seconds CAP
  Its existing NativeMatchAnything implementation, safe local weights,
  dependency lock, analytic controls and four-file evidence are reused.
  No model integration or checkpoint conversion is duplicated here.

  Fill matchanything_two_thread_settings_prepared_v74.json with actual data
  audit, existing optional dependency directory, cap and new output, then use
  the same prepare/publish/run commands. The worker calls EXACTLY:
    control_native_matchanything_v17.run(child_output,dependency_dir,
                                         'native512',2,cap)
  Original four files are under OUTPUT/native/{protocol,summary,controls,
  provenance}.json. The wrapper additionally records the shared baseline
  charge and its external deadline. validate_native_profile must pass on those
  actual four files; no saved one-thread result is relabeled as two threads.

  Before future comparison admission, ALSO validate/pin the outer protocol
  and receipt using validate_receipt(protocol_path,receipt_path). A passing
  inner summary cannot override a failed/late outer watchdog. Preserve the
  native_pretrained pretrained_evidence exact four-file schema and place this
  outer proof separately in comparison input pins/profile metadata.

The original native control runner can be called directly with its supported
flags, but that alone does not add the shared canonical budget/source gate;
the prepared wrapper above supplies it. No existing renderer/live sources or
old deadlines are changed. No comparison or research superiority is inferred.

M74 follow-up: completed qualification and actual comparison binding
  admission() is metadata-only and does not enforce the CURRENT dispatch time,
  require another cap to remain available, reserve, or launch. Only run() uses
  the latest-start gate. Consequently a valid completed job can be qualified
  after its dispatch window closes, even if subsequent charged jobs exhaust
  the budget. No old lease/deadline or allowance is extended. Regression stops
  before a test qualification is written and confirms no new shared charge.

  comparison_patch/external_control_comparison_v74.diff is a minimal OPTIONAL
  outer-control integration into native_pretrained_adapters_v73.py and
  run_solar_evaluation_v70.py. The full patched snapshots are also supplied for
  review, not for overwriting unrelated parent edits. Apply the diff to root's
  latest source, then re-pin/publicize before any real control/comparison.

  Every NEW MatchAnything two-thread profile made from this wrapper must add:
    external_control = {
      'protocol': OUTER_PROTOCOL,
      'protocol_sha256': sha(OUTER_PROTOCOL),
      'receipt': OUTER_RECEIPT,
      'receipt_sha256': sha(OUTER_RECEIPT),
      'publication_commit': EXACT_OUTER_PUBLIC_COMMIT
    }
  Pass this dictionary as external_control=... to existing native_profile().
  The helper validates actual source publication, the successful outer receipt,
  canonical charge and the IDENTICAL four inner file paths/hashes. It performs
  no neural work or new admission charge. Native adapter construction invokes
  this gate before any model loading; actual comparison admission also requires
  each outer source hash in comparison.source_sha256 and the outer protocol /
  receipt hashes in comparison.input_sha256. No failed/late outer proof can
  fall back to the passing inner four-file evidence when this field is present.

  Existing prequalified profiles without external_control remain supported;
  the new MatchAnything profile builder must not omit the field. This optional
  integration does not invent an outer watchdog for old control attempts.
