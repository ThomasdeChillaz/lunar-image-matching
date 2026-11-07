# M75 remote CPU execution

Scope
This package does not dispatch or perform inference. The primary study remains
on the local Intel/Torch2.2.2 stratum. The optional standard ubuntu-24.04-arm,
two-thread Torch2.8 stratum compares the same frozen candidate and SIFT against
untouched LoMa-R/common and RoMa-v2/common. Compare latency only within each
hardware/runtime stratum. No MA/MOON runtime port, secondary track, training,
checkpoint selection, reserve access, chunked scientific run or automatic retry.

Install mapping
  solar_remote_relay_v75.py -> repository root
  solar-remote-cpu-v75.yml -> .github/workflows/solar-remote-cpu-v75.yml
  test_solar_remote_relay_v75.py -> repository root
  run_solar_evaluation_v70.patch -> merge into existing runner
The full runner copy is review-only. Merge the patch; preserve any newer edits.
Data companion: solar_capsule_v75.py and solar_portable_evidence_v75.py from the
separately delivered data package; their hashes must be in the source freeze.

Protocol construction after real artifacts exist
M74's MODES does NOT support modern_comparison. Do not call its builder with an
invented mode. A reviewed modern protocol must explicitly derive these fields
from actual evidence and remain byte-identical through reservation and worker:

  schema_version=70; builder_schema_version=74; prepared_only=false;
  purpose=cost_pilot (later modern_comparison); target_purpose=modern_comparison;
  split=model_selection; calibration_or_final_access=false; cpu_threads=2;
  tracks=[estimated_sun]; accepted_results=false; automatic_execution=false;
  source_publication_required=true; automatic_model_promotion=false;
  learning_contract/path+SHA; settings_path/path+SHA; manifest; geometry_audit;
  case_ids=ALL ordered12*N cases from full qualified manifest (N<=32);
  pilot_case_ids=at least4 distinct cases, two terrains, positive and nonoverlap;
  raw inventory: all original32 selection units and eight groups, independent
  eligible N, exclusions/reasons, eligible groups and explicit empty groups;
  compute_ledger/path+SHA; reservation_journal=runs/solar-compute-reservations-v67.json;
  baseline_admission={category:baseline,initial_reservations:exact current prefix,
    remaining_at_preparation_seconds:actual remaining,
    maximum_full_calendar_seconds:predeclared <=18000,refund_on_failure:false};
  reference_preparation_evidence={protocol,summary} and actual startup-inclusive
  reference_preparation_costs from the completed full-generic preparation;
  output=an unused repository-relative output directory;
  input_sha256=all actual case images/reference archives/scoring inputs/seals,
    settings, learning contract, raw geometry/audit history, selected checkpoints,
    actual study summary/history/watchdog evidence, actual native controls and
    provenance, selected official model weights, author source and cached DINO code,
    full physical prep evidence/portable export and all recursively required pins;
  source_sha256=the existing complete runtime closure plus modern_baselines_v2.py,
    this relay, exact workflow path, solar_capsule_v75.py and
    solar_portable_evidence_v75.py; case_cache_seals={};
  max_calendar_seconds=predeclared positive integer<=18000;
  created_utc, publication_dispatch_lease_seconds<=3600,
  latest_start_utc=created+lease, absolute_deadline_utc=latest+cap.

remote_cpu_stratum must equal exactly:
  {runner:ubuntu-24.04-arm,cpu_threads:2,torch:2.8.0+cpu,
   torchvision:0.23.0,torch_home:data/models/modern-v2-torch-cache}
Modern profiles require backend=modern, method=loma_r or romav2,
author_profile=common, source=data/models/<dedicated official source directory>,
pretrained_control=<actual passed native512/two-thread control JSON>,
pretrained_provenance=<its actual pinned source/weight/settings provenance>.
Candidate bridge profiles use backend=solar and the unchanged selected model
configuration/checkpoint, successful actual2000-update study summary+watchdog,
training_seed/training_step and exact history/checkpoint hashes. SIFT uses
backend=sift. Initialization-only candidates remain cost-pilot-only controls.

selected_weight_inputs must explicitly enumerate only the selected modern
weights below the frozen TORCH_HOME/hub/checkpoints directory, with their actual
locked SHA256. It must also be present in input_sha256. Preserve original source
revisions, default BF16 policy and native coordinate conversions. The source
setup helper restores git provenance by fetching only the two exact official
commits into dedicated source directories; no .git directory is transported.
All cached DINOv3 Python files must be in input_sha256. No path is rewritten.
Original Mac executable strings inside historical receipts are opaque provenance;
the remote primary loader never creates a Blender lookup when every generic
column is materialized. Missing computed columns fail BEFORE neural construction.

Full run after an actual ARM cost pilot
Keep source/profile/config/case/track/reference-preparation identities identical.
Require actual complete pilot summary, inner COMPLETE, reservation, watchdog,
execution identity, no FAILED/WORKER_FAILED, independently replayed row metrics,
and successful one-use remote_watchdog with actual workflow/run/ticket/source
bindings. Check the identical original local canonical charge and published
pilot source/protocol. Completed evidence remains valid after its old lease;
never call live admission on old results or compare ARM software to local Intel.
Use the RECORDED ARM pilot software for local full-protocol preparation. The
actual remote run still uses unchanged runner.admission, which requires CURRENT
ARM software to match its pilot. Freeze cost_pilot_summary/path+SHA,
cost_pilot_evidence={protocol,summary,metric_replay}, measured_cost_admission and
all actual pilot evidence hashes. Derive cap using existing measured_cost:
1.5*(measured model load+worst case time*ALL12*N cases), summed over profiles.
It must fit the predeclared maximum, current cumulative budget and18000s limit.
If it does not fit, report the explicitly predeclared pilot as a limited resource
result; do not drop cases/profiles, silently chunk, restart or extend a deadline.
The original full manifest remains frozen even when only the cost pilot runs.

One-use sequence (separate reviewed operations)
1. Publish the actual protocol and exact source closure; confirm the full96 gate.
2. pre_reserve(protocol,source_commit,authorization_path) locally charges exactly
   protocolSHA+/evaluation once in the existing canonical journal. Authorization
   contains the IDENTICAL entry including reserved_utc, plus exact post-charge
   snapshot bytes. Failure/ambiguity never refunds or permits reissue.
3. Data pack_capsule(...pre_reservation_authorization={path,sha256},
   portable_reference_evidence=...) includes that authorization. make_zip produces
   the immutable ZIP. Optional ordered public release chunks are each<=1.9GB;
   whole payload<=4GiB. Publish assets only to this public repository.
4. freeze_ticket(auth,capsule_manifest,ZIP,parts,ticket_path) freezes all hashes;
   parts is [{url,sha256,bytes},...] in concatenation order. URLs are restricted
   to this repository's public GitHub release assets. No custom token/secret.
5. Publish ticket at a new reviewed commit without changing scientific sources,
   and publish lightweight tag solar-ticket-<nonce> pointing to that exact commit.
6. Either dispatch_once(ticket,ticket_commit) for one explicit gh dispatch, OR
   prepare_browser_dispatch(ticket,ticket_commit) for one reviewed browser action.
   Browser preparation performs no gh/network/credential inspection, consumes the
   same exclusive DISPATCHED marker, and returns exact workflow/tag/input values.
   After any ambiguous network response/click, investigate the existing run;
   never repeat the dispatch or create a new ticket for the same charge.
7. Worker requires actual GitHub repository/event/run/source/tag/nonce bindings,
   unique historical run title, concurrency serialization and attempt1. It
   restores only the exact canonical snapshot in the fresh worker checkout;
   the existing runner consumes that entry without a second reserve().
8. Outer watchdog covers transport/source setup/inference and preserves failures.
   Original latest_start applies to ACTUAL remote controller launch; queue/setup
   cannot refresh it. No Mac monotonic value is transferred. Environment install
   consumes the fixed dispatch lease; the remote cap includes capsule/source setup.
9. Artifacts return remote_watchdog.json, controller.log and scientific_results.zip
   containing ACTUAL output rows/NPZs/execution/COMPLETE or failed partial results,
   worker logs, protocol/ticket/authorization. Inputs/model weights are excluded.
   Independently replay results on local assets, but NEVER import/replace the live
   local canonical journal or append worker-created reservations. Old completed
   snapshots are historical evidence, not a new launch allowance.

Status and validation
No actual source publication, capsule export, release asset, tag, dispatch,
model construction, neural forward, lunar measurement, ray or paid job was made.
35 focused procedural metadata/array/fake-process tests pass on final sources.
88 evaluator/guard/protocol/relay regressions passed with importlib mode before
the final two focused worker/loader guards were added (both independently pass).
These are software checks, never new lunar matching results or baseline wins.

Interpretation of the learning study
The twelve available training illuminations per physical point share one subsolar latitude. They do not identify an arbitrary 36-coefficient angular response. Degree-five functions restricted to this circle span at most eleven angular Fourier modes; the existing regularizer chooses an extension. The study can test changed-hour-angle interpolation, not establish learned subsolar-latitude extrapolation. No extra lighting data or training change is introduced here.
