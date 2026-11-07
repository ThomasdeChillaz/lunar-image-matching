M74 artifact-driven evaluation preparation (future integration; no jobs run)

Integrate the two new builder modules, the physical-reference M74 package,
and the proposed run_solar_evaluation_v70.py guard/cost patch before use.
The existing native-admission fixture receives a small test-only update for
schema74. The actual matching/training sources remain owned by the root agent.
Run from the repository root with its validated Python environment.

1. Copy solar-evaluation-settings-template-v74.json to a NEW settings file.
   Keep prepared_only=true until the actual qualified case manifest, full96
   audit, completed2000-update learning contract/studies, and externally watched
   physical preparation evidence exist. Fill finite pilot/maximum-full caps,
   explicit positive/nonoverlap pilot IDs from two terrains, and a new output
   directory. The canonical ledger/journal come from the learning contract.

2. Set prepared_only=false only with those real artifacts. Prepare a concrete
   cost protocol, without executing it:

   python build_solar_evaluation_protocol_v74.py prepare --settings reports/ACTUAL-settings.json --phase cost_pilot --output reports/ACTUAL-cost-protocol.json

   This retains all independently eligible N cameras from the original32,
   all eight raw terrain groups and exclusions, and exactly12*N cases. Pose
   control failures never become eligibility exclusions. N=32 gives384 cases;
   N=31 gives372. Groups without eligible cameras remain explicit metadata.

3. Separately publish the exact protocol and its dynamically collected sources.
   The existing parent workflow must perform and verify remote publication.
   The read-only local Git byte barrier is:

   python build_solar_evaluation_protocol_v74.py verify-publication --protocol reports/ACTUAL-cost-protocol.json --commit ACTUAL_40_CHARACTER_COMMIT

   A separate invocation of the existing bounded evaluation controller performs
   actual scientific inference. This builder never starts it. Schema74 admission
   rejects late starts, alternate budget journals, changed histories, and
   duplicate/partial output paths. Fixed execution caps and absolute deadlines
   remain binding. No failed or duplicate reservation is refunded.

4. After that actual controller completes, run the existing independent metric
   replay. Point cost_pilot_evidence at the real protocol, summary and replay.
   Prepare the full protocol with --phase full and a NEW protocol filename.
   Its cap is the existing conservative measured-cost projection; it must fit
   both the predeclared maximum and current canonical remaining allowance.
   Every pilot profile/case/track/source must match. Incomplete, failed, cached,
   unreplayed or uncharged pilots cannot supply a full-study admission.

Supported stage modes:
- checkpoint_selection_sh / checkpoint_selection_pooled: all five checkpoints
  0/500/1000/1500/2000 of an actually completed primary2000-update study; primary
  estimated-Sun track only.
- principal_ablations: all eight matched A/B/C settings use the same pose-selected
  SH checkpoint/configuration; pooled control has its own completed/selected
  study and evidence. Both selection tables must cover the same complete cases.
- additional_controls_sh / additional_controls_pooled: independent-Sun control
  plus active/uniform/appearance/geometry policies at32/64/128 searches, using
  the corresponding completed, pose-selected checkpoint.

The builder uses existing ranking/profile/cost helpers. Initialization may win
an actual completed study as a negative finding; no checkpoint is automatically
promoted. Proposal angles/pose policy cannot change after selection. Profile
and case lists are not narrowed when the measured study exceeds the budget.

Physical setup costs bind successful external watchdog receipts, startup-inclusive
per-archive costs, internal sidecars, phase calendar costs and full fixed-cap
reservations. The future evaluator summary reports outer setup costs separately
from internal timing and online inference. Study amortization is not a claim
about deployment query frequency.

Historical completed evidence is checked against its frozen actual start,
execution deadline, source identity and canonical charge. It stays replayable
after old launch leases expire; replay grants no authority to restart old jobs
or reapply their old cap against today's smaller remaining allowance.

Scope limits: metadata/software preparation only. No real protocol/job, model
forward, image decode, ray or render was executed here. No final/calibration
access, acceptance fitting, paper submission or email is included. Modern
baseline and repeat-seed evaluation builders remain separate; future native
baselines must also bind their successful OUTER control/watchdog evidence,
not only an inner pretrained summary.
