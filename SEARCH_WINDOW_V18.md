# Align coarse learning with native refinement

**Completed negative study, 1 November 2026.** All six runs completed12,000
updates. Every selected checkpoint is initialization54/72; primary trained
endpoints49/50/53/43 and repeats43/45. Same SIFT-first baseline57/72. These
trained variants are not adopted. [Complete outcomes](reports/window_cohort_complete_v18.md)
and [independent audit](reports/native_window_complete_audit_v18.json).

The cohort audit checks2,160 saved checkpoint rows and all12,000 update/sampler
records; no exact full retraining is claimed. The train-only100-update pilot
was independently retrained exactly. Actual selected-model forward controls
and the full development capsule are queued. Do not restart this terminal
cohort or alter its public76ec8bf fitting sources. The v16 study remains
immutable. The method motivation/protocol below is retained as executed
history, not evidence that this hypothesis worked.

## Measured reason for the change

The pointwise diagnosis evaluates all72 exposed selection pairs and108 fixed
training fixtures, across four orientations and checkpoints0/500/2000. Every
diagnostic proposal exactly matches the actual matcher. Independent geometry
replay covers all2,160 records. From initialization to update2,000:

-Selection target NLL improves7.123→5.783, while12px-correct proposals decrease
11,880→9,409 and geometrically reachable33px proposals decrease12,694→10,506.
-Training-fixture NLL improves7.061→5.935, while reachable proposals decrease
16,685→13,717. This is not explained solely by an exposed-selection overfit.
-Truth-dependent filtering of the saved failed native matches recovers only
two additional cases among the tested filters. That is an oracle diagnosis,
not a model result or proof that every possible filter is incapable of more.

The current bilinear likelihood demands probability distributed around the
projected grid cell. Native refinement can instead recover a point from any
coarse candidate whose rounded original-pixel33px search contains that point.
The new tests demonstrate a concrete difference: two such reachable proposals
receive the same set likelihood, while an unreachable distractor is penalized.

## Implemented core and controlled comparisons

`search_window_learning_v18.py` implements:

-An exact original-pixel search-window label, based on audited3D visibility,
the real pixel-center conversion and the same rounding used by native search.
Darkness remains separate from occlusion.
-Positive set likelihood and the existing absolute positive/negative margins.
An optional anchor protects frozen mutual proposals only when independent
training geometry labels them as reachable genuine proposals.
-A separately frozen original native fine CNN. The coarse CNN can either stay
frozen or learn; native fine features and pose estimation remain fixed.

The planned ablations change one main factor at a time:

| Configuration | Coarse loss | Anchor | Coarse CNN | Context |
|---|---|---|---|---|
| Historical v16 | Bilinear likelihood | No | Frozen | Pointwise |
| Window-only | Search-window set likelihood | No | Frozen | Pointwise |
| Anchored window | Search-window set likelihood |0.5 | Frozen | Pointwise |
| Coarse adaptation | Search-window set likelihood |0.5 | Train | Pointwise |
| Pair context with adaptation | Search-window set likelihood |0.5 | Train | Cross-image |

New model totals include both CNNs:81,696 pointwise or82,208 cross-context
parameters. Trainable counts are25,152 for the frozen-coarse pointwise control,
53,424 when its coarse CNN learns, or53,936 for cross-context adaptation.
All remain below100,000 total parameters. Different numbers of trainable
parameters are disclosed; the inference architecture includes both CNNs even
when their initial weights are identical.

## Gates before fitting

The core numerical/gradient controls pass, including exact initial proposal
scores, protected frozen fine weights and the meaningful reachable-cell
counterexample. **The complete runner is implemented in `train_search_window_v18.py`; all
gates below passed and the full scientific cohort completed from public
freeze76ec8bf, without a learned improvement.** The required gates were:

1. Publish and verify all complete runner/fitting source/protocol bytes.
2. Verify actual native points/poses on all72 exposed selection pairs reproduce
the frozen baseline at initialization. Run actual known-transform/blank and
invalid-input controls. Do not infer that gate from unit tests alone.
3. Measure a100-update train-only cost/gradient pilot; verify every fine-encoder
weight is unchanged and independently replay labels/sampler/optimizer state.
4. Execute all four new2,000-update ablations at20261101. Select on unfiltered
pose count over all72 pairs, then2px-correct point count, then earlier update,
including initialization. Repeat the selected configuration at20261102/03.
No warm start from the measured pilot or failed v16 checkpoints.

Use AdamW, context lr0.0001 and coarse-CNN lr0.00001, decay0.0001 and gradient
clip5. Keep the same18 training geometries/12 components and train-only
sampling/quarter-turn/sensor augmentations. Physical light changes come from
released rendered views. Evaluate at0/500/1000/1500/2000, preserving all failures.
One two-thread scientific fitting job at a time; cumulative local training
remains subject to the original48-hour cap. Compare against both54/72 frozen
surface and57/72 SIFT-first surface, plus the required modern native methods.

Query depth/true pose enter training labels and scoring only. Preserve the
reference-to-query coordinate convention and unchanged3-native-pixel final
AP3P. Confidence calibration, nonoverlap risk, fresh final data, transfer,
remaining pretrained comparisons and paper remain required. This development
study cannot repair the failed fresh-data gate by relabeling consumed views.
No paid compute, human review or email. No SOTA or orbital accuracy claim.

## Commands and integrity

Use the validated environment and two CPU threads. Every command requires the
public full40-character source commit, and refuses different fitting bytes.
Do not overlap with the native baseline timing job.

```
python train_search_window_v18.py --mode initial --source-commit COMMIT --output runs/localization-v18-window-initial-control
python train_search_window_v18.py --mode pilot --source-commit COMMIT --output runs/localization-v18-window-cost-pilot
python verify_window_pilot_v18.py --run runs/localization-v18-window-cost-pilot --output reports/native_window_pilot_audit_v18.json
python train_search_window_v18.py --mode cohort --source-commit COMMIT --output runs/localization-v18-window-development
```

The pilot is train-only and measures100 updates. Its audit independently checks
3D projection/visibility and every candidate search-window membership, then
exactly retrains the same implemented loss to compare gradient history,
weights, optimizer and RNG. This is not an independent loss implementation.
The full cohort must not warm-start from that pilot. Initialization checks both
distinct inference graphs on all72 native selection pairs, plus actual transform,
blank and invalid-mask controls. Same frozen native feature maps may be cached;
coarse descriptors, context, points, warped queries and poses are recomputed.
No isolated inference timing claim applies to the training runner.

The new full cohort has a six-hour wall deadline shared across every ablation,
repeat and resume. This bounds the new allocation within the original local
compute budget; a timeout preserves partial evidence and is not a completed
comparison. Earlier major cohorts record approximately5.24 hours including
evaluation where fitting-only timing was unavailable; that is not a complete
pilot/audit resource ledger.

## Executed gates and live cohort

Both initialized graphs exactly reproduce all72 frozen native/pose outputs.
Actual transform/blank/invalid controls pass. The100-update train-only pilot
completed in23.05s and independently exactly retrained model/gradients/optimizer/
RNG; geometry/window audit covers97,344 points and65,804,544 memberships. This
is a reproducibility gate, not a pose result. The full12,000-update cohort has
started from fresh initialization. Its live stage/log and the original source
freeze are recorded in STATUS.md. Do not launch a second fitting/research job
or mutate its pinned sources. No completed learned gain is claimed yet.
