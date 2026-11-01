# Small global-context proposal experiment

The completed v14 diagnosis found zero correct broad pose proposals among all
60 orientation branches for the 15 previously missing development poses.
Median best genuine correspondence fraction within 12 native pixels was 2.83%.
Seven pairs had fewer than six such correspondences in every branch. Increasing
initial pose tolerance added 15 wrong poses and no correct ones, even after
the unchanged 3-pixel final refinement. That control is rejected.

The next question is whether a small pair-context module can improve global
correspondence discrimination before native refinement and pose estimation.
Generic attention is established machinery; no architectural novelty is
claimed. A useful lunar result requires measured gains and independent data.

## Implemented and executed

`global_context_v15.py` contains pointwise, self-context and cross-context
variants with shared image weights and synchronous cross-image updates. It
receives frozen descriptors and image pixel coordinates only. It does not
receive query depth, query pose, case identity or terrain labels.

The residual score correction starts exactly at the original frozen matching
scores. This preserves the old matching threshold, mutual selection, ranking
and point budget while allowing gradients to change context. No parameters
have been trained yet. The cross-context module has 25,664 parameters:
53,936 including the frozen backbone, or 89,169 if combined later with the
existing separate pair-refinement network. Neither combination exceeds 100,000.

Actual native-image controls pass. The cross-context initial pipeline exactly
reproduces all 72 historical reference points, query points and camera poses,
including all 54 primary and 16 strict successes. Blank images return no pose.
The initial-control artifacts are in `runs/localization-v15-context-initial-control`.
The initial-model stage passed 201 tests with four optional skips and a fresh 100-row
software smoke. Unit tests verify initialization, swapping symmetry within
floating-point error, actual proposal matching, finite gradients and weight
updates. These checks prove software readiness, not a trained model advantage.

## Supervision implemented; measured fitting still pending

The train-only data builder, bidirectional loss, sensor augmentations,
nonoverlap sampler and capacity factory are implemented in
`context_supervision_v15.py`. Geometry/rotation, darkness, candidate-edge and
loss-preference controls pass. Preparation verifies all 18 training geometries
and 12 components. The current repository passes 205 tests with four optional
skips, plus a new 100-row smoke.

`train_context_pilot_v15.py` implements a bounded 10–100-update pilot. It
requires every fitting source byte to match a published Git commit, reads
training pixels only, freezes the backbone and records all optimizer/RNG/source
states. It has not been executed yet. Run its measured pilot after publication,
then implement the full cohort runner before any full fitting.

The protocol requirements are:

- Use only the existing 18 training geometries in 12 spatial components.
  Keep the eight exposed selection geometries in four components separate;
  calibration and test pixels remain excluded. The fresh-final gate still fails.
- Build forward and reverse projected 3D correspondence labels on the original
  128-pixel proposal grid, with the established native pixel-center conversion,
  audited axial depth and geometric visibility. Handle the finite candidate-grid
  boundary explicitly. Never infer occlusion from darkness.
- Use the released lighting pairs and actual image exposure, blur, noise and
  quantization perturbations. Query quarter-turn augmentation must rotate the
  images and their projected labels consistently. Brightness is not sunlight.
- Include verified nonoverlapping training terrain and occluded/out-of-field
  nodes. Test an explicit absolute-score margin as well as correspondence
  likelihood so relative ranking does not push all true scores below the
  unchanged 0.5 proposal threshold. Do not call that threshold calibrated risk.
- Freeze the CNN and train only the context module initially. Compare pointwise,
  self-context and cross-context models. Use approximately matched parameter
  budgets by widening pointwise/self-context feedforward layers, and disclose
  remaining capacity and compute differences. `context_supervision_v15.make_context` now supplies capacity controls with
  25,152 / 25,408 / 25,664 parameters, all within 3% of one another.
- Measure a short training pilot before committing to full fitting. Use two
  CPU threads and the remaining 48-hour local training budget. Preserve source,
  data, RNG, optimizer, checkpoint and case hashes; cache and recover work.
- Use 2,000 updates per primary ablation at seed 20261101. Select checkpoints
  by unfiltered pose success on all 72 exposed selection pairs, then 2-pixel
  correspondence count and earlier checkpoint, including initialization.
  Repeat the selected configuration with 20261102 and 20261103. Publish failures.
- Keep the frozen native refinement and reference-surface pipeline for the
  initial context study. Give every candidate the same 3-native-pixel final
  pose estimator. Compare against the stronger 54/72 surface and 57/72 SIFT
  baselines, plus the completed v13 learned study once its repeats are audited.
  Combining learned fine refinement with context is a separate later ablation.

Pair acceptance, nonoverlap error control, fresh evaluation, remaining modern
native baselines, sensor transfer, timing, uncertainty and the paper remain
required by the full research objective. None is established by initial-model
equivalence. Do not silently replace that objective with software completion or
a one-pair exposed-development gain. No paid compute, human review or email.
