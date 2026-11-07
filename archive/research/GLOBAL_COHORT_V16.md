# Capacity-controlled global proposal context

The completed geometry-normalized study selected55/72 correct raw poses in
each of three seeds, versus54/72 for frozen surface refinement. The same SIFT
fallback gives58/72 versus57/72, with one wrong raw estimate. A one-pair gain
on four exposed selection components is inconclusive. Broader initialization
added15 wrong poses and no correct ones; it is not adopted.

The next experiment tests whether global proposal discrimination helps before
the unchanged native refinement and pose estimator. This is generic attention,
not an architectural novelty claim. No context localization result is available
at preregistration. A train-only100-update pilot completed in14.33 fitting
seconds under concurrent load; independent replay reproduced all optimizer
updates, weights and random states. That is a cost/control result only.

## Frozen comparison

-18 existing training geometries/12 components; eight exposed selection
geometries/four components. No calibration/test pixels. Fresh-final gate fails.
-Frozen28,272-parameter descriptor backbone; train only capacity-matched
pointwise25,152, self-context25,408 and cross-context25,664 parameters. Combined
models53,424/53,680/53,936, all below100,000.
-Train each variant from seeded initialization for2,000 AdamW updates at
20261101, lr0.0001, weight decay0.0001, gradient clipping5, two CPU threads.
No warm start from the cost pilot. Use the same train-only component, lighting,
quarter-turn, sensor-augmentation and one-in-four verified nonoverlap draws.
-Bidirectional audited axial-depth3D supervision on676 tokens/image with the
already controlled likelihood and absolute score margins. Darkness is not
geometric occlusion. Illumination variation comes from released rendered views.
-Evaluate at0/500/1000/1500/2000 on all72 exposed selection pairs. Select by
unfiltered correct poses, then2px correct correspondence count, then earlier
checkpoint. Initialization is eligible. Rank primary variants by the same key;
declared pointwise/self/cross order breaks exact ties. Repeat the selected
variant from scratch at20261102 and20261103. Publish all outcomes.
-Matching receives images, reference geometry and query intrinsics only.
Query depth and true query pose enter scoring, never proposals or selection
between pose branches. Keep four orientations, deterministic native refinement,
two reference-surface updates and the final3-native-pixel AP3P estimator fixed.
-Cache only frozen native CNN maps across evaluations. Recompute context,
warped queries, correspondences and poses. Concurrent evaluation costs are not
isolated inference latency measurements. Existing speed claims do not transfer.
-Save every fitting source, manifest/input/checkpoint hash, loss, sampler draw,
optimizer and both random states. Checkpoint every100 updates and before each
readout. Resume with identical arguments/source freeze; never restart completed
training or discard failed configurations. Run one two-thread job at a time.

`train_global_context_v16.py --source-commit PUBLIC_SHA --output
runs/localization-v16-global-context-development` executes the complete cohort.
The source must match its public commit before launch. `--pilot --steps100
--interval100` uses the historical16-pair bank and is a software/cost exercise,
not the primary cohort. The full runner verifies each variant's initial native
points/poses against the historical surface baseline before its first update.

The raw-pose criterion remains rotation≤2° and translation≤1% of median
reference range. A raw estimate is not an accepted pose. The stronger simple
baseline is57/72 SIFT+surface. Reliable acceptance, genuinely fresh test groups,
remaining required pretrained baselines, transfer, uncertainty, paper and the
public compact-v3 pose interface remain required. Do not change the objective
to a small exposed-development gain. No paid compute, human review or email.
