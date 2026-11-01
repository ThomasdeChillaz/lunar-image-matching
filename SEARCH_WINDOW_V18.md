# Align coarse learning with native refinement

This is the next controlled hypothesis, **not a trained model or demonstrated
gain**. The v16 cohort remains immutable. Its primary pointwise/self/cross
variants all retain initialization at54/72; every trained checkpoint regresses.
The repeat seeds must finish and be published regardless of that outcome.

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
counterexample. **The full cohort runner is not implemented yet; no v18
scientific fitting has started.** Before full fitting:

1. Implement the complete runner and publish all fitting source/protocol bytes.
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
