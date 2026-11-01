# Global-context training pilot: feasible, not evaluated for localization

The cross-context proposal module completed **100 optimizer updates in 14.33
seconds**, with 25 verified nonoverlap training pairs. All fitting used the
existing 18 training geometries in 12 components, two CPU threads and a frozen
backbone. No model-selection, calibration or final-test pixels were read by
this pilot. Timing was concurrent with another training job, so it is an
observed fitting cost rather than an isolated inference-speed result.

Independent verification projects 97,344 forward/reverse label checks across
18 geometries and four image orientations. These repeat surface points and
are not independent terrain tests. It then repeats all 100 optimizer updates,
reproducing losses, gradients, final weights, AdamW state and RNG exactly.
The context parameters change; the backbone remains unchanged.

The context module has 25,664 parameters, or 53,936 including the frozen
backbone. The pointwise/self-context controls have 25,152/25,408 trainable
parameters. Combining context with the separate learned fine model later
would total 89,169. Initial context scores, native points and poses exactly
match the frozen surface pipeline on all 72 development cases.

**No localization gain has been measured for the trained context pilot.**
Training loss and reproducibility are not pose accuracy or calibrated risk.
Do not warm-start a headline full cohort from a selectively inspected pilot.
Implement the declared fresh-initialization, capacity-controlled full cohort
and compare against both the stronger simple pipeline and completed learned
surface model. See GLOBAL_CONTEXT_V15.md. The original research targets remain
unmet. No paid compute, human review, automatic email or orbital accuracy claim.
