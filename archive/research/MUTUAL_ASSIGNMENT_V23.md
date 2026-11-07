# Couple window supervision to bidirectional competition

Prepared 1 November 2026. **No new training or localization result yet.**
The current MOONSt3R comparison and queued clean replay take priority. Do not
run another fitting/model-comparison job while they are active. Historical
fitting sources, checkpoints and all failed outcomes remain unchanged.

## Evidence and hypothesis

The completed v18 study changed the positive label to every coarse candidate
whose native search can reach the correct point. All six runs still selected
initialization at54/72; none improved localization. Larger descriptor context
also failed to improve pose despite better pooled point precision. These
findings do not identify one unique cause.

There is a specific remaining mismatch worth testing. Training independently
rewards reachable probability mass in each direction. Inference keeps only
mutual row/column maxima. Several nearby references may compete for one
reachable candidate, so a better training objective value need not yield more
mutual proposals. The existing genuine-match anchor compares an anchor with
geometrically wrong candidates, excluding competing reachable candidates.

The small algebra control in `diagnose_mutual_assignment_v23.py` uses the actual
v18 directional loss, margins and anchor. A reachable score increase lowers
that objective while reducing two mutual proposals to one. Adding coupled
normalization penalizes the fixture. This is a counterexample to a guarantee,
**not proof that this mechanism caused the actual lunar regressions**.

## One controlled change

For score matrix S and the unchanged temperature0.1, form

`log P = log_softmax(S /0.1, rows) + log_softmax(S /0.1, columns)`.

Replace the positive row-set probability with the summed dual-softmax mass
inside the same exact3D-labelled recoverable windows; average both directions.
Implementation adds `log(row allowed mass /dual allowed mass)` to the existing
v18 loss. Existing visibility, native rounding, positive/negative margins,
anchor0.5, sampler, image augmentations, model and optimizer remain unchanged.
No new parameters, match threshold or inference decision is introduced.

Dual-softmax coupling is established matching machinery, including
[EfficientLoFTR](https://zju3dv.github.io/efficientloftr/files/EfficientLoFTR.pdf).
This is a lunar training hypothesis using that machinery, not a new operator
or a reproduction of the full LoFTR/EfficientLoFTR method. Neither operator
mass nor a lower loss is a calibrated acceptance probability or pose gain.

## Bounded train-only pilot

`train_mutual_pilot_v23.py` is prepared for100 updates, seed20261101, with
the same coarse-adaptation architecture:81,696 total parameters /53,424
trainable. Context AdamW1e-4, coarse1e-5, decay1e-4, clip5; one pair/update,
verified nonoverlap every fourth update. Native fine encoding stays frozen.
Use the existing18 training geometries /12 components only. No selection,
calibration or test image is loaded and no checkpoint is selected here.

Before fitting, require all software tests and a fresh smoke/replay, then a
byte-verified public source freeze. The existing two-graph actual72-case gate
can be reused only after checking every bound source and exact fresh seeded
weights against the historical controlled initialization. This is valid
because the model/inference graph does not change. Never warm-start from any
fitted pilot/cohort weight. The prior independently audited geometry/window
labels can be reused only with identical manifest, input and source hashes.

```sh
python train_mutual_pilot_v23.py --source-commit PUBLIC_COMMIT \
  --output runs/localization-v23-mutual-cost-pilot
python verify_mutual_pilot_v23.py \
  --run runs/localization-v23-mutual-cost-pilot \
  --output reports/native_mutual_pilot_audit_v23.json
```

The pilot preserves every attempt, checkpoints every10updates and stops after
600 fitting seconds. Its replayer independently recomputes the new scalar in
float64 NumPy at updates1/4/50/100 (both directions), exactly retrains all100
updates, and checks gradient/loss history, optimizer, RNG, original-augmentation
draws, fine-weight protection and native input/source fingerprints. Same-code
neural retraining is not an independent neural implementation.

## Decision before full fitting

Passing the pilot establishes correctness/cost only. It cannot select a model
or demonstrate pose improvement. `train_mutual_cohort_v24.py` is prepared for
three fresh predefined seeds20261101/02/03,2,000 updates each, with all72 exposed
pairs evaluated at0/500/1000/1500/2000. All three seeds are retained even if
the hypothesis fails. It reuses the frozen v18 checkpoint/resume/evaluation
loop under an explicit temporary objective binding in one dedicated worker;
the binding is restored on failure and changes no historical source file.
The shared six-hour cohort deadline cannot be reset on resume, and a sealed
protocol prevents changing its settings/time origin. A measured100-update
pilot must project at most three fitting hours for6,000updates before launch.
No full fitting has been executed yet. Freeze the complete runner/protocol
before any new model-selection inference. Use
unfiltered common-pose success, including failed estimates, rather than loss
or descriptor-point precision. Compare with the matched historical v18
coarse-adaptation control, frozen54/72 surface and stronger57/72 SIFT-first
baseline, all on the same exposed cases. Initialization remains eligible.
No threshold or individual case may be chosen using query truth in inference.

Fresh final data, reliable acceptance/nonoverlap, remaining comparisons,
transfer, uncertainty, paper and the compact-v3 interface remain required.
This hypothesis does not repair those gaps or imply novelty, SOTA, verified
orbital navigation accuracy or a promised breakthrough.

```sh
python train_mutual_cohort_v24.py --source-commit PUBLIC_COMMIT \
  --output runs/localization-v24-mutual-development
python verify_mutual_cohort_v24.py \
  --run runs/localization-v24-mutual-development \
  --output reports/native_mutual_complete_audit_v24.json
python control_mutual_cohort_v24.py \
  --cohort runs/localization-v24-mutual-development \
  --audit reports/native_mutual_complete_audit_v24.json \
  --output runs/localization-v24-mutual-selected-controls
```

The complete-cohort audit replays all1,080 saved native checkpoint rows and
6,000 update/sampler records through the existing independent geometry/pose
auditor. It deliberately does not call the old training loss or claim a full
optimizer retraining; new-loss exact retraining is confined to the100-update
pilot. Selected-model controls cover all three seeds, plus an actual primary
72-case forward replay. None of these software checks alone establishes a
localization improvement or satisfies the remaining fresh-data/acceptance goals.
