# Coupled-assignment training: complete three-seed negative result

All **6,000updates** finished. Every predefined seed selects **initialization54/72**, with16strict poses and3wrong raw proposals. The new loss is not adopted as a model improvement. The stronger unchanged SIFT-first+surface baseline remains57/72. All counts describe72dependent exposed development pairs from eight geometries in four terrain components.

| Seed | Selected update | Selected correct /72 | Selected strict | Selected wrong raw | Trained endpoint correct /72 | Endpoint strict | Endpoint wrong raw |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20261101 | 0 | 54 | 16 | 3 | 47 | 17 | 2 |
| 20261102 | 0 | 54 | 16 | 3 | 47 | 17 | 1 |
| 20261103 | 0 | 54 | 16 | 3 | 49 | 18 | 1 |

Correct raw poses at updates0/500/1000/1500/2000:

- 20261101: 54 / 45 / 45 / 49 / 47
- 20261102: 54 / 44 / 42 / 46 / 47
- 20261103: 54 / 45 / 41 / 45 / 49

## What the experiment shows

The added bidirectional reachable-window dual-softmax correction is standard assignment machinery, tested here with the same81,696parameter model, same geometry labels, training draws, optimizer and inference pipeline. The separate toy fixture showed that it penalizes one proposal-collapse counterexample. It **did not deliver a selected-model pose gain** in this study. This does not prove a universal failure of the method or identify the unique cause of the lunar regressions.

For the predefined primary seed, the matched historical old-loss2000-update endpoint recovers53/72, versus47/72 with the added correction. Both studies still select initialization. Other old repeat seeds used a different variant and are not substituted as matched controls. Modest strict-threshold changes do not replace the predeclared unfiltered-pose selection criterion.

## What was verified

The independent audit replays **1,080saved native checkpoint rows and6,000sampler/update records**, including geometry/window-label checks. Only the separate100-update pilot was independently retrained with exact model/optimizer/RNG/gradient replay; the full6,000-update optimizer history was **not** independently retrained. All three selected models pass actual transform/invalid-input controls. A fresh72-case primary neural forward exactly matches every saved row. Source, protocol, checkpoint and audit hashes are recorded in the JSON.

Initialization remains eligible under the predeclared rule; no best-seed headline or post-test tuning is used. The descriptive paired selected-minus-frozen intervals are zero because selected outputs match the frozen pipeline. They are not evidence of generalization or equivalence on fresh terrain.

Raw pose recovery is not calibrated acceptance. The selected pipeline still proposes three wrong poses; accepted-error/nonoverlap rates and the required10percentage-point correctly accepted gain remain unproven. A75%raw pose rate does not complete the70%correct-accepted target. Training time is not an inference-efficiency comparison.

## Next research decision

Preserve the complete negative result, retain the original matcher and prioritize verified additional3D terrain/lighting data and diagnostic evidence before another training change. The prospective northern scene study has no certified fresh benchmark or model result yet. The existing CRESENT terrain-registration discrepancy and original fresh-LunarGeo shortfall remain recorded. Real orbital accuracy, remaining modern author profiles, transfer, confidence calibration and the full research/paper targets remain unresolved. No email is sent.
