# Reproduce the localization diagnosis

This milestone contains development diagnosis and a failed fresh-data gate.
There is no newly trained compact-v3 model, final pose ranking or orbital
navigation validation. Start with [the report](reports/LOCALIZATION_DIAGNOSTICS_V3.md).

## Replay saved measurements

1. Install the validated requirements-lock environment and compact-v2 data/model
   assets using [COMPACT_REPRODUCTION.md](COMPACT_REPRODUCTION.md). Keep the
   current repository sources; the v2 matching/scoring sources are unchanged.
2. The new [measurement capsule](reports/localization-diagnostics-v3.zip) and
   [SHA256](reports/localization-diagnostics-v3.sha256) are in this repository.
3. Install and replay from the repository root:

```bash
python package_localization_v3.py --install reports/localization-diagnostics-v3.zip
python verify_localization_v3.py --output reports/localization_replay_new.json
python verify_localization_pose_refit_v3.py --output reports/localization_refit_replay_new.json
```

Outputs must use new paths: measurement/audit files are immutable. Installation
verifies the whole capsule before writing and refuses different existing files.
The capsule supplies development records, full compressed camera metadata,
executed source snapshots, protocols and reports; it excludes weights/source
imagery. Independent replay needs the existing v2 depth caches, not the large
original LunarGeo image archives. The capsule was installed and replayed in a
separate checkout with v2 assets.

The inventory's original source snapshot is under runs/localization-v3/source;
the current inventory module tightens future training prerequisites without
changing its inventory routine. Both the initial and lazy-import-cleanup
diagnostic executions are retained. The validated execution is the reported
one; counts agree. No outcome-based model or threshold selection occurred.

## Run new development measurements

Acquire/verify the source LunarGeo archives through lunargeo_dataset.py and keep
all original v2 assets. Then use fresh output paths:

```bash
python localization_data_v3.py --output reports/data_inventory_new.json --records runs/camera_inventory_new.jsonl.gz
python localization_diagnostics_v3.py --output runs/diagnosis_new
python localization_pose_refit_v3.py --source runs/diagnosis_new --output reports/pose_refit_new.json
python verify_localization_v3.py --run runs/diagnosis_new --gate reports/data_inventory_new.json --output reports/diagnosis_audit_new.json
python verify_localization_pose_refit_v3.py --input reports/pose_refit_new.json --output reports/pose_refit_audit_new.json
```

The first experiment uses fixed EPNP/RANSAC. The later LM experiment explicitly
tests a second estimator using only the original RANSAC inliers. No query depth
or true query pose is used in matching/refinement/pose inference. Reference
ray-depth is simulated map geometry and is shared across these comparisons.

The twelve-fresh-group condition is unmet. Future training requires verified
fresh footprints, successful split audit and verified batch/trajectory grouping;
folder counts alone cannot authorize it. Do not reuse these diagnostic cases as
a final reserve or silently relax the contract.
