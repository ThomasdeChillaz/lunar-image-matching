# Replay the joint-correlation pilot

Use the tagged source, `solar-joint-v135-development`, and two release assets:

1. The unchanged [v134 development bundle](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-native-v134-development),
   `lunar-native-v134-development.zip`, SHA256
   `ac12965b4faf714ad06aba64a10259a2f9cbe4e4c6f6392539313868c465d182`.
2. The v135 addendum, `lunar-joint-v135-development.zip`. Its exact digest,
   size and payload count are recorded in [artifact.json](artifact.json).

Extract both into the tagged repository root. Never overwrite an existing file
with different bytes. Verify both inventories and use `requirements-lock.txt`.
The addendum's `JOINT_REPLAY_INVENTORY_V135.json` inventories every payload file
and explicitly records its dependency on the original v134 bundle. This is an
addendum, not a standalone data package.

Replay without matcher inference or camera fitting:

```sh
python reports/solar-joint-v135/replay_saved.py --output replay-joint-v135.json
python reports/solar-joint-v135/diagnose_saved.py --output diagnostic-joint-v135.json
python verify_procedural_smoke.py runs/ci-smoke-v135 --output smoke-joint-v135.json
```

The first two outputs must equal `results.json` and `saved-diagnostic.json`
byte-for-byte. Replay checks all 192 scores against independent simulation
geometry, the twenty exact historical controls, fresh hard-peak endpoints and
camera, the eight saved support/field archives and dependency/source hashes.
The last command checks the separate procedural benchmark's CSV schema, known
transforms and grouped arithmetic; it is not lunar evidence. `verify_artifacts.py`
is for a different historical JSON schema and is not its verifier.

Stored coordinates are camera-conditioned image-grid peaks. They are not
independent landmarks. The primary is `joint`, using unchanged AP3P; do not
replace it with `joint_camera_proposal` when reporting recovery. Cached timings
are historical, and new timings exclude initializer computation.

The bundle contains no canonical reservation journal, launch authority, weights,
private correspondence, credentials or calibration/final imagery. Completion
receipts preserve historical charges but grant no new execution authority.
Do not restart the controller from replay files. New scientific inference needs
an admitted prospective protocol and the resource decision described in
[RESULTS.md](RESULTS.md). All original failures and benchmarks remain unchanged.
