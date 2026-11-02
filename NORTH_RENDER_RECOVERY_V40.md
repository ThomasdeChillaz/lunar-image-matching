# Separate recovery after the original render deadline failed

The v37 corrected render **did not meet its original calendar deadline**.
Its controller recorded a TimeoutError at 07:46 UTC on 2 November, after
16 of 24 component summaries completed. Earlier observations showed the actual
Blender process progressing normally. An overnight wall-clock gap occurred;
the available evidence does not distinguish suspension from a clock change.
Both original processes are terminal. No matcher or optimizer ran on these
new images. [Preserved failure](reports/north_render_timeout_v40.json).

This is an **explicit supplemental amendment**, not a reset of the failed
six-hour protocol. Permit at most **1,800 additional calendar seconds** for
render generation, using two CPU threads and existing free local resources.
The new process also checks elapsed monotonic time and fixes an absolute
deadline at its start. A timeout remains a failure; do not extend this recovery
silently or relabel the original render complete.

## Preserve work without changing the experiment

- Byte-verify every file in the 16 completed component folders, together with
  their native images, four geometry records, 15 exact-supervision banks and
  three dense label banks. Copy them to a new run; do not link or mutate originals.
- Regenerate only the eight components lacking complete summaries. The partial
  seventeenth component remains untouched in the original failed folder.
- The actual rendering and supervision kernel is byte-identical to v37.
  Keep terrain, camera templates, solar family, shader, random seeds, native
  pixels, geometry labels and oracle tolerances unchanged. No model outcomes
  determine reuse or case selection.
- Keep four calibration and twelve candidate-final supports unread.

The [pre-execution contract](reports/north_render_recovery_contract_v40.json)
binds prior file hashes, the failed state, prior protocol and new source hashes.
Four new regressions check corruption, non-overwrite, independent copies and
the identical renderer kernel. All **307 repository tests pass** (four optional
skips); a fresh 100-row procedural software smoke also passes. These are
software checks, not lunar-model results.

## Serialized execution

Publish and byte-verify the source first. Only then run:

```sh
/path/to/pinned/Blender --background --factory-startup --threads 2 \
  --python-exit-code 1 --python render_north_development_v40.py --
python verify_north_development_v37.py \
  --run runs/north-v40-development-recovery
python north_training_bundle_v38.py \
  --render-root runs/north-v40-development-recovery
python train_north_pilot_v38.py --publication-commit PUBLIC_COMMIT
python verify_north_pilot_v38.py
python benchmark_north_control_v38.py --publication-commit PUBLIC_COMMIT
```

The unchanged v37 auditor evaluates the completed combined data, including all
72 oracle poses and 243,360 exact label points. Its versioned audit output did
not exist before recovery; the original renderer state stays failed. The v38
bundle binds the recovery summary and this full audit. It does not pretend
the original generation met its deadline.

After the actual 100-update optimizer/model/loss/gradient/RNG replay and complete
cost bank, proceed with the full baselines, independent saved-metric replay,
measured cost admission and three-seed fitting in
[the v39 control](NORTH_EXPANDED_CONTROL_V39.md). A failed or missing gate stops
the queue. Saved artifacts resume explicitly within their original budget;
there is no automatic blind restart. All new model results remain unmeasured
at this amendment's publication.

The original fresh released-LunarGeo gate remains failed. Self-rendered northern
data, fixed relative camera templates, idealized solar geometry and unknown
external pretraining overlap retain their earlier limitations. Training loss
or geometry controls do not establish a model gain, calibrated acceptance,
state of the art or real orbital navigation accuracy. No email is sent.
