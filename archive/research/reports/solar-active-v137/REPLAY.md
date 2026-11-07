# Replay the complete active-acquisition study

Download `lunar-active-v137-development.zip` and its SHA256 sidecar from the
`solar-active-v137-development` prerelease. Extract into a **new, empty directory**.
The capsule includes the exact reference/scoring arrays, source closure, original
predictions and all 832 saved scores; older release downloads are unnecessary.
No model weights are required for saved-output arithmetic.

Use Python 3.12 and the versions in `requirements-lock.txt`. The independent replay
itself needs NumPy 1.26.4 and opencv-python 4.10.0.84. From the extracted root:

```sh
python reports/solar-active-v137/verify_package.py
python reports/solar-active-v137/replay_saved.py --output independent-replay.json
python -c 'import json; from pathlib import Path; assert json.loads(Path("independent-replay.json").read_text()) == json.loads(Path("reports/solar-active-v137/results.json").read_text())'
```

Use an output path that does not exist; the replay refuses overwrite. The verifier
checks all inventoried bytes and sizes, frozen source/input pins and the sealed
prediction digest. The replay independently unprojects axial depth, applies camera
frames and visibility, computes native errors and camera errors, retains failures,
compares all 20 historical controls, and verifies fresh initialization/Direct and
eight no-extra pilot rows. It then recomputes terrain-component descriptive
intervals, including both endpoints of negative pairs. It imports no matcher,
pose estimator, controller, private journal or compute admission authority.

The scientific inference source was public before inference at
`48c6e9ac4f15a1787a3bfdce6a5b84de54090763`. Protocol SHA256:
`c39533b10b16e9ba0cc48c5c2aa8d671f1c037bbcc8dc89670a5d3c5cfce49b7`.
The tagged analysis/presentation source was added after inference and cannot
change the sealed predictions. The main prediction seal is
`4d2cbcde916293efd1ec89f12354ecb90c345cc63329fbaf7f78a27a9bf46719`.

## Software and presentation checks

In a full source checkout with the validated environment:

```sh
python -m pytest -q
python reports/solar-active-v136/smoke.py --output fresh-generated-smoke.json
python reports/solar-active-v136/replay_smoke.py fresh-generated-smoke.json --output fresh-generated-replay.json
```

The full local suite passed 1,490 tests with four explicit skips. The fresh
generated control is a small wiring check with two extra searches per policy;
it is not a replacement for the separately charged full32-search timing pilot.
The completed full-budget pilot is in `runs/solar-active-v137/timing/`.

Regenerate figures, the saved-output demonstration and PDF drafts with:

```sh
python reports/solar-active-v137/build_research_package.py
```

This last command needs Matplotlib, ReportLab and the EXR decoder dependencies
from the full pinned environment. It only presents sealed outputs and existing
public images; it performs no matching, camera fitting or training. The demo's
success, nonoverlap and failure are chosen after evaluation for illustration;
every case remains in the scientific table. Contrast is display-only.

## Resource and claim boundaries

The capsule contains a non-sensitive allowance summary, not the current private
ledger, reservation journal, invocation identity, execution ticket or serial lease.
Three **historical expired execution receipts** remain because the frozen source
protocol already hashes them and preserves earlier failures. They are public
audit evidence, not new execution authority. No calibration/final imagery, private
correspondence or credentials is distributed.

Saved replay does not reserve compute or authorize a new inference experiment.
Successful fixed charges remain 120+1,800 seconds from the approved supplement;
its 480-second recovery envelope is unspent. Original accounting is unchanged.
New scientific inference needs a prospective, bounded resource protocol.

The result is 8/16 raw poses on four exposed conditions/three terrains using
matching reference illumination. It is not calibrated acceptance, unknown-Sun,
fresh-terrain, modern same-case, sensor-transfer or verified orbital accuracy.
Read [the complete result](RESULTS.md) and
[the fourteen unchanged requirements](../research_requirements_status_v137.json).
