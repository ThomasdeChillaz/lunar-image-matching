# Small research metadata fixtures

`metadata.zip` contains 21 exact, historical JSON records used by metadata-only
unit tests: the model-selection case manifest, eight completed camera-unit
summaries and seals, failed timing evidence, a prediction seal and the original
MOONSt3R software-control protocol. Its inventory preserves each file hash.

The pytest session restores only absent allowlisted JSON paths. It refuses
different existing records. No raster, array, model weight, executable,
reservation journal, worker ticket or new computation is supplied. Tests that
require the original local reservation journal remain explicitly optional.
The synthetic matching and camera controls run in a clean checkout.

The MoonAnything/LunarGeo-derived metadata retains the attribution and CC BY 4.0
notice in [THIRD_PARTY.md](../../THIRD_PARTY.md). These are development records;
no calibration or final imagery is included.
