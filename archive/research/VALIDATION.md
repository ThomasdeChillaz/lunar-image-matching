# Executed verification

## Installation
A fresh Python 3.12 virtual environment was installed from requirements-lock.txt, including the pinned LightGlue source revision. `python -m pip check` reported no broken requirements.

## Tests
`RUN_LEARNED=1 python -m pytest -q`: **33 passed** in both the development and fresh environments. This includes an actual pretrained DISK/LightGlue forward pass; missing/corrupt model files; known-transform recovery; blank/invalid masks; source and region split leakage; annotation import; metric denominators; independent target points; and the end-to-end procedural CLI/report path.

## Data and model acquisition
A second download into an empty data directory retrieved and verified all 15 raster checksums. A separate empty model cache retrieved both pinned upstream weight files and verified their SHA256 before model loading.

## Full experiment
`python evaluate.py --output runs/final-v2` executed **126 evaluations**: 12 repeat pairs x 3 methods x 2 preprocessing settings, plus 3 regions x 3 exact-transform controls x 3 methods x 2 settings. The frozen configuration used no parameter search. `verify_artifacts.py` verified the measurement count, unique experiment keys, input/source hashes and recomputed split/reference-specific summaries.

Published `lunar_` artifacts come from that run. Its code hashes identify the executed sources even though browser-created GitHub commits have different IDs from local commits. Use `python verify_artifacts.py runs/reproduction` on a fresh run; for the published flat files, compare their frozen inputs and source hashes in lunar_environment.json.

## Visual/UI verification
The two-page A4 PDF was rendered and inspected for clipping, alignment, chart labels and table readability. The offline viewer's pair selector, method selector and overlay slider were exercised in the browser. The annotation page's paired clicks and undo were exercised; test annotations were cleared and never imported as scientific references.

## Scientific boundary
All twelve real pairs were visually inspected for overlapping terrain. This is not independent subpixel annotation. The upstream dataset supplies no map-error covariance or independent check points. Those scientific acceptance checks remain open in issue #1. No navigation-accuracy, physical shadow-model, additional-DEM or LIMA-reproduction claim is made.
