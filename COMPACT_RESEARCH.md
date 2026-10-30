# Compact lunar matching, research iteration 6

## Accepted objective

Investigate whether a model below100,000 parameters can correctly accept more
changing-light lunar alignments and provide a useful CPU accuracy/runtime tradeoff.
No paid compute, manual annotation, automatic outreach or flight-readiness claim.
The earlier Tycho reserves are consumed and all original measurements stay intact.

## Data contract

MoonAnything DOI10.5281/zenodo.18415119 is CC BY4.0 according to the acquired
Zenodo record. The actual StereoGeo release contains SouthPole nadir, oblique
and dynamicgood archives; the TychoCrater folder in that outer archive is empty.
Source acquisition preserves inner ZIP CRC32 and SHA256. Original bytes remain
outside Git. Derived common-input packages retain attribution and transformation
notices, while model weights are distributed as release assets, never Git blobs.

Do not assume six consecutive filenames identify one three-light geometry.
The inspected nadir release contains27,066 camera files,4,518 distinct stereo
geometries and4,498 complete three-light geometries. Twenty incomplete/differently
repeated geometries are excluded with their metadata diagnostics preserved.
The loader regroups by actual intrinsics and camera-to-world matrices.

Select up to64 groups per mode using ascending camera radius, a SHA tie-break and
2,500m minimum camera-center separation within each mode. This metadata-only
resource-limited sample favors lower cameras/smaller footprints; it is not a
claim about the whole release. Join full-depth XYZ footprint bounds with a200m
buffer and shared source poses across every selected mode. Require28 components
before RGB preparation:12 final components across three polar longitude sectors,
four model-selection, four calibration, and remaining components for training.
Fail explicitly if the actual sample cannot support this contract.

EXR values are ray distances, not optical-axis depth. Camera+depth geometry gives
simulation references. Inference and internal scoring use128px derived images;
errors are also converted to the original512px source renders. The fixed3px
benchmark tolerance is12 original source pixels, not3 source pixels. POLAR uses
a1216px crop resized to128px, so its nominal3px tolerance is28.5 sensor pixels. Three-
dimensional matching uses depth visibility and camera projection separately from
partial-affine registration. The inspected camera NPZs contain no Sun angles:
report source lighting-pair IDs rather than inventing angular separations.

## Experiments and selection

Four cumulative ablations: expanded-data architecture control; matchability;
hard nonoverlap negatives; shared two-scale descriptors/local subpixel refinement.
Use2,000 AdamW steps, seed20261030, a fixed model-selection objective bank and
200-step checkpoint intervals. Exposure/gamma/blur/noise/quantization are sensor
appearance augmentation, not physical sunlight. Training and model selection
load neither calibration nor final-test pixels.

Choose the configuration by raw correct registration count on model-selection
groups; break ties by fewer parameters, then declared variant order. Re-train
the selected configuration with seeds20261031 and20261032 and retain every seed.
Seed20261030 is the primary model, not whichever seed looks best on final data.

Confidence uses fixed L2 logistic fitting (1,000 steps, regularization.01) on
model-selection evidence; choose the threshold only on calibration groups.
All methods receive the same metadata-selected six cases per component for
confidence fitting: three lighting pairs and three disjoint negative partners.
Choose maximum correct acceptance subject to <=1% empirical accepted error and
<=1% negative acceptance. Reject-all is an explicit possible result. This small
calibration sample cannot certify a1% deployment risk.

The primary compact/original/SIFT evaluation includes all planned cases. The
resource-bounded modern comparison uses the identical common cases only. Native
source pixels, masks, transforms, errors and case completion are audited.
Compare runtime only within the same CPU/runtime job; never divide a Linux
baseline latency by an unrelated local-machine compact latency.

## Frozen test and sensor transfer

Before final inference, publish case lists, source/model hashes, confidence rules
and the separate40-case NASA POLAR protocol. Then consume the final renders and
fresh laboratory lighting configurations once, with no result-based adjustments.
POLAR terrains01/06/11/12 were previously inspected and reuse rocks and the rig.
New270/350-degree configurations are fresh lighting tests, not new independent
terrain. Nominal fixed-camera identity has unknown calibration/drift uncertainty.
Do not use these sensor labels to fit the matcher or confidence.

Targets: >=10 percentage points more correct acceptance than the original
frozen CNN gate on the identical new test, with both error fractions <=1%.
For a competitive-value finding: better correct acceptance than the strongest
completed common baseline at comparable error, or within3 points while at least
5x faster on the same CPU. Report grouped paired uncertainty, all seeds, failures,
denominators and any targets missed. An inconclusive result remains inconclusive.

## Entry points

```sh
python lunargeo_dataset.py fetch
python lunargeo_dataset.py inspect
python lunargeo_dataset.py prepare
python train_compact_v2.py --manifest reports/compact_dataset_v2.json --variant expanded --output runs/compact-expanded-new
python compact_research.py evaluate --manifest reports/compact_dataset_v2.json --method compact --model runs/compact-expanded-new --output runs/compact-selection-new
python compact_research.py evaluate --manifest reports/compact_dataset_v2.json --method compact --model runs/compact-expanded-new --split calibration --output runs/compact-calibration-new
python compact_research.py confidence --selection-run runs/compact-selection-new --calibration-run runs/compact-calibration-new --output runs/compact-policy-new.json
```

Final-test commands require a public protocol commit and exact matching frozen
sources. The CLI default remains the earlier v1 model; compact-v2 requires its
explicit model directory and frozen protocol. Rejected transforms remain null.

## Delivery

Publish software and measured progress per milestone; release source, checkpoint
assets, audited outcomes, three plots, a two-page brief and a failure-inclusive
demo after evaluation. Prepare outreach text without sending it. Never mark
scientific targets passed merely because all software is implemented.
