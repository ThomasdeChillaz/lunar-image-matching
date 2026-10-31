# Compact-v3 development training

Both five-run cohorts are complete: 20,000 optimizer steps plus the pilot.
Balanced reliability seeds give 27/72, 31/72 and 36/72 poses versus baseline
26/72; the primary descriptive interval includes zero and the 70% target is unmet.
See [all measured outcomes](reports/LOCALIZATION_BALANCED_V3.md).

## Supplemental balanced-sampling follow-up

The first five runs established no learned pose advantage and exposed a concrete
sampling issue: only 2,079/8,000 primary-seed examples were positive, some batches
had zero spatial loss, and the reliability head rejected every selection-bank
proposal at its fixed cutoff. Those outcomes remain unchanged.

A second explicitly supplemental cohort changes only sample balancing: exactly
two geometrically valid in-window positives and two invalid local matches per
batch4. It retains the same network classes, initialization, physical images,
augmentation,2,000-step budget, selection bank, pose estimator and0.5 point cutoff.
Run all three ablations with seed20261101 and the winner with20261102/20261103.
It remains development tuning after inspecting the first cohort, not a new test.
The supplemental trainer has its own source/protocol hashes and run directory.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python train_balanced_localization_v3.py --development-only --steps 2000 --batch-size 4 --interval 500 --output runs/localization-v3-balanced-development
python verify_balanced_localization_v3.py
python report_balanced_localization_v3.py
```

The user requested training after the iteration7 fresh-data gate failed.
Development-only fitting is authorized on the original train split; the gate
continues to govern fresh final validation. No calibration/test images are
loaded, and no old result or checkpoint is overwritten.

## Model and experiment

The frozen coarse matcher proposes reference-to-query matches on 128px images.
A second structural encoder evaluates original-resolution image detail. A
33x33-native-pixel window is searched around each coarse query proposal.

Three ablations use the same original eighteen train geometries/eight spatial
components and the same 8,000 sampled examples per run:

| Candidate | Trainable parameters | Total including frozen coarse network |
|---|---:|---:|
| True3D-trained descriptor |28,272|56,544|
| Learned pair-conditioned refinement |35,233|63,505|
| Refinement plus local matchability |35,826|64,098|

Each runs 2,000 AdamW optimizer steps, batch 4, learning rate 0.0001, two CPU
threads, seed 20261101. Labels come from ray-depth and camera projection with
query-depth visibility, not from matcher agreement or image darkness. Sampling
includes actual proposals, local jitter, incorrect locations and confirmed
nonoverlapping train terrain. Photometric augmentation is exposure, gamma,
blur, noise and quantization; changing sunlight comes from the rendered data.

The reference training crop is 27px, sufficient for the encoder's 12px support
and fractional pixel sampling; the query crop is 57px for the 33px output window.
A numerical test verifies agreement with whole-image native feature inference.
The initial100-step runtime pilot used57px reference crops/batch8 and is a
separately retained pilot, not an eligible primary candidate.

Checkpoint selection occurs at steps 0/500/1000/1500/2000 using a fixed sixteen-case
metadata-selected bank covering all eight selection geometries/four components.
Rank by correct raw poses, then2px-correct correspondence count, then earliest
step. Initial weights are eligible and must be identified if retained. Choose
the ablation winner using that same key; repeat only that configuration with
seeds 20261102 and 20261103. All selected checkpoints are also evaluated on all72
previously inspected development stereo/light pairs. Neither set is a fresh test.

Pose fitting uses the same reference geometry and EPNP/RANSAC plus original-inlier
LM refit for every candidate. The local reliability ablation has a predeclared 0.5
point-score cutoff. It is not a calibrated pair-level acceptance policy. The
development command returns transform:null/pose:null and no public matches by
default; --include-proposals explicitly requests research correspondences.

## Commands

Use the validated requirements-lock environment. Install earlier compact-v2
assets, acquire the checksummed original LunarGeo archives, then prepare only
train/model-selection native caches:

```bash
python prepare_localization_training_v3.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python train_localization_v3.py --development-only --steps 2000 --batch-size 4 --interval 500 --output runs/localization-v3-development
python verify_localization_training_v3.py
python report_localization_training_v3.py
```

All outputs are immutable; use new output paths for another
cohort. A latest.pt snapshot contains optimizer/model/RNG state every 100 steps;
the current driver does not automatically resume an interrupted directory.
Preserve it and its executed source snapshots for an explicit continuation.

Every run saves initial, selected and final weights, actual weight-change norm,
source snapshots/hashes, all selection outcomes, seed, sampled-label counts and
complete development correspondences/poses. The independent audit reprojects
all 234 supervised geometry/light cases and replays all 360 development outcomes,
checkpoint selection, ablation selection and seed completeness.

To inspect a trained candidate after installing released assets:

```bash
python refine_lunar_development.py --model-dir runs/localization-v3-development/descriptor_3d-20261101 --reference reference.png --query query.png --include-proposals --output research_correspondences.json
```

The images must have equal dimensions, with coordinates expressed in their
original pixels. --checkpoint final requests the last optimizer state instead
of the development-selected checkpoint. No query depth or true query camera
pose is supplied to the refiner. Both are used only for geometric supervision
or scoring; map-aided pose inference uses reference geometry and query intrinsics.

Results and failure analysis are in the
[primary report](reports/LOCALIZATION_TRAINING_V3.md) and
[supplemental report](reports/LOCALIZATION_BALANCED_V3.md). A successful
training run does not establish a new method advantage or independently verified
orbital navigation accuracy. Trajectory/batch identities remain unavailable.

## Install and replay the measured release

Download all three ZIP files and their matching SHA256 sidecars from the
`compact-v3-development` pre-release into `reports/` in a checkout of that tag.
The models archive contains both cohorts' initial, selected and final own weights,
last optimizer/RNG snapshots and the own coarse checkpoint. The data archive contains the 26 native
train/model-selection caches. The experiments archive contains executed source
snapshots, full case outcomes and protocols. No calibration/test pixels are included.

```bash
python package_localization_training_v3.py --install reports/localization-v3-development-models.zip
python package_localization_training_v3.py --install reports/localization-v3-development-data.zip
python package_localization_training_v3.py --install reports/localization-v3-development-experiments.zip
python verify_localization_training_v3.py --output runs/replayed-primary-training-audit.json
python verify_balanced_localization_v3.py --output runs/replayed-balanced-training-audit.json
```

The installer validates every archive/member hash before writing and refuses to
replace a different existing file. Use the tagged source checkout to avoid source
hash mismatches. These replays verify saved geometry, metrics and selection;
rerunning training uses the commands above with new output directories.
