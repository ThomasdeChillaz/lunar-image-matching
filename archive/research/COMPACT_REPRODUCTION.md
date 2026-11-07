# Reproduce the compact south-pole study

Use the exact tagged source and published assets. The model remains an independent
research artifact; selected spatial footprints, shared rendering dependence and
unknown trajectory membership limit the claim. Render inference uses 128px grids;
the 3px input criterion equals 12 original 512px source pixels.

## Install and obtain the published study

Python 3.12 is the validated runtime. Install the pinned requirements in a fresh
virtual environment. The compact training/reference runtime uses PyTorch 2.2.2;
modern baseline jobs use their separately pinned CPU environments. Do not merge
those environments or silently replace an unavailable model.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements-lock.txt
```

Download `compact-v2-models.zip`, `compact-v2-data.zip` and
`compact-v2-experiments.zip` from the compact-v2 GitHub release. Keep their
published checksum sidecars. These provide seven own checkpoints, selected
LunarGeo caches, original POLAR photos and case-level experiment records.
Third-party pretrained weights are not repackaged.

From the repository root, install only hash-verified assets and replay saved
metrics into a fresh output:

```sh
python replay_compact_v2.py --models /path/to/compact-v2-models.zip --data /path/to/compact-v2-data.zip --capsule /path/to/compact-v2-experiments.zip --expected-sha256 EXPERIMENT_CAPSULE_SHA256 --output runs/compact-replay-new.json
python verify_compact_tables_v2.py --output runs/compact-tables-new.json
python report_compact_v2.py --output runs/compact-report-new
```

For the publication layout, run:

```sh
python render_compact_publication_v2.py --output runs/compact-publication-new
```

It verifies the frozen analysis hash and changes only figure labels, PDF
metadata and footer links. Measured outcomes, thresholds and uncertainty
calculations remain in the frozen source. The release's small report ZIP can
also be extracted and opened at `compact_research_v2.html` without model weights.

Read the actual experiment capsule digest from
`reports/compact_experiment_assets_v2.sha256`. The installer refuses to replace
an existing file with different bytes. The replay checks frozen sources,
checkpoint/data hashes, native pixels, analytic errors, confidence fitting,
threshold calibration, rejected public outputs, depth visibility and nominal
sensor scores. No new model inference or threshold fitting on test labels is
required. PDF timestamps can differ; measured outcomes remain fixed.

## Rebuild data from original public archives

The smaller published data asset is sufficient for replay. To independently
rebuild from the public StereoGeo sample, use a fresh directory for derived
files and preserve the original inventory/manifest for comparison:

```sh
python lunargeo_dataset.py fetch
python lunargeo_dataset.py inspect --output runs/lunargeo-inventory-new.json
python lunargeo_dataset.py prepare --inventory runs/lunargeo-inventory-new.json --output runs/lunargeo-manifest-new.json
```

Original source acquisition is about 11.4 GB. The loader regroups actual camera
geometry instead of assuming six consecutive filenames. It samples metadata
before image outcomes, groups selected full-depth footprints/shared poses and
keeps all lighting variants together. Camera files have no trajectory IDs or
Sun angles. Do not describe the resulting spatial groups as verified independent
trajectories. Rebuilding requires a fresh `data/compact-v2` cache; do not delete
or replace installed study files merely to reuse those paths.

## Training and development experiments

Train into new directories; do not overwrite released checkpoints. Each of the
four cumulative configurations uses 2,000 steps and seed 20261030:

```sh
python train_compact_v2.py --manifest reports/compact_dataset_v2.json --variant expanded --seed 20261030 --steps 2000 --output runs/expanded-retrain-new
python train_compact_v2.py --manifest reports/compact_dataset_v2.json --variant reliability --seed 20261030 --steps 2000 --output runs/reliability-retrain-new
python train_compact_v2.py --manifest reports/compact_dataset_v2.json --variant hard_negatives --seed 20261030 --steps 2000 --output runs/negative-retrain-new
python train_compact_v2.py --manifest reports/compact_dataset_v2.json --variant multiscale --seed 20261030 --steps 2000 --output runs/multiscale-retrain-new
```

Run development inference with `compact_research.py evaluate`; select the maximum
raw-correct configuration with the declared tie-break, then repeat that exact
configuration with seeds 20261031/20261032. The published selected configuration
is `expanded`; publish every seed, never a best-of-test seed. Train/source and
selection records bind checkpoint hashes. Different CPU/runtime kernels can
change retraining bytes; compare those separately from replaying frozen weights.

For a new candidate, fit confidence only on model-selection cases and choose its
threshold only on separate calibration cases:

```sh
python compact_research.py evaluate --manifest reports/compact_dataset_v2.json --method compact --model runs/expanded-retrain-new --output runs/new-selection
python compact_research.py evaluate --manifest reports/compact_dataset_v2.json --method compact --model runs/expanded-retrain-new --split calibration --output runs/new-calibration
python compact_research.py confidence --selection-run runs/new-selection --calibration-run runs/new-calibration --output runs/new-policy.json
```

This is additional development on already inspected groups, not a fresh final
study. A new model cannot reuse the consumed final test as an untouched reserve.
The published old Tycho reserves remain historical and are never reused for
selection in this iteration.

## Frozen evaluation and modern baselines

The public commit is recorded in `reports/compact_publication_v2.json`; the
protocol binds source/checkpoint/policy hashes and all case lists. Final
controlled and 3D commands require that exact published protocol. An intentional
replay of frozen inference must use a new output directory and must not be
presented as a new untouched test. Saved-metric replay is usually sufficient.

The manual `compact-baselines-v2.yml` workflow uses blind, checksummed common
inputs and two CPU threads. Select standard `ubuntu-24.04-arm` for the three
newer BF16 methods and standard `ubuntu-22.04` for legacy integrations, as
recorded in the released execution provenance. No paid runner is needed. Its default matrix covers all 40 method/shard jobs;
its optional JSON matrix can recover only missing workers. Preserve every failed
attempt and acquisition log. Each completed shard must pass the actual
pretrained known-transform control. Author and common settings/weight hashes are
in `reports/compact_baseline_lock_v2.json`; corrected-coordinate profile controls and resource limitations are in
`reports/compact_pretrained_controls_centers_v2.json`. Earlier coordinate-v1
controls remain in `reports/compact_pretrained_controls_v2.json` as historical
software diagnostics.

Common baseline inference uses exactly the same 128px source information and
validity masks. LoMa-B/R internally use 504px tensors; RoMa v2 uses fast 512.
Author LoMa path controls retain 1024px detection and 784px description; author
RoMa-v2 control settings are recorded from the actual run. Runtime includes
independent forward and reverse inference. Final jobs run the compact checkpoint
on the same inputs/CPU/runtime, preserving paired latency and accuracy. Memory
is process high-water RSS, not isolated weight memory or spacecraft memory use.

Only complete declared common comparisons enter the primary table. The scorer
can preserve compute-limited subsets with `--allow-partial`, explicitly marked
`PARTIAL.json`; they do not become complete comparisons or confidence-calibration
budgets. Error/coverage curves describe the consumed test and do not select new
thresholds. Point targets and exploratory grouped uncertainty remain separate.

## Use the matching command

After installing the released model asset, select compact-v2 explicitly:

```sh
python match_terrain.py --model-version compact-v2 --model-dir runs/compact-v2-expanded-20261030 --protocol reports/compact_freeze_v2.json --reference reference128.png --query query128.png --output runs/my-compact-alignment.json --correspondence-output runs/my-compact-correspondences.json
```

The default command still uses v1. Coordinates map reference x,y pixel centers
to query x,y centers. Rejected outputs return `transform: null` and no public
correspondences. Add `--include-proposal` only for an explicitly ungated research
proposal. For nonplanar study, `compact_research.py stereo` extracts raw matches
before planar fitting and uses reference-map depth for pose; query depth is only
an external scoring reference. The 3D diagnostic does not certify navigation.

A post-study geometry software control uses exact projected correspondences
on the first development geometry in each acquisition mode. This is a diagnostic
after the observed pose failures; it changes no matching result or test metric:

```sh
python verify_compact_oracle_v2.py --output runs/oracle-control-new.json
python diagnose_compact_gate_v2.py --output runs/old-gate-diagnosis-new.json
```

Run repository tests and a fresh procedural smoke after changing matching or
metrics. No human landmark review, paid compute or automatic outreach is part
of this workflow.
