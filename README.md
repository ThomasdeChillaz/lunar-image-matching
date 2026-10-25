# Lunar image matching

A reproducible research project comparing classical and pretrained learned image matching on real lunar repeat observations and controlled image warps.

**Question:** How do matching methods behave when illumination changes on the same lunar terrain?

The project includes checksummed data acquisition, masked geospatial loading, NCC and SIFT baselines, pinned DISK + LightGlue inference, frozen region splits, quantitative evaluation, plots, an interactive offline viewer, a two-page PDF report, and an independent-checkpoint review tool.

**Scientific status:** repeat-observation measurements use the publisher's map coordinates. Their accuracy is unknown. These are map-reference diagnostics; independent checkpoint validation remains open. Exact-transform controls provide a separate correctness check. See [STATUS.md](STATUS.md) and [DATASET_CARD.md](DATASET_CARD.md). No NASA/JPL affiliation or endorsement is implied.

## Install and run

Python **3.12** is the verified environment. From this repository:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python reproduce.py --output runs/reproduction --test
```

The command downloads 15 source images (about 26 MB) and two pretrained weight files (about 52 MB), verifies their hashes, runs the test suite including actual pretrained inference, and evaluates all three regions. Original source rasters and weights are excluded from Git. CPU execution may take several minutes. Every run requires a fresh output directory.

For the small procedural software check:

```sh
python benchmark.py --config smoke.json --output runs/smoke-new
```

For individual stages:

```sh
python dataset.py fetch
python fetch_models.py
python evaluate.py --split test --output runs/held-out
```

Use `python reproduce.py --offline --output runs/offline` after assets and weights are cached. Missing or corrupt files produce explicit errors; learned matching never silently falls back to a classical method.

## Inspect results

The published pilot artifacts use the `lunar_` prefix: [report](lunar_REPORT.md), [PDF](lunar_REPORT.pdf), [per-pair measurements](lunar_results.csv), [summary](lunar_summary.json), and [environment/source hashes](lunar_environment.json).

Open `lunar_demo.html` locally for an interactive pair/method selector and overlay slider. GitHub displays HTML source, so download the file or serve this directory locally:

```sh
python -m http.server 8000 --bind 127.0.0.1
```

Then visit `http://127.0.0.1:8000/lunar_demo.html`. For a new run, open `runs/reproduction/demo.html` instead.

![Held-out reference agreement](lunar_agreement.png)

## Independent checkpoint review

```sh
python checkpoint_review.py --output runs/checkpoint_review.html
```

Open that page locally, identify at least ten distributed stable landmarks per pair, enter reviewer and picking uncertainty, and export `checkpoints.json`. Avoid using shadow edges. Import into a new manifest and rerun:

```sh
python apply_checkpoints.py checkpoints.json --output reviewed_manifest.json
python evaluate.py --manifest reviewed_manifest.json --output runs/reviewed
```

Imported annotations are recorded as independent reviewer input; the tool does not claim they are automatically verified. A second independent review and explicit uncertainty assessment are appropriate before navigation-accuracy claims.

## Data and methods

The subset uses five LROC NAC observations from each published CNSFM scene: Chang'e 3 (S1 reserved), Apollo 11 (S2 development), Peak Near Shackleton (S3 held-out). All source observations remain within one split. No training or parameter search is performed. Three exact-transform controls per region are evaluated separately from the twelve real repeat pairs.

- **NCC:** central template, translation-only; rotation is labeled out-of-model.
- **SIFT:** ratio-filtered features and partial-affine RANSAC, up to 2000 features.
- **DISK + LightGlue:** pretrained depth-supervised DISK features (512 maximum), learned matching and the same robust geometry estimator.
- **Ablation:** each method with and without CLAHE.

Transforms map reference `(x,y)` pixel centers to query `(x,y)`. Failures stay in denominators. Endpoint errors use valid evaluation points and independent targets when provided. Timings exclude downloads/model loading, use warmup, and synchronize the selected device. Different feature budgets and a single timing repeat limit speed comparisons.

The shared map grid is an external reference, not guaranteed subpixel truth. Gain/gamma perturbations do not reproduce moving lunar shadows. No physical rendering, additional DEM, or reproduction of LIMA is claimed.

## Repository guide

[AI_PLAN.md](AI_PLAN.md) defines the agent workflow; [MILESTONES.md](MILESTONES.md) preserves acceptance criteria; [STATUS.md](STATUS.md) distinguishes implementation from scientific validation. [THIRD_PARTY.md](THIRD_PARTY.md) records source/model notices. Code is MIT licensed; third-party data and weights retain their own terms.

References: [CNSFM paper](https://doi.org/10.3390/rs17132302), [source dataset](https://github.com/Bin501/CNSFM), [LightGlue](https://arxiv.org/abs/2306.13643), [DISK](https://arxiv.org/abs/2006.13566), [Adnan Ansar's public research](https://www-robotics.jpl.nasa.gov/who-we-are/people/adnan_ansar/).
