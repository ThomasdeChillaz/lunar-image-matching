# Lunar image matching: reproducible pilot study

## Research question
How do classical and pretrained learned matchers behave when real lunar terrain is observed under different illumination, and how does that differ from performance on controlled warps?

## Data and frozen protocol
15 published LROC NAC map-projected crops from CNSFM, 5 per region; 12 anchored repeat-observation pairs. Entire regions are assigned S1 reserved, S2 development, S3 held-out. No fine-tuning, parameter search or test-driven selection was performed. See lunar_manifest.json for source IDs, acquisition times, angles, CRS, bounds and SHA256. Each pair is resampled independently to a shared 512px map grid with source nodata masks. All 12 pairs were visually inspected for terrain overlap; independent checkpoint accuracy is not established.

Three methods: NCC (translation), SIFT + partial-affine RANSAC, and pretrained DISK + LightGlue + the same estimator. None/CLAHE preprocessing is evaluated for every method. NCC rotation cases are explicitly out-of-model and separately identified in summaries. Feature budgets differ: SIFT is capped at 2000, DISK at 512, so this is a practical preset comparison, not a matched-feature-count experiment. Learned inference is warmed on procedural input and CPU timing excludes downloads/model loading. One repeat per condition limits timing precision.

Controlled tests use one actual source image per region and three known warps (translation, intensity change, rotation). They have exact geometry but do not simulate moving shadows. Repeat-observation metrics use the authors' map coordinates as an external reference. Map uncertainty is unknown; disagreement with that reference is not true registration/navigation error. No outcome should be claimed as reproduction of LIMA.

## Held-out S3 measurements

| Experiment | Method | Preprocessing | Pairs | Estimated | Within 3px | Median scored discrepancy (px) |
|---|---|---|---:|---:|---:|---:|
| Exact controls | ncc | none | 3 | 3 | 2 | 0.00 |
| Exact controls | ncc | clahe | 3 | 3 | 2 | 0.00 |
| Exact controls | sift | none | 3 | 3 | 3 | 0.03 |
| Exact controls | sift | clahe | 3 | 3 | 3 | 0.02 |
| Exact controls | disk_lightglue | none | 3 | 3 | 3 | 0.05 |
| Exact controls | disk_lightglue | clahe | 3 | 3 | 3 | 0.06 |
| Map reference | ncc | none | 4 | 2 | 0 | 4.61 |
| Map reference | ncc | clahe | 4 | 2 | 0 | 4.61 |
| Map reference | sift | none | 4 | 1 | 0 | 4.80 |
| Map reference | sift | clahe | 4 | 1 | 0 | 4.80 |
| Map reference | disk_lightglue | none | 4 | 3 | 0 | 4.83 |
| Map reference | disk_lightglue | clahe | 4 | 3 | 0 | 4.92 |

## Findings and limits
Known-transform controls establish implementation behavior on actual lunar textures. Repeat observations expose major illumination/shadow changes and failures absent from simple intensity warps. A returned transform, a high inlier count, or agreement with the source map alone does not establish correct navigation. The current subset is a pilot: only one held-out geographic region and four held-out repeat pairs. Pairs share a reference image and are statistically dependent. Do not report pair-wise confidence intervals as though these were independent sites.

Source map accuracy is not provided, and a partial-affine model may not capture relief/viewpoint distortion. The checkpoint-review page allows independent, distributed point annotation; validated points with picking uncertainty can replace the map reference in the same evaluation pipeline. That scientific validation remains open. No separate DEM or physical rendering is used.

## Reproduction
Use Python 3.12; install requirements-lock.txt, then requirements-learned.txt without dependency upgrades. Run `python dataset.py fetch`, `python fetch_models.py`, and `python evaluate.py --output runs/reproduction`. Each run refuses overwrites and saves frozen inputs, source hashes, environment, per-pair JSON/CSV, split-specific summaries, three plots, a gallery, this report and the PDF. The committed results preserve the actual executed source and manifest hashes. Hardware-dependent timings may vary.

## References
Li et al., Remote Sensing 17(13), 2302 (2025), https://doi.org/10.3390/rs17132302; data https://github.com/Bin501/CNSFM . Lindenberger et al., LightGlue, ICCV 2023, https://arxiv.org/abs/2306.13643 . Tyszkiewicz et al., DISK, NeurIPS 2020, https://arxiv.org/abs/2006.13566 . Exact source revisions, licenses and weight hashes: THIRD_PARTY.md and model_lock.json.
