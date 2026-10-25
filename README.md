# Lunar image matching

A reproducible research project studying image registration under changing appearance, with an explicit separation between procedural smoke tests, controlled transformations of real imagery, and real repeat lunar observations.

**Status:** implementation and milestone progress are tracked in [STATUS.md](STATUS.md). Procedural results are software checks, not lunar-science findings. No NASA/JPL affiliation or endorsement is implied.

## Start

Python 3.10 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python benchmark.py --config smoke.json --output runs/smoke
```

The command writes per-pair results, a summary, configuration/environment metadata, three plots and a gallery. It refuses to overwrite an existing run. Transforms map reference `(x,y)` coordinates to query `(x,y)` coordinates. Error is median endpoint error on valid evaluation points, in pixels. Failures count against success rates and are not silently discarded.

## Research path

Read [the AI execution plan](AI_PLAN.md), [milestones](MILESTONES.md), and [data protocol](DATA_PROTOCOL.md). The default benchmark uses generated textures only. Gain/gamma changes are not physically simulated solar illumination or moving shadows.

For a real grayscale source image, use `--image path/to/image.png`; the resulting known-transform experiment is still synthetic geometry, not repeat-observation validation. Preserve product IDs, source URLs, checksums and preprocessing before interpreting it scientifically.

## Methods

- Normalized cross-correlation: central template, translation-only baseline.
- SIFT + ratio filtering + RANSAC partial-affine estimation: rotation/translation/uniform-scale model.
- CLAHE ablation, independently applied to both images.
- Learned matching and real repeat-observation evaluation are later milestones, not claimed results.

References: [OpenCV template matching](https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html), [OpenCV geometric matching](https://docs.opencv.org/4.x/d7/dff/tutorial_feature_homography.html), [LightGlue](https://github.com/cvg/LightGlue), [LRO data](https://pds-geosciences.wustl.edu/missions/lro/lola.htm), [Adnan Ansar's public research](https://www-robotics.jpl.nasa.gov/who-we-are/people/adnan_ansar/).
