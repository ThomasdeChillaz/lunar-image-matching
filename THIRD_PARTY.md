# Third-party sources

## Learned model
LightGlue source is pinned to `eb42fee2d71449efb0aa5c10549752b5d75384d8` in requirements-learned.txt. Its inference code and pretrained LightGlue weights are Apache-2.0: https://github.com/cvg/LightGlue/blob/eb42fee2d71449efb0aa5c10549752b5d75384d8/LICENSE . Citation: Lindenberger, Sarlin and Pollefeys, *LightGlue: Local Feature Matching at Light Speed*, ICCV 2023, https://arxiv.org/abs/2306.13643 .

DISK features use Kornia's wrapper around the depth-supervised pretrained DISK model. Upstream code/weight source: https://github.com/cvlab-epfl/disk , revision `8dc6d4d65b47210f629e21ad1fbfb56944b11696`, Apache-2.0. Citation: Tyszkiewicz, Fua and Trulls, *DISK: Learning local features with policy gradient*, NeurIPS 2020, https://arxiv.org/abs/2006.13566 . Kornia: Apache-2.0, https://github.com/kornia/kornia .

DISK replaces the originally suggested SuperPoint extractor because the upstream LightGlue documentation flags a distinct restrictive license for SuperPoint. This project does not redistribute either model's weight files. Model URLs and SHA256 values are pinned in model_lock.json. Cached files are checked before loading; downloads are an explicit command.

## Lunar observations
CNSFM research subset: source revision and source image metadata in lunar_manifest.json; citation and use conditions in DATASET_CARD.md. The source repository supplies no dataset-specific license. Source rasters, executable and crater-detector weights are not bundled. The published paper is the source for the three site identities. We do not use its executable or crater detections to produce evaluation references.

## Software
NumPy (BSD), OpenCV (Apache-2.0), Matplotlib (PSF-based), PyTorch/torchvision (BSD), Rasterio (BSD), Pillow (HPND), pytest (MIT), ReportLab (BSD), Requests (Apache-2.0). See the installed distributions for exact notices; no licenses are replaced by this project's license.
