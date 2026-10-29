# Third-party sources

## Learned model
LightGlue source is pinned to `eb42fee2d71449efb0aa5c10549752b5d75384d8` in requirements-learned.txt. Its inference code and pretrained LightGlue weights are Apache-2.0: https://github.com/cvg/LightGlue/blob/eb42fee2d71449efb0aa5c10549752b5d75384d8/LICENSE . Citation: Lindenberger, Sarlin and Pollefeys, *LightGlue: Local Feature Matching at Light Speed*, ICCV 2023, https://arxiv.org/abs/2306.13643 .

DISK features use Kornia's wrapper around the depth-supervised pretrained DISK model. Upstream code/weight source: https://github.com/cvlab-epfl/disk , revision `8dc6d4d65b47210f629e21ad1fbfb56944b11696`, Apache-2.0. Citation: Tyszkiewicz, Fua and Trulls, *DISK: Learning local features with policy gradient*, NeurIPS 2020, https://arxiv.org/abs/2006.13566 . Kornia: Apache-2.0, https://github.com/kornia/kornia .

DISK replaces the originally suggested SuperPoint extractor because the upstream LightGlue documentation flags a distinct restrictive license for SuperPoint. This project does not redistribute either model's weight files. Model URLs and SHA256 values are pinned in model_lock.json. Cached files are checked before loading; downloads are an explicit command.

## Lunar observations
CNSFM research subset: source revision and source image metadata in lunar_manifest.json; citation and use conditions in DATASET_CARD.md. The source repository supplies no dataset-specific license. Source rasters, executable and crater-detector weights are not bundled. The published paper is the source for the three site identities. We do not use its executable or crater detections to produce evaluation references.

## Software
NumPy (BSD), OpenCV (Apache-2.0), Matplotlib (PSF-based), PyTorch/torchvision (BSD), Rasterio (BSD), Pillow (HPND), pytest (MIT), ReportLab (BSD), Requests (Apache-2.0). See the installed distributions for exact notices; no licenses are replaced by this project's license.
# Research extension additions

- **MoonAnything / LunarPhoto**, Grethen et al. (2026), [Zenodo record 18415119](https://zenodo.org/records/18415119), CC BY 4.0. The experiment uses the authors’ physical Hapke lighting renders and original metadata. ZIP-member CRC32, selected bytes and SHA256 are retained; the complete outer archive MD5 is not claimed. Original archives/rasters are downloaded separately and excluded from Git. Cite the authors and dataset when sharing figures or adapted results.
- **Original CNSFM executable and crater checkpoint**, [Bin501/CNSFM](https://github.com/Bin501/CNSFM), revision `94cebaaa7fd3e070f8040620975554c3067651ef`. Downloaded separately; neither the Windows executable/DLLs nor pretrained detector weights are redistributed in this repository. The release supplies no explicit software/model-specific license. Native execution is recorded as original-author software; common-grid redetection is a documented preprocessing variant.
- **YOLOv9 optional detector source**, [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9), revision `5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff`, GPL-3.0. The optional `cnsfm_detector.py` integration is marked GPL-3.0-only; upstream source stays external. Other original project files retain their existing notices. The author crater-checkpoint SHA256 is `498a647d5c80d7a5a9c4f151c264c98f762db3a0f1574ff05f5ae8041bbfd860`.

