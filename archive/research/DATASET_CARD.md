# Lunar observation subset

Source: Li et al., *Robust Feature Matching of Multi-Illumination Lunar Orbiter Images Based on Crater Neighborhood Structure*, Remote Sensing 17(13), 2302 (2025), https://doi.org/10.3390/rs17132302. Data: https://github.com/Bin501/CNSFM at revision `94cebaaa7fd3e070f8040620975554c3067651ef`.

## Selection and provenance
15 real LROC NAC map-projected images, five per scene; 12 pairs use the first image as the region reference. Observations were selected at first/quartile/median/third-quartile/last positions in the authors' incidence-sorted metadata spreadsheets, before evaluating matchers. Each image covers approximately 1000 by 1000 m. All URLs, source IDs, source checksums, times, incidence/emission/phase, native pixel scales, nodata values, map footprints and CRS definitions are in lunar_manifest.json.

- S1, Chang'e 3 (44.1214 N, 19.5117 W): development/training-reserved region.
- S2, Apollo 11 (0.67416 N, 23.47314 E): development validation.
- S3, Peak Near Shackleton (88.80638 S, 123.77827 E): held-out test.

No model is trained or fine-tuned in this project. The regions and source images never cross splits. The source spreadsheet's northAzimuth column is retained as north_azimuth_deg; it is not relabeled solar azimuth. Unknown solar azimuth stays null.

## Geometry and masks
Each pair is independently resampled from its source GeoTIFF onto a shared 512x512 map grid using bilinear interpolation and original nodata=255 masks. Masks use nearest-neighbor resampling and are eroded by two pixels to suppress boundary artifacts. Zero-valued shadow pixels remain valid unless the source mask says otherwise. Neither an evaluated matcher nor the authors' crater detections is used to create the map reference. No extra DEM is used: relief/orthorectification residuals remain a limitation. Separate terrain rendering is outside this experiment.

## Reference accuracy
The identity relation on the shared map grid is an external georeferencing reference. It is **not independently verified subpixel ground truth**. The public release does not supply map-registration covariance or independent checkpoint pairs. Consequently, real-pair outputs use “map-reference disagreement,” and threshold statistics describe agreement with that reference. They must not be interpreted as true navigation accuracy. The annotation tool supports adding independent correspondences and an uncertainty estimate; it does not mark them verified automatically.

Controlled warps of each actual lunar image provide a separate, exact-transform experiment. It tests geometry and intensity perturbations; it does not generate physically correct changes of illumination or shadow direction.

## Use and distribution
The dataset is a public research release with no explicit dataset-specific license in the source repository. This project distributes metadata, download recipes and aggregate measurements; the original rasters and authors' executable/weights are not redistributed. Retrieve originals directly from their pinned source. Do not infer an unrestricted dataset license from this project's code license.
