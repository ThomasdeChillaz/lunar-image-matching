# Data and evaluation protocol

Use separate dataset_kind values: procedural, real_image_synthetic_transform, real_repeat_observation. Never aggregate these into a single headline result.

A future real-data manifest must record pair_id, region_id, split, dataset_kind, reference/query product IDs, source URLs, acquisition dates, solar incidence/azimuth where available, pixel scale and coordinate reference system, footprint, local paths, SHA256, nodata/mask paths, processing recipe/version, license/use conditions, and ground-truth type with uncertainty. Missing metadata stays null rather than guessed.

For repeat imagery, preserve at least 10 spatially distributed independently annotated correspondences per pair when feasible. Use multiple annotators or a checked geospatial reference to assess uncertainty. Exclude ambiguous points before freezing evaluation. Separate resampling errors from matcher errors. A single homography/affine model may not explain terrain relief and viewpoint differences.

Real-data acquisition starts from official product catalog entries, not rendered website screenshots. Do not use LOLA altimetry as if it were camera imagery. Confirm current LROC/LOLA product availability and read their documentation before downloading. Git tracks manifests and recipes, not large raw rasters.

The procedural smoke benchmark has no geographic train/test claim and performs no hyperparameter search. It exists to catch implementation errors. Synthetic image patches at multiple seeds are not independent lunar sites.
