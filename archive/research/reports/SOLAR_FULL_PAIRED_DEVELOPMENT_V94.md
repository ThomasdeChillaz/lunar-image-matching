# Lunar localization on the full development set

The fixed primary track includes 192 positive and 192 nonoverlap pairs across eight terrain groups. All reported poses are raw outputs. Same-light and changed-light slices use the declared light indices. Correct poses meet 2° rotation and 1% relative translation thresholds. Strict poses meet 0.5° and 0.1%.

| Method | Light indices | Correct / positive | Strict / positive | Wrong / positive | No pose / positive | Raw pose / nonoverlap | Median inference (s) | Median input + inference (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| reference_camera_render_match | overall | 114/192 | 14/192 | 18/192 | 60/192 | 0/192 | 1.293 | 1.317 |
| reference_camera_render_match | same light index | 40/64 | 4/64 | 6/64 | 18/64 | 0/64 | 1.338 | 1.405 |
| reference_camera_render_match | changed light index | 74/128 | 10/128 | 12/128 | 42/128 | 0/128 | 1.286 | 1.294 |
| SIFT2048 | overall | 41/192 | 4/192 | 6/192 | 145/192 | 0/192 | 0.120 | 0.131 |
| SIFT2048 | same light index | 41/64 | 4/64 | 5/64 | 18/64 | 0/64 | 0.125 | 0.138 |
| SIFT2048 | changed light index | 0/128 | 0/128 | 1/128 | 127/128 | 0/128 | 0.116 | 0.127 |
| MatchAnything-EfficientLoFTR-native512 | overall | 42/192 | 3/192 | 84/192 | 66/192 | 51/192 | 3.820 | 3.829 |
| MatchAnything-EfficientLoFTR-native512 | same light index | 40/64 | 3/64 | 15/64 | 9/64 | 20/64 | 3.900 | 3.909 |
| MatchAnything-EfficientLoFTR-native512 | changed light index | 2/128 | 0/128 | 69/128 | 57/128 | 31/128 | 3.777 | 3.790 |

| First minus second | Light indices | Correct-pose difference | 95% terrain interval | Raw nonoverlap difference | 95% two-endpoint interval |
|---|---|---:|---:|---:|---:|
| reference_camera_render_match − SIFT2048 | overall | +38.02 pp | [+20.83, +54.17] pp | +0.00 pp | [+0.00, +0.00] pp |
| reference_camera_render_match − SIFT2048 | same light index | -1.56 pp | [-4.69, +0.00] pp | +0.00 pp | [+0.00, +0.00] pp |
| reference_camera_render_match − SIFT2048 | changed light index | +57.81 pp | [+32.81, +81.25] pp | +0.00 pp | [+0.00, +0.00] pp |
| reference_camera_render_match − MatchAnything-EfficientLoFTR-native512 | overall | +37.50 pp | [+19.27, +55.73] pp | -26.56 pp | [-52.08, -4.17] pp |
| reference_camera_render_match − MatchAnything-EfficientLoFTR-native512 | same light index | +0.00 pp | [-17.19, +14.06] pp | -31.25 pp | [-62.50, +0.00] pp |
| reference_camera_render_match − MatchAnything-EfficientLoFTR-native512 | changed light index | +56.25 pp | [+32.03, +80.47] pp | -24.22 pp | [-49.48, -6.25] pp |
| SIFT2048 − MatchAnything-EfficientLoFTR-native512 | overall | -0.52 pp | [-7.29, +4.17] pp | -26.56 pp | [-52.08, -4.17] pp |
| SIFT2048 − MatchAnything-EfficientLoFTR-native512 | same light index | +1.56 pp | [-15.66, +15.62] pp | -31.25 pp | [-62.50, +0.00] pp |
| SIFT2048 − MatchAnything-EfficientLoFTR-native512 | changed light index | -1.56 pp | [-3.91, +0.00] pp | -24.22 pp | [-49.48, -6.25] pp |

Medians retain every case. Inference includes the matcher and common pose solve; input-plus-inference includes measured loader setup. Calendar time also includes replay, scoring and I/O.

Descriptive percentile bootstrap with eight terrain groups. A positive uses its terrain multiplicity and a negative uses the product of both endpoint multiplicities.

Terrain and directed negative-endpoint tables, source identities and original timing receipts are in the JSON companion.
