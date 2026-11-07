# Procedural test of fixed-pattern PNG dithering

With Blender's default dithering, two featureless images produced **2,025 native SIFT matches** after the frozen exposure normalization. All 2,025 were within one pixel of zero displacement. With dithering disabled, the same images produced zero matches.

The source and full matrix were published before rendering. The test reused the original Blender 4.5.9 executable, Standard color transform, native 512-pixel resolution and RGB 8-bit PNG write path. The only content was a synthetic emission plane. Two flat radiances nominally correspond to 1.5 and 2.5 sRGB codes. A repeated flat render and a smooth ramp control completed the matrix. No terrain, query pose or matching calibration data were opened.

| Dither setting | Matches before exposure normalization | Matches after exposure normalization | After, within 1 px of zero displacement |
|---|---:|---:|---:|
| Blender default, measured 1.0 | 0 | 2,025 | 2,025 |
| Explicit 0.0 | 0 | 0 | 0 |
| Explicit 1.0 | 0 | 2,025 | 2,025 |

Default and explicit 1.0 outputs matched exactly for all four stimuli. Across the two different flat radiances, the grayscale pattern correlation was 0.99933 at the same pixel coordinates and between −0.0040 and −0.0009 at the four predeclared offsets. With dither disabled, both flat images were spatially uniform and the correlation was undefined. The dithered ramp also retained a same-coordinate residual correlation of 0.613 after subtracting each column's mean, with all four shifted correlations near zero.

This isolates a causal pathway in the rendering fixture. The fixed-coordinate pattern creates matchable texture where the scene has none, and exposure normalization makes it strong enough to survive SIFT. Blender's published output-conversion implementation describes dither tied to pixel coordinates. The procedural test establishes the mechanism without assuming which archived lunar-development images contain it. [Blender 4.5 render settings](https://docs.blender.org/api/4.5/bpy.types.RenderSettings.html#bpy.types.RenderSettings.dither_intensity) · [Blender implementation change](https://github.com/blender/blender/commit/02869cc6c8630432743b75db44001951cec371d6)

Twelve renders completed in 18.41 seconds under the 120-second procedural limit. An independent arithmetic check reproduced all native histograms and the fixed-coordinate and shifted correlations. A next pose comparison should preserve the historical 8-bit images and use a separately frozen small set to test high-bit-depth, dither-free data and independent sensor-noise realizations. A match-count or inlier-count gate alone cannot detect the demonstrated artifact.

[Published protocol](solar-png-dither-control-v105.json) · [Exact results](solar_png_dither_control_result_v105.json) · [Observed pose failure](SOLAR_IDENTITY_LOCK_V104.md)
