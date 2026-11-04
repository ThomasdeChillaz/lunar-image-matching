# A matched supplied-Sun illumination synthesis control

The public LIMA description uses two co-registered references with known illumination to form a contrast template for a supplied Sun vector. We implemented the horizontal Lambertian component of that idea for the separate known-Sun track. This benchmark control uses the project’s native SIFT and common AP3P pipeline in both arms, which isolates the template construction. [LIMA paper](https://doi.org/10.2514/6.2025-2073) · [Technical description](https://patents.justia.com/patent/20260219064)

For a pair of even-index reference Sun vectors, the method solves the horizontal 2 × 2 system to reproduce the target Sun’s east and north components. It requires nonnegative weights and a horizontal sine determinant of at least sin(5 degrees). It minimizes the larger of the two Sun-angle differences and breaks ties by coefficient norm and source order. Under an unclipped Lambertian response, the difference between the weighted image and the target response is the residual vertical coefficient times the terrain normal’s up component, multiplied by a radiometric scale. Where the up component is locally constant, this contributes a constant intensity offset. Cast shadows, varying slope and clipped response can violate the approximation.

Both methods use the same inverse sRGB conversion and global linear min-to-max contrast encoding to native 8-bit codes. The nearest arm chooses the closest even reference using only the supplied Sun direction. The composite arm combines the two reference planes from the same camera. The frozen renderer uses achromatic diffuse material, Blender Standard display transform, zero exposure offset and unity gamma. Exact reference matches use the single corresponding template in both arms. [Blender color management](https://docs.blender.org/manual/en/3.0/render/color_management.html)

## Metadata support

| Frozen cohort | Exact single reference | Two-reference angular bracket | Unsupported |
|---|---:|---:|---:|
| Positive, odd query light | 0/128 | 128/128 | 0/128 |
| Positive, even query light | 64/64 | 0/64 | 0/64 |
| Nonoverlap, odd query light | 0/128 | 72/128 | 56/128 |
| Nonoverlap, even query light | 0/64 | 36/64 | 28/64 |

The nearest even reference for every withheld-light positive is 7.497 degrees from its supplied query direction. Nonoverlaps often pair separate terrain frames, which can put their supplied direction outside the reference catalogue. Full denominators and the common-support slice therefore answer different questions. An unsupported composite contributes an explicit null outcome and cannot establish image-based rejection.

A deterministic analytic control reproduced the Lambertian offset relation. A native 512-pixel SIFT and common-pose smoke test recovered a known synthetic translation in both arms. All three focused tests passed. The angular audit used metadata only, and no lunar matcher or pose study ran with this method. The estimated-Sun results remain a separate track.

- [Metadata audit with exact pairs](solar_supplied_sun_metadata_support_v102.json)
- [Original withheld-light study](SOLAR_OFFCATALOGUE_DIRECT_V95.md)
