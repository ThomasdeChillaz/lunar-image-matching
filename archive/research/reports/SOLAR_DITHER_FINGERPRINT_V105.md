# Blender's fixed dither pattern in archived lunar matching failures

The exposure experiment produced two wrong poses with 1,890 and 1,862 correspondences. Every correspondence stayed within one pixel of zero image displacement, despite camera motion. The resulting camera locations missed the query truth by 192 m and 203 m. A subsequent controlled render reproduced the failure mechanism with two featureless emission planes. Default Blender dithering plus exposure normalization produced 2,025 same-coordinate matches. Disabling dithering produced zero. [Saved pose replay](SOLAR_IDENTITY_LOCK_V104.md) · [Controlled renderer test](SOLAR_PNG_DITHER_CONTROL_V105.md)

A separately frozen pixel audit links that procedural pattern to the archived failure images. It thresholds the original native grayscale codes at 2 and compares their binary masks to the low-radiance procedural fixture. The audit checked both the selected reference template and the query. It used existing development cases and performed no matching, pose fitting or rendering.

| Sealed case | Fixed-coordinate binary correlation, reference / query | Fixture high pixels missing from either archived high mask |
|---|---:|---:|
| u025, query light 0 | 0.7214 / 0.7216 | 0 of 131,137 per image |
| u027, query light 1 | 0.7218 / 0.7215 | 0 of 131,137 per image |
| u061, query light 17 | 0.7161 / 0.7166 | 0 of 131,137 per image |

Across these six archived images, all 131,137 high pixels in the default-dither fixture also had code 2 or higher in each archived image. Correlations at the four fixed displaced offsets were between −0.00334 and −0.00028. The explicit-zero fixture was uniform, with undefined correlation. Independent arithmetic replay recovered all 180 contingency rows from their four counts.

Two bright correct controls also contained the code-2 pattern in their darkest pixels. Thresholding those images above each image's median removed detectable agreement with the procedural mask. The dim u061 case shared the low-code signature and had only six matches. The fingerprint links the encoding pattern to the images, while brightness and useful scene structure determine whether it dominates the matcher. A prospective rejection threshold would require independent data.

This is an experimental reliability finding. The procedure exposes a way that apparent feature matches and tiny fitted residuals can arise from a benchmark's image encoding. A clean next pose comparison should use separately frozen high-bit-depth, dither-free images, then simulate any sensor noise independently for each exposure. Keep the original benchmark as a reproducibility record. The existing comparison with MatchAnything is conditional on the historical simulated inputs, and real sensor transfer remains unverified. [State-of-the-art and transfer context](LUNAR_SOTA_AND_TRANSFER_V104.md)

[Frozen image hashes and metrics](solar-dither-fingerprint-protocol-v105.json) · [Full pixel audit and control rows](solar_dither_fingerprint_result_v105.json)
