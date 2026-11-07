# Geo-LoFTR on a native 512-pixel translation control

The official Mars-trained Geo-LoFTR checkpoint loaded and executed twice on a generated image pair. Both forwards reached the reference-depth backbone. The unchanged reference-only pose solver recovered the known translation within its frozen limits. The median correspondence error exceeded the one-pixel limit, so this qualification failed and no lunar inference ran.

| Measurement | Result | Qualification limit |
|---|---:|---:|
| Native correspondences | 3,350 | ≥6 |
| Median native correspondence error | 1.4571 px | ≤1 px |
| Rotation error | 0.05147° | ≤0.5° |
| Translation error | 0.04397 m | ≤0.05 m |
| Common pose inliers | 3,335 | ≥6 |

Blank images, all-invalid reference depth and an empty reference mask each returned an explicit empty result without running the network. The cold and warm forwards took 13.87 and 10.70 seconds on the two-thread Intel Mac. Process peak RSS was 2,547 MiB. These measurements apply to the generated fixture on that machine.

The adapter uses native query and reference pixels, reference axial depth normalized over valid pixels, and the original model's query-first convention. The source checkout is pinned to `e1590f0fe25a7d19b6b22d65c22f9305e46cf868`. The exact official simulation member `loftr/V1/geo/2024_08_29-17_28_02.pt` has SHA-256 `275b0dd1c02d63b0a303ec84b3d907e30880847fc36930f9905d3f268c6ef1ad`. The paper uses orthographic reference windows and a pose prior. This prospective perspective-depth transfer keeps the query pose and depth with the scorer. [Paper](https://arxiv.org/html/2502.09795v3) · [Pinned source](https://github.com/nasa-jpl/mbl_mars/tree/e1590f0fe25a7d19b6b22d65c22f9305e46cf868) · [Author checkpoint archive](https://huggingface.co/dpisanti/mbl-mars/blob/8c66a64949382f0b140ef65e1ae4b3b438fa0d21/weights_loftr.zip)

The [procedural archive](solar-geoloftr-v114-procedural.zip) contains the adapter, the qualification entry point, dependency and source hashes, and all saved native coordinates. Independent arithmetic reproduced the 1.4571-pixel median. The next comparison can prospectively isolate the effect of a common native refinement applied to each proposal source, while retaining this original checkpoint-only result.
