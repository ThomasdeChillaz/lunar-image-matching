# A paired test of dither-free lunar image encoding

Removing PNG dithering eliminated all three wrong low-light poses in the targeted pilot. Every clean arm preserved the bright case's correct pose. None recovered a correct pose on the three dark positives. The result validates an image-encoding control for simulated low-light navigation, while leaving low-light recovery open.

The published protocol fixed four previously observed positives and their four original nonoverlap partners. Each arm searched all 24 reference illuminations with native SIFT and the unchanged common pose solver. Query Sun and true query pose were unavailable to the matcher. The clean arms used the same render result as the archived-style image. One sensitivity arm added an independent, deterministic read-noise draw to each 16-bit exposure. [Prospective protocol](SOLAR_CLEAN_RADIANCE_PROTOCOL_V106.md)

| Fresh input and exposure policy | Correct poses | Wrong positive poses | Positive cases with no pose | Nonoverlaps returning a pose |
|---|---:|---:|---:|---:|
| Historical RGB8 with default dither | 1/4 | 3/4 | 0/4 | 0/4 |
| Dither-free RGB8 | 1/4 | 0/4 | 3/4 | 0/4 |
| Dither-free RGB16 | 1/4 | 0/4 | 3/4 | 0/4 |
| Dither-free RGB16 with independent read noise | 1/4 | 0/4 | 3/4 | 0/4 |

On u025 and u027, the fresh default arm reproduced the original 1,890 and 1,862 apparent correspondences within one pixel of zero motion. None were correct within two native pixels. The dither-free RGB8 and RGB16 arms returned no correspondences on those cases. The independent-noise arm returned one each, with neither correct. On u061, default dithering yielded six zero-motion matches and a wrong pose. Every clean arm returned fewer than six correspondences and no pose. The bright case remained correct even at the stricter pose tolerance in all four arms.

All 104 fresh default images were exactly equal to the archived decoded RGB pixels. Their PNG file bytes differed. The separate default writer control was byte-identical to the new original-style writer on all 104 exposures, so writer choice did not account for the clean-arm difference. Across 12 camera roles, 3,145,728 recomputed native camera rays and their archived arrays agreed exactly. The fresh default arm also exactly reproduced the eight archived v101 correspondence and pose arrays. All 32 method–case records passed independent score replay.

Rendering and export took 455.61 seconds. The full two-thread render, inference and replay took 581.99 seconds under the single 1,200-second reservation. This small cohort was chosen after observing the failures. Its noise arm uses one predeclared sensitivity setting, and its results cannot estimate a sensor's error rate or rank the method against modern matchers. The earlier matched comparison with MatchAnything used the historical simulation and is a separate result. [Earlier benchmark context](LUNAR_SOTA_AND_TRANSFER_V104.md)

[Render receipt](solar_clean_radiance_render_receipt_v106.json) · [Independent pose replay](solar_clean_radiance_pose_replay_v106.json) · [Run and budget receipt](solar_clean_radiance_watchdog_v106.json)
