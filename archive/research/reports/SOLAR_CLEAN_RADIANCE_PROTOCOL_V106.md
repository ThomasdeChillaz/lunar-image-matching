# Prospective dither-free low-light pose comparison

This targeted development pilot asks whether the severe pose failures reproduced under the original 8-bit image encoding persist when the same camera, terrain and illumination are recorded without shared dither. It separates dither removal from increased intensity precision and checks sensitivity to an independent noise realization per image. The selected cases were already observed and include deliberately difficult failures. Results from this cohort will guide design and cannot estimate deployment error or establish a state-of-the-art rank. [Measured encoding failure](SOLAR_DITHER_FINGERPRINT_V105.md)

## Fixed image and pose controls

The four positives are u013, u025, u027 and u061 with the exact lighting and camera conditions in the frozen cohort. Each keeps its original nonoverlap partner, for four positive and four nonoverlap pairs. Four reference cameras each retain all 24 original Sun directions. Eight distinct query exposures bring the total to 104 native 512-pixel renders. Query Sun direction and true query geometry are unavailable to the matcher. The native camera and geometry are independently checked against the archived ray arrays.

The renderer uses the historical Blender executable, scene, lighting, camera, PNG output path and CPU thread limit. It renders every exposure once. The original-style RGB8 image is checked against the archived pixels. Dither-free RGB8 and RGB16 use the same Render Result. A default RGB8 save through that writer checks whether changing the output API introduced a pixel difference. Both dither-free versions retain the original Standard display transform. [Blender image export API](https://docs.blender.org/api/4.5/bpy.types.Image.html#bpy.types.Image.save_render)

Every new pose arm uses the frozen exposure percentile, 24 template visits, eight candidate pose calls and one final common pose call. RGB16 luminance is retained before scaling and converting to the SIFT input. The fourth arm adds fixed independent scene-linear read noise as an engineering sensitivity check, with a scale tied to the RMS of one near-black sRGB code's quantization error. It does not represent a measured flight camera noise model.

The independent replay reports native correspondence precision, pose error, strict correctness, wrong poses and nonoverlap outputs. Archived v101 results remain separate. A causal comparison requires the fresh default writer, original archive pixels and camera geometry to pass their parity checks. A frozen final result will account for all 32 method–case cells, including failures.

## Admission

The archived exposures took 379.81 seconds to render, and their native geometry checks took 20.07 seconds. Twice that measured total fits the 900-second render bound. The whole render, pose and replay pilot has one 1,200-second reservation under the existing canonical baseline budget, with the original two-thread policy. The held-out calibration and final terrain are outside this cohort.

[Frozen cohort and image identities](solar-clean-rerender-cohort-v106.json) · [Dispatch and limits](solar-clean-radiance-dispatch-v106.json)
