# Usable contrast in the dither-free native images

This retrospective diagnostic reads 27 sealed images across nine predeclared optical exposures. Each of the four original positive cases contributes its query and the reference-camera template at the physical query Sun index. Case u025 also contributes the different reference template selected in the archived v101 run, kept separate. The diagnostic measures optical intensity and spatial variation without correspondence alignment, matching or pose estimation. It examines the full image and a fixed central 416 × 416 crop.

The clean16 intensity spans below are the central crop's 99.5th minus 1st percentiles, in equivalent sRGB8 codes. To compare with the pilot's declared noise sensitivity, we invert the sRGB transform, subtract the crop's 1st-percentile level and divide by the pilot's linear noise standard deviation. That standard deviation is a synthetic sensitivity scale. It does not estimate noise in any real camera.

| Fixed case | Role | Clean16 intensity span, sRGB8 codes | Floor-subtracted P99.5 / noise standard deviation | 32-pixel block-mean P95−P05 / noise standard deviation | Pixels at modal clean8 level |
|---|---|---:|---:|---:|---:|
| u013 bright control | Query | 67.4981 | 675.50 | 394.057 | 42.53% |
| u013 bright control | Same-light reference | 65.7800 | 642.20 | 382.351 | 49.34% |
| u025 | Query | 0.0506 | 0.175 | 0.0094 | 99.995% |
| u025 | Same-light reference | 0.0506 | 0.175 | 0.0094 | 99.998% |
| u025 | Archived selected light 1 reference | 0.0506 | 0.175 | 0.0091 | 99.996% |
| u027 | Query | 0.0506 | 0.175 | 0.0093 | 99.993% |
| u027 | Same-light reference | 0.0506 | 0.175 | 0.0099 | 99.997% |
| u061 | Query | 8.2101 | 28.44 | 0.642 | 97.51% |
| u061 | Same-light reference | 8.3997 | 29.10 | 0.589 | 97.24% |

The two darkest cases have some variation below an 8-bit code but almost no contrast at the 32-pixel spatial scale. The full native frame gives the same result. High-bit-depth retention alone therefore offers little evidence for recovering their poses under this simulated exposure. Case u061 has a higher-intensity tail even after black-floor subtraction, but most of the image remains at the floor. The diagnostic cannot establish whether those brighter regions overlap geometrically or carry distinctive local features. The bright control verifies that the statistic responds strongly when optical contrast is present.

The frozen protocol and per-image statistics preserve exact source hashes and all measured regions. The published v106 pose result is a separate experiment.

[Frozen pixel-audit protocol](solar-clean-signal-protocol-v107.json) · [All native and crop statistics](solar_clean_signal_v107.json) · [Paired pose result](SOLAR_CLEAN_RADIANCE_RESULTS_V106.md)
