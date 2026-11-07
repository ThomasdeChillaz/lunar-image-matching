# Does the brightest low-light texture preserve its shape between views?

The earlier co-visibility audit found one robustly textured shared window in the sparse-tail case u061. The presence of rarer bright pixels was widespread after shifting the query image. This follow-up compares the location of individual bright pixels at the true terrain projection with the same four fixed 32-pixel shifts.

The protocol was frozen before decoding. The four previously exposed development pairs use the archived native cameras and dither-free RGB16 images under the same physical light within each pair. The analysis projects native reference pixels through sealed depth and checks conservative subpixel co-visibility with reciprocal depth and pixel agreement. The aligned and shifted measurements use the same supported reference pixels. The primary tail is above each view’s 99th percentile and at least three synthetic quantization standard deviations above its 1st percentile. Jaccard measures the share of pixels flagged in either view that are flagged in both. Phi also accounts for the marginal frequencies.

| Case | Common pixels | Both bright at true projection | Jaccard, aligned | Best shifted Jaccard | Phi, aligned | Best shifted phi |
|---|---:|---:|---:|---:|---:|---:|
| u061 sparse tail | 167,234 | 20 | 0.00620 | 0.00707 | 0.00265 | 0.00472 |
| u013 bright control | 163,765 | 1,412 | 0.79594 | 0.02187 | 0.88992 | 0.03259 |
| u025 near-uniform control | 167,401 | 0 | Undefined | Undefined | Undefined | Undefined |
| u027 near-uniform control | 152,670 | 0 | 0 | 0 | −0.000011 | −0.000009 |

In u061 the strict top-1% result misses its predeclared criterion. At the same percentile, the fixed one-pixel tolerance gives a small advantage in Jaccard, 0.05253 versus a maximum shifted score of 0.04718. The undilated top-2% comparison also gives a small advantage, while the top-0.5% comparison fails. The weak signal depends on spatial tolerance and the tail threshold. This motivates testing whether physical signal survives independent rendering seeds and a camera noise model before spending a pose study on those pixels.

The depth check uses exact synthetic cameras and conservative dense depth agreement. It does not cast new independent mesh rays. The known query light and camera geometry were used only for scoring. The four selected pairs and shifted samples are dependent. The result gives no fresh terrain, pose recovery or sensor evidence.

The independent replay used relative camera extrinsics and separate rank and contingency arithmetic. It reproduced all four geometry support tables and 120 comparison cells, including spatial counts.

[Frozen protocol](solar-clean-tail-morphology-protocol-v109.json) · [Complete measurements](solar_tail_morphology_v109.json) · [Independent replay](solar_tail_morphology_replay_v109.json) · [Earlier contrast audit](SOLAR_CLEAN_COVISION_V108.md)
