# Is the surviving low-light contrast shared across views?

The v107 audit found a sparse bright tail in the clean 16-bit rendering of case u061. This follow-up measures whether a native image window at an independently projected terrain anchor has detectable contrast in both views. The reference uses the physical query illumination for this diagnostic. The same-light reference is selected from the sealed v106 render receipt using the known Sun label.

The protocol was frozen before decoding the eight existing images. It binds the public v107 predecessor at `dba59e49c974e2feca40c4c7b25e4e2d12e042a8`, the v106 images and historical exact visibility labels, the 676 original native proposal-grid centres, the window statistics and the four fixed shifts. It uses the v107 scene-linear conversion and quantization reference `sigma = 1 / (255 * 12.92 * sqrt(12))`. The primary contrast statistic is the 95th minus the 5th percentile in each 33 × 33 native window. Both views must reach three sigma. The 99th minus 5th percentile is a fixed sensitivity to smaller bright fractions. The threshold is a synthetic quantization reference and has no calibrated photon or read-noise model.

| Model-selection case | Visible centres of 676 | Aligned windows inside both images | P95−P05 at least 3 sigma in both | P99−P05 at least 3 sigma in both |
|---|---:|---:|---:|---:|
| u061, sparse tail | 648 | 629 | 1 | 616 |
| u013, bright control | 660 | 648 | 443 | 567 |
| u025, near-uniform control | 632 | 605 | 0 | 0 |
| u027, near-uniform control | 664 | 652 | 0 | 0 |

The one primary u061 anchor is in the lower-right 128 × 128 tile in both images. On the 547 centres where every shifted window also fits, primary co-visibility is zero for the aligned and all shifted windows. The tail-sensitive aligned count is 536 of 547, compared with 520, 538, 535 and 537 under the four ±32-pixel shifts. The tail statistic is widespread within u061 at this scale and does not distinguish the true alignment from these nearby shifts by its presence alone. The near-uniform controls have zero co-visible windows even at one sigma under both statistics. In the bright control the primary aligned count is 423 of 599 on common shift support, versus 379, 420, 405 and 389 for the shifts.

The visibility labels apply to window centres. The two 33 × 33 windows are sampled in native camera coordinates with no patch warp or per-pixel geometric correspondence. Overlapping windows and nearby shifts are dependent. These measurements cannot tell whether the exact arrangement of the few brightest pixels carries a usable shape cue, or whether camera localization is possible on new terrains or sensors. No image matcher or pose solver ran. The four cases are previously exposed development data.

The measurement used 1.00 second of local wall time. The independent replay used explicit extraction and order statistics and reproduced the 24 aligned and common-support count tables and the primary tile counts in 0.67 second. Both used only model-selection inputs.

Reproduce from the repository root with the sealed v106 render and scoring artifacts in place. The scripts create new result files and keep the archived outputs available for comparison.

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  python audit_clean_covision_v108.py measure \
    --protocol reports/solar-clean-covision-protocol-v108.json \
    --protocol-sha256 ccd1d2d4220f3ef118703b42f57109328df0671d271e0e5ae044d71ba3fc86e0 \
    --output runs/solar_clean_covision_v108_repeat.json
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  python replay_clean_covision_v108.py reports/solar-clean-covision-protocol-v108.json \
    runs/solar_clean_covision_v108_repeat.json runs/solar_clean_covision_v108_repeat_replay.json
```

| Artifact | SHA-256 |
|---|---|
| Frozen protocol | `ccd1d2d4220f3ef118703b42f57109328df0671d271e0e5ae044d71ba3fc86e0` |
| Measurement source | `f1edab8fb0f6df2a27ac40b5387e90ccee9215e6d8dcc95b93180f1820205ade` |
| Measurement result | `d92ab62049b149cc9188d1267d4add78bfca892cd1c95b7cf085099d0c57f4c5` |
| Replay source | `1ac7f13c438628d73b47d5bdfc374adb7e0ca490fd2ea419dc17fabf55865e8f` |
| Replay result | `3c4262e6eda60233ef49f245ceb0f138d7674eaad1189b5163e8e728a28eb403` |
