# Completed training-bank qualification — 4 November 2026

The new coarse-feature bank is complete and independently verified. This is
training-input evidence, not improved camera localization or a selected model.

| Completed stage | Actual wall time | Original full charge | Result |
|---|---:|---:|---|
| Teacher-domain qualification | 8.986 s | 600 training s | 24/24 exact feature comparisons; maximum difference 0 |
| Four-camera bank cost pilot | 11.420 s | 72 training s | All 246 original point rows retained |
| Full 64-camera training bank | 65.214 s | 366 training s | All 3,933 rows / 16 training groups retained |

The full-bank cap was fixed before extraction using twice the measured pilot
cost projected to all 64 cameras. Four successful pilot records were reused
byte-for-byte; 60 were computed afresh. Whole reservations remain charged.
The canonical remaining allowances at this milestone are **14,829 training
seconds and 49,080 baseline seconds**.

Only even-indexed lighting images were read. Odd teacher slots stay unavailable
and NaN. Reference-input support covers all 3,933 rows; query support covers
3,463. The 470 missing query feature targets are explicit receptive-field or
geometry-support exclusions, separate from geometric occlusion, darkness and
the historical bank's 163 earlier omissions. No rows were silently removed.

The new targets use the frozen descriptor on complete 128-pixel fields made by
area-resizing the 512-pixel inputs. They match the coarse inference domain;
they do not qualify the native 33-pixel refinement domain. The generated
teacher check obtains 248/256 anchors within 16 native pixels among eight
candidates. It uses a newly seeded noise image and establishes software
behavior, not lunar accuracy or a gain over an earlier texture control.

The view prior weights groups equally, then camera pairs, then retained points,
using query-view training labels once per point. Repeated lighting does not
increase a point's weight. Independent saved-only replay reconstructs its mean
and covariance exactly. Query pose/depth/view labels remain training or scoring
data, and are not primary inference inputs.

The candidate has 95,116 total parameters, including the frozen coarse encoder,
with 66,844 trainable parameters. Its fixed 100-update training pilot and exact
replay are the next step. No new candidate training, pose improvement,
calibrated acceptance, modern-model advantage or final evaluation is claimed at
this bank milestone. Four calibration and twelve final terrain groups remain
closed. All earlier positive and negative results remain unchanged.

Reproduction and evidence:

- [Published full extraction protocol](solar-coarse128-bank-full-v88.json)
- [Exact full bank manifest](../runs/solar-coarse128-bank-v88/full/manifest.json)
- [Original timing and reservation](../runs/solar-coarse128-bank-v88/full/watchdog.json)
- [Independent full-bank and prior audit](solar_coarse128_bank_full_review_v88.json)
- [Earlier teacher qualification, a historical snapshot](SOLAR_TEACHER_QUALIFICATION_V87.md)

The full bank manifest SHA-256 is
`e08adbd5ff997e8c67b33ae2556cabe6af714ccaedc4ca327f2cd5e5576b6085`.
The protected 49 learning sources and 42 physical sources remain byte-identical.
