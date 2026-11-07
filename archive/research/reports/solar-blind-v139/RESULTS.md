# Unknown-light recovery: software passes, full comparison exceeds its cap

The full-cost controls recovered all six generated poses on both flat and layered
geometry. They measured 46.06 and 51.88 seconds per complete comparison pair.
The frozen eight-case forecast is **652.51 seconds**, after the 1.5 safety factor
and 30-second scoring/guard allowance. The timing worker itself took 98.06 seconds
and peaked at 867,209,216 bytes (whole process, not a per-method memory ranking).

The remaining original deadline could not cover that forecast, so the controller
stopped **before lunar inference**, exactly as the public protocol required.
There are no new lunar predictions or accuracy results; a cost refusal is not
zero poses on eight lunar cases. No smaller subset or free retry was substituted.

The unchanged-source [public tests](https://github.com/ThomasdeChillaz/lunar-image-matching/actions/runs/42130127253)
pass 1,499 tests with eight explicit skips and a fresh 20-cell smoke. The complete
local suite passes 1,503 with four skips. The generated controls check software and
cost; their random image patterns are not physical lighting simulation.

## What this establishes

Image-selected shared catalogue illumination, native RGB16 handling, camera
initialization and all four acquisition policies work together on the controls.
The current implementation is too expensive for the 480-second recovery envelope.
It has not established lunar recovery, correct acceptance, an active-policy gain
or superiority to modern matchers under unknown illumination.

V137 remains the latest measured acquisition result: active 8/16 raw poses versus
Direct 4/16 and equal-search uniform/geometry 7/16, on supplied-light images. Its
one-pose isolated gain remains inconclusive. The clean RGB16 cohort proposed here
uses a different image pipeline, so those counts cannot be carried across.

## Accounting and next step

The approved 2,400-second supplement now has 120 + 1,800 + 480 seconds in fixed
reservations. A reservation remains charged when a job finishes early or fails its
cost gate. This is conservative admission accounting, not a statement that all
2,400 seconds were actually consumed. Original training/comparison accounting is
unchanged; there is no paid compute or calibration/final access.

A [concrete continuation](../solar-blind-v140/NEXT_EXPERIMENT.md) keeps all matching
settings, six methods and eight cases unchanged. It needs a separate resource
decision. All 14 research requirements and the broader project remain active.

Evidence: [timing](timing.json), [public execution receipt](result.json),
[verification](verification.json), [frozen protocol](protocol.json).
