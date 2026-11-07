# Proposed continuation: finish the frozen unknown-light comparison

Status: prepared, **not authorized or launched**. No allowance is created here.

Request: up to **900 additional local CPU wall-seconds (15 minutes)**, two threads
per scientific job, no paid compute. This is separate from the fully reserved
2,400-second supplement; none of its charges is refunded or reused.

## Concrete experiment

Use the exact published v138/v139 matching and scoring sources at commit
`aaa3d02ed73667c352eaa92f4d142e4adb2c7b6c`, with all 101 source and 125 input hashes
from the unchanged v139 protocol. No model fitting, confidence changes, new
renderer settings or new case selection.

Run the complete eight clean RGB16 cases (four positives and four nonoverlaps),
with the exact six arms: Direct, joint without extras, and uniform, appearance,
geometry and active selection with 32 extra patch searches each. All methods
select reference illumination from the same 24-image catalogue. Query geometry,
pose and Sun metadata remain scoring-only. Rejected/uncalibrated public poses
remain null; saved raw hypotheses are research outputs.

Repeat the two full-budget generated controls first, preserving 24 lights, 2,048
reference camera candidates and all six arms. The previous complete measurements
were 46.06/51.88 seconds, with a 652.51-second eight-case forecast after a 1.5
safety factor and 30-second guard/scoring margin. Repeating the controls and that
forecast totals about 751 seconds, fitting the proposed 900-second cap. Admission
must use the new measurements and original shared deadline; if they do not fit,
stop before lunar inference. No smaller substitute or free retry.

## Guards and deliverables

Before execution, publish the new controller/protocol. A new private ledger can
be issued only after an explicit user resource decision; bind it to the original
unchanged cumulative journal and the fully charged supplement. Reserve one fixed
900-second cap, retain one serial lease and watchdog across every stage, and
preserve failures and incomplete outputs. Do not alter the v139 failed attempt.

Seal all eight predictions before scoring, independently replay all 48 cells and
report raw/strict poses, wrong positive/nonoverlap outputs, native correspondence
precision, lighting proposal error, full latency and memory. Compare active with
the strongest equal-search control as well as Direct. Group uncertainty by terrain
and both negative endpoints; four positives on three terrains cannot establish a
1% risk or meet fresh-test requirements.

Publish measured outcomes and reproducible artifacts, including negative findings.
No calibration/final terrain, automatic email or paper submission. Historical
default-dither modern scores are not a clean matched ranking. All original 14
research targets remain unchanged.

## Prepared implementation

[The continuation controller](continuation_v140.py) calls the unchanged v139
engine and independent replay. Its four guard tests cover absent/unapproved
authority, scope expansion, original deadlines, full-budget timing and sealed
prediction requirements. Freezing is metadata-only:

```sh
python reports/solar-blind-v140/continuation_v140.py freeze
```

Execution additionally requires exact published source bytes and the hash of a
new private approval ledger recording the actual user decision. Neither this
document, the prepared protocol nor source-only checkouts grant that permission.
