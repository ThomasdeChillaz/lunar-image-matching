# Preserve decisions and repair rejected-match diagnostic records

The v133 study performed the full prediction loop but failed while writing its
results. A rejected peak had no finite competing peak, represented as an infinite
margin; strict JSON serialization refused that diagnostic. No valid prediction
seal, scoring, metric result or pose advantage exists for that attempt. Preserve
its malformed output, logs, original deadline and full 900-second charge.

The only correction replaces a positive-infinite margin on an already rejected
measurement with `null` and an unavailable-comparison flag. Its rejection reason,
accepted/null endpoint and every numerical matching decision remain unchanged.
Unexpected nonfinite accepted diagnostics still raise an error. A regression
test reproduces an isolated supported peak with real masked correlation, verifies
that it remains rejected, and checks strict JSON output. Finite outputs are
unchanged. Historical v132/v133 sources stay byte-for-byte intact.

Repeat exactly the same twenty method/solver combinations, sixteen positives and
sixteen certified nonoverlaps from [v133](../solar-local-v133/METHOD.md). Same
images, reference libraries, 36-anchor budgets, physical camera optimization,
AP3P/refit, three-native-pixel threshold and scoring. All seven historical
controls must still replay exactly, and frame controls must preserve endpoints.
No tuning, model selection or query-truth use is added. Calibration/final groups
stay closed. Matching illumination is supplied; three terrain components and
repeated noise variants do not establish fresh geographic generalization.

The failed full-inference process took 378.34 wall-seconds. Its saved stack trace
identifies the final prediction dump after the complete loop. A 1.5-times forecast
plus 30 seconds is 597.51 seconds. A prospective 600-second repeat therefore fits
the remaining 695-second original baseline envelope. No old charge is refunded,
deadline reset, partial prediction scored or method silently substituted. Publish
this source, protocol and measured admission before repeating inference. Seal
valid predictions before scoring and independently replay all saved metrics.
