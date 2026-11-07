# Native measurement repair with reference-only reuse

This preserves all five v127 methods, all 16 positives, all 16 certified
nonoverlaps and the 60-second timing / 600-second comparison limits.
The two prior timed-out pilots remain failures with their original charges.
The instrumented v128 trace measures 11.15/17.33 seconds for initialization,
0.49–0.67 seconds per original patch arm, 1.32–1.44 seconds per repaired
patch arm, and 8.70 seconds for the first Direct search. Its second Direct
search was interrupted by the unchanged deadline.

Cache geometry-keyed reference visibility and image/settings-keyed reference
SIFT features. Validate the exact reference world coordinates, plane, camera
bank and intrinsics before each visibility reuse. Recompute the current
query's photometric ranking, SIFT features, descriptor matches and camera
pose. Query feature or match outputs are never cached. The matching algorithms,
scores and thresholds remain those in the published v127 source.

The generated timing pilot retains flat and multidepth scenes. Each measures
both cold and warm initialization, checks identical ordered hypotheses, and
runs all five methods. Forecast all 32 pairings using the slower warm work,
four cold reference overheads, 104 reference SIFT extractions and 40 seconds
for input and scoring, then apply the original 1.5 safety factor. The full
study starts only if this fits 600 seconds. Cache misses are included in
measured inference times; cold and reference-resident timings must be
distinguished. Reference cache memory is part of process memory.

The primary accuracy check remains at least two more correct poses than the
matched Direct control without increasing wrong positive or nonoverlap
outputs. The four conditions reuse three exposed terrain groups and supplied
matching illumination. This cannot establish the original final-test or
novel learned-method requirements.

[Original repair and comparison](../solar-local-v127/METHOD.md) ·
[Exact reference caches](reference_cache.py) · [Controller](experiment.py)
