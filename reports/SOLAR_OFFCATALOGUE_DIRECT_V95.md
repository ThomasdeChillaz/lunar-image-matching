# Direct relighting with the exact lighting templates withheld

Removing alternate lighting renders reduced changed-light recovery from 74/128 to 46/128. The restricted library still recovered more changed-light poses than either image-only baseline on the same simulated development cases.

| Method | Correct positive poses | Same-light correct | Changed-light correct | Wrong positive poses | Nonoverlap poses | Median online inference |
|---|---:|---:|---:|---:|---:|---:|
| Direct relighting, 24 lights | 114/192 | 40/64 | 74/128 | 18/192 | 0/192 | 1.293 s |
| Direct relighting, 12 even-index lights | 86/192 | 40/64 | 46/128 | 26/192 | 0/192 | 0.821 s |
| MatchAnything–EfficientLoFTR | 42/192 | 40/64 | 2/128 | 84/192 | 51/192 | 3.820 s |
| SIFT | 41/192 | 41/64 | 0/128 | 6/192 | 0/192 | 0.120 s |

The two direct arms use the same 512-pixel inputs, reference geometry, SIFT matcher, eight candidate pose solves and final 3-pixel pose estimator. The restricted arm filters all 32 reference libraries to the same 12 even indices before reading any query. The odd query indices in this manifest coincide with the changed-light cases. Their exact light angles are absent from the restricted library. Query lighting truth is used only for scoring and stratification.

On odd query indices, the change from 24 to 12 lights cost **21.88 percentage points** of raw pose recovery. The 95% paired terrain-group interval is **−43.75 to −4.69 points** for the restricted library minus the full library. The result suggests the full-library advantage depends in part on having the exact lighting template, while a substantial advantage over the two completed image-only controls remains on this test.

All 384 restricted-library rows independently replayed. The run took 426.99 seconds of a 1,256-second reserved ceiling. Online times exclude reference render preparation. Both libraries and the query images still share the same physical simulator and terrain sources. The experiment varies reference lighting availability within this simulator. Transfer, calibrated acceptance and the final reserve remain open.

- [Frozen restricted-library protocol](solar-offcatalogue-direct-dispatch-v95.json)
- [Independent replay and paired terrain intervals](solar_offcatalogue_direct_replay_v95.json)
- [Full-library matched development benchmark](SOLAR_FULL_PAIRED_DEVELOPMENT_V94.md)
