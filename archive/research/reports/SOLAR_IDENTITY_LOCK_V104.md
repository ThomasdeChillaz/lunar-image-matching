# Exposure gain can lock the pose to the reference camera

The fixed p99.5 exposure pilot recovered 17 of 24 raw correct poses, compared with 15 of 24 for the unchanged Direct control. It also returned four wrong positive poses instead of one. The six near-black positive misses with zero baseline matches remained pose failures. Neither arm returned a pose on the 24 paired nonoverlap cases.

A read-only replay independently matched the protocol, terminal receipts, all 96 cell seals and the saved metric replay. It reconstructed each positive's co-visible projection from the pinned geometry and recomputed reference and fitted camera reprojection errors on the saved correspondence arrays. The replay opens no optical images and performs no matching or pose fitting.

| Exposure case | Saved matches within 1 px of zero motion | Fitted residual p95 | Median true error | Fitted camera versus reference | Fitted camera versus query truth |
|---|---:|---:|---:|---:|---:|
| u025, query light 0 | 1,890/1,890 | 0.062 px | 78.43 px | 0.0037 m, 0.0004° | 191.80 m, 19.72° |
| u027, query light 1 | 1,862/1,862 | 0.066 px | 161.26 px | 0.0187 m, 0.0005° | 203.45 m, 52.81° |
| u061, query light 17 | 6/6 | 0.087 px | 36.15 px | 1.210 m, 0.037° | 339.73 m, 16.53° |

The first two queries had a p99.5 white code of 2. Their selected template was light 1 in both cases. All 1,890 and 1,862 correspondences passed the common solver's 3 px inlier rule, yet each case had zero true correspondences within 2 px. The lower-gain third case, on another terrain, also had zero true correspondences within 2 px. Low fitted reprojection residual and large consensus can therefore coexist with a pose hundreds of metres from the query truth.

Across the 96 saved cells, a retrospective descriptive criterion of at least six matches and at least 90% within one native pixel of zero displacement identified zero control outputs and three exposure outputs. All three were wrong positives. The criterion has no prospective validation and a valid near-reference query can also have near-zero displacement. The result localizes the failure to image-coordinate agreement but cannot identify the origin of that agreement or validate a rejection rule.

- Source SHA-256 `ad81fba100c987236aaaab3d3525744152f0079e67f2a71fa3072804f9b108f8`
- Result SHA-256 `fc6c45c64c859022a08c0ddff621936dcade0940085d4a9b6520196f249d0884`
