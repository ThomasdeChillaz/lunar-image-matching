# Shared-scale fine refinement: no pose gain on the fixed bank

All three profiles recover **12/16 correct raw poses**, including exactly the
same successful cases. Native-only gives five strict poses; native+half and
native+half+quarter give four each. Every profile returns one wrong estimated
pose. The historical original-border control gives12/16 with six strict poses.
No scale upgrade is adopted and the conditional full72-case run is not taken.

| Shared frozen descriptor scales | Correct raw /16 | Strict /16 | Wrong raw | 2px correspondence precision |
|---|---:|---:|---:|---:|
| Native | 12 | 5 | 1 | 3,025/3,880 (77.96%) |
| Native + half | 12 | 4 | 1 | 3,150/3,879 (81.21%) |
| Native + half + quarter | 12 | 4 | 1 | 3,161/3,879 (81.49%) |

The point-precision improvement does not improve camera localization. This
rules out adopting this particular deterministic scale fusion as a pose
upgrade on this bank; it does not prove that every multiscale learned method
would fail. The wider common support also costs one strict pose relative to
the historical control, so support changes cannot be ignored.

All profiles start from identical image-estimated poses and original reference
points, use the same33nativepixel search, conservative descriptor/warp support,
two surface updates and3nativepixel AP3P estimator. Query truth enters scoring
only. Actual selected-checkpoint rotation/pose and blank controls pass. Source
was public and byte-verified at `fac436adf00c9c46d7f24f40b793154830574f9f`
before inference. No weights or rejection thresholds were fitted here.

Independent replay passes all48 saved native geometry/pose rows, all profile
and case denominators, source/input fingerprints and aggregates. See
[native replay](native_fine_pyramid_bank_audit_v22.json),
[summary](native_fine_pyramid_bank_summary_v22.json) and
[protocol](native_fine_pyramid_bank_protocol_v22.json).

The16 cases are an old fixed bank from four exposed terrain components, not
new independent terrain. Correspondence points are correlated and the pooled
percentages are descriptive. Raw poses are not calibrated accepted navigation.
Cached diagnostic execution is not an isolated latency study. No fresh final,
SOTA, reliable1%risk or orbital-accuracy claim follows.
