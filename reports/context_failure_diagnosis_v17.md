# Why the context study did not improve lunar poses

All three primary configurations retained initialization at54/72. Their final trained checkpoints give47/72 pointwise,43/72 self-context and39/72 cross-context. These final checkpoints are failed experiments, not selected upgrades.

| Pointwise checkpoint | Training NLL | Selection NLL | Training reachable proposals | Selection reachable proposals |
|---:|---:|---:|---:|---:|
| 0 | 7.061 | 7.123 | 16,685 | 12,694 |
| 500 | 6.039 | 5.908 | 13,193 | 10,505 |
| 2000 | 5.935 | 5.783 | 13,717 | 10,506 |

Reachable means the independently projected visible point lies inside the actual rounded33px native query search. It does not guarantee successful refinement or pose. Totals pool four correlated orientation branches. Every proposal was checked against the actual matcher, and independent geometry replay covers all2,160 records. Likelihood aggregates replay; no independent neural reimplementation is claimed.

The likelihood improves while geometrically useful proposals decrease on both the training fixtures and selection cases. This is consistent with a mismatch between the current objective and useful matching. It does not identify a unique causal mechanism. A unit-controlled counterexample shows that bilinear cell labels can penalize an alternative coarse candidate from which native refinement could reach the true point.

The truth-dependent filtering probe examines all60 native-refined branches on the15 missing poses and the three wrong returned poses. Across the tested3/4/8/12px oracle filters, at most two additional cases yield any correct pose. Perfectly preserving the54 known-correct outputs and choosing oracle branches would produce56/72, but that requires query truth and is not a matcher result or universal upper bound. Independent replay covers all252 oracle fits. Filtering these saved point sets alone does not demonstrate the required gain.

The next hypothesis tests search-window positive sets, anchors for independently geometry-labelled genuine frozen proposals, and coarse-image feature learning while the native fine encoder stays frozen. The prepared model has81,696–82,208 total parameters. Its numerical/gradient controls pass; no new fitting or localization gain exists yet. See SEARCH_WINDOW_V18.md for the pending actual-forward and fitting gates.

Fresh final data, low-error calibrated acceptance/nonoverlap, remaining native pretrained comparisons, transfer, timing and the paper remain unfinished. No paid compute, human review or email.
