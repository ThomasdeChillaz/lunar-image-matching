# Research evidence and state-of-the-art comparison

Snapshot: 4 November 2026. This is a requirements audit, not a claim of superiority. The subject fits [Ansar’s localization and feature-matching research](https://www-robotics.jpl.nasa.gov/who-we-are/people/adnan_ansar/), but a convincing new result is still missing.

The completed learned control recovers **15/96 raw development poses (15.625%)**, with 16 wrong raw poses and six raw proposals on 96 nonoverlap pairs. Acceptance was not calibrated; these are not accepted-pose or accepted-error rates. The new 95,808-parameter response network has **zero lunar training updates**. The published 537 passing software tests establish implementation checks, not a scientific accuracy gain. The original acquisition failed after **10/96 units**; its 3 November 22:42:36 UTC deadline expired. A separately declared completion remains pending. Historical results and the untouched reserve must stay preserved. Local report hashes and exact requirements appear in the companion JSON.

Published numbers answer different questions; the rows below are **not a ranking against our 15/96**.

| Primary source and version | Actual published measurement | Comparison limit |
|---|---|---|
| [RoMa v2, arXiv v3, 6 July 2026](https://arxiv.org/html/2511.15706v3), Table 4 | MegaDepth-1500 relative-pose AUC at 5°/10°/20°: **62.8/77.0/86.6**. | Angular AUC on terrestrial pairs; not joint accepted absolute lunar pose. Its GPU throughput cannot establish our CPU advantage. |
| [RoMa-Ω, arXiv v1, 8 September 2026](https://arxiv.org/html/2609.09507v1), Table 2 | WxBS mAA@10 pixels: RoMa v2 **64.8**, Ω **72.9**. | Compare these within this paper’s setup. RoMa v2’s original Table 11 reports **55.4** under its own setup; do not splice that into this comparison. |
| [MOONSt3R, arXiv v1, 20 October 2025](https://arxiv.org/html/2510.18172v1), Table 3 | Dynamic trajectory: **98.0%** relative rotation below 2°; **91.7%** translation-direction error below 2°. | Separate angular marginals, not joint success or absolute translation within 1% of range. Its scale-invariant reconstruction differs from our reference-map pose task. |
| [Georgakis and Ansar, SIW 2024](https://seal.ae.gatech.edu/sites/default/files/2024-09/GeorgakisGeorgios.pdf), page 3, Figures 4–5 | LoFTR-fine, 180° azimuth/3° elevation change: seen-map reprojection accuracy **41.8% at 1 pixel /79.9% at 5 pixels**, 479 mean matches; unseen-map **12.0%/27.0%**, 193.3 matches. | Correspondence accuracy, not pose recovery. These are **2024** results, not numerical results from the 2025 AIAA paper. |

[LoMa-R’s official implementation](https://github.com/davnords/LoMa) supplies a rotation-invariant descriptor and matcher relevant to aerial imagery; test that variant rather than attributing LoMa-B scores to it. [Lunar model distillation](https://clementinegrethen.github.io/publications/ECCV.html) already studies efficient lunar reconstruction: students have **94.9–154.9 million parameters**, versus a 688.6-million teacher. Public paper/checkpoint availability was not verified from the [repository](https://github.com/clementinegrethen/mast3r-distillation). Our smaller parameter count proves neither equivalent functionality nor better localization.

The full goal remains:

| Requirement | Evidence still needed |
|---|---|
| **≥70% correctly accepted poses**, denominator all eligible positives; joint rotation ≤2° and absolute translation ≤1% median reference range. | Qualified complete data, trained candidate and frozen-test results retaining rejected/failed cases. |
| **≥10 percentage points** over the strongest matched compact control, including deterministic refinement; positive paired terrain uncertainty interval. | Same-case ablations, terrain-group intervals and all selected seeds 20261101/02/03. |
| **≤1% empirical accepted-pose error and ≤1% nonoverlap acceptance**. | Separate calibration, common recipe, full error–coverage curves and uncertainty; both negative terrain endpoints grouped. No deployment guarantee. |
| Modern accuracy advantage, or **within 3 points while ≥5× faster on the same CPU** at comparable error. | Common native-512 content, reference geometry and AP3P/RANSAC pipeline; SIFT, render-and-match, MatchAnything–EfficientLoFTR, LoMa-R, RoMa v2 and MOONSt3R controls, actual latency/memory and disclosed timeouts. |
| All three techniques tested. | All eight on/off combinations; pooled-response, solar-off/independent-Sun controls; active selection versus three controls at 32/64/128 searches; estimated-Sun and fairly supplied-Sun tracks. |
| Reliable geometry, training and transfer. | ≥95% exact-pose gate; 16 training/eight selection groups; even lights only for training/teachers; 100-update replay then 2,000-update studies; ±5/10/20 m terrain tests; fresh laboratory/orbital diagnostics and LuSNAR 7–9 transfer. Variants are not independent terrain. |
| Resource limits and untouched evaluation. | <100,000 total parameters, native 512 pixels, two threads, free compute, cumulative 48-hour training/72-hour comparisons; four calibration/twelve final groups preserved; published freeze before one final run. |
| Research package. | Independent metric replay, release/checkpoints/reproduction, two-page brief, 6–8-page paper draft, four figures, success/rejection/failure demo and unsent outreach. |

Novelty is still a hypothesis: [Ansar’s 2025 work](https://doi.org/10.2514/6.2025-2073) already combines LIMA and learned features; [2026 follow-up](https://noahzegna.github.io/) synthesizes illumination-matched maps. Full numerical tables were inaccessible. [DLR 2015](https://elib.dlr.de/97110/) uses known-Sun shadow matching; [CMU 2020](https://publications.ri.cmu.edu/terrain-relative-navigation-for-lunar-polar-roving-exploiting-geometry-shadows-and-planning) combines geometry, shadows and information-aware planning. The defensible candidate contribution is therefore the **specific integrated mechanism**, supported by a measured localization gain. A supplied-Sun operational track matters because ephemeris information can be available. No superiority or outreach-ready performance claim is justified yet.
