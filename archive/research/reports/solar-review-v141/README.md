# Current research brief, working paper and comparison boundaries

This documentation update incorporates a dated review of nine primary sources,
including recent lunar foundation-model distillation. It preserves every measured
result, frozen experiment source and original performance requirement. No new
matching, training, rendering, calibration, final-test access or outreach occurs.

- [Two-page research brief](lunar-localization-v141-research-brief.pdf)
- [Eight-page working paper](lunar-localization-v141-working-paper.pdf)
- [Primary-source notes and metric definitions](literature.json)
- [Saved evidence and input hashes](evidence.json)
- [Document and preservation checks](verification.json)
- [All original research targets](../research_requirements_status_v141.json)

The paper separates the supplied-light acquisition pilot, the historical
default-dither modern comparison and the unrun clean unknown-light experiment.
It treats active selection's one-pose advantage over the strongest equal-search
controls as inconclusive. Published match-precision, relative-pose AUC and dense
reconstruction scores are not converted into our absolute accepted-pose rate.

The candidate claim concerns a specific coupling of shared illumination, camera
inference and map-uncertainty-aware measurement selection. Compact lunar models,
active search, learned refinement and lunar adaptation each have prior art. The
targeted review is not an exhaustive novelty search or proof of superiority.

`build_package.py` generates the documents from unchanged saved statistics and
figures. Run it at the repository root with the existing validated environment
and ReportLab. Final PDFs are written to `output/pdf`; identical distribution
copies appear here. The old v137 PDFs remain byte-identical. All ten rendered
pages were visually checked, and document extraction verifies page counts,
links, expected metrics and absence of missing-glyph boxes.

The prepared [v140 experiment](../solar-blind-v140/NEXT_EXPERIMENT.md) remains the
next scientific step and still needs the outstanding resource decision. This
update neither creates an allowance nor replaces that full eight-case roster.
The broader goal remains active; the model has not met the original targets.
