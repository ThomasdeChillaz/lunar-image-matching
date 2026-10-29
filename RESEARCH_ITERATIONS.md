# Iterative research: hypotheses, development and fresh evaluation

## Objective and claim boundary

Add LoFTR to the exact rendered benchmark; expand training terrain; test fresh geographic areas; separate translation, rotation, scale and combined changes; select rejection policies on development data and freeze them before fresh evaluation. Pursue substantial improvements through measured hypotheses and combinations. A state-of-the-art claim requires stronger current baselines under the same inputs, metrics, geometry and compute disclosure. Better development numbers, a weak baseline or a different paper's metric cannot establish it.

Human annotation remains outside scope. Known render geometry supplies accuracy labels. Real orbital observations supply unverified diagnostics only. Original pilot inputs, results and method sources remain frozen.

## Data contract, frozen before new model outcomes

`research_manifest_v1.json` selects 384 train and 64 development patches from the existing checksummed physical-render archive, plus two untouched reserves of 32 patches each. All nine Sun positions are retained. Selection uses metadata, fixed longitude bands, deterministic ordering and acquisition round-robin; it never uses matcher outcomes or image appearance. Entire acquisitions stay in one split. Cross-split footprints and a .03-degree buffer are checked. Reserve acquisitions and buffered footprints also exclude every previously evaluated physical and real Tycho sample.

These are four separated areas **within Tycho**, not different craters or south-pole environments. Overlap/acquisition components are only 2 train, 2 development, 4 reserve A and 8 reserve B; patch counts must not be treated as independent sample sizes. Reserve A and B must stay unread by fitting, calibration and hypothesis selection. Once final outcomes from a reserve are inspected, that reserve cannot be described as fresh for a later iteration.

`research_geometry.py` defines separate translation, rotation, scale and combined families, with seeded continuous rotation in [-30,30] degrees, scale in [.75,1.3], and shifts in [-12,12] pixels where applicable. Different Sun positions are actual physical renders. The artificial image warp supplies geometric supervision only. Matchers see pixels/masks, never the ground-truth motion. Known non-overlap negatives remain separate.

## Literature-to-hypothesis ledger

| Hypothesis | Source idea | Implemented experiment | Decision/evidence |
|---|---|---|---|
| H0: detector-free matching is a stronger baseline | [LoFTR, CVPR 2021](https://github.com/zju3dv/LoFTR) | Pinned Kornia 0.7.4 outdoor weights, native 128px, common geometry, every original Sun pair | Completed: 271/576 correct. This is an off-the-shelf baseline, not JPL's lunar-trained LoFTR. |
| H1: local self-similarity survives brightness polarity changes | [MIND, Medical Image Analysis 2012](https://doi.org/10.1016/j.media.2012.05.008) | Original 2D self-difference channels plus normalized correlation | Development translation 262/288 versus NCC 179/288; 0/72 non-overlap fits. Promising; fresh accuracy not yet tested. Not a full MIND reproduction. |
| H2: gradient orientation without polarity helps | [RIFT, TIP 2020](https://ljy-rs.github.io/web/Files/TIP2020.pdf) motivates structural representations | Original squared directional-gradient channels | Development translation 227/288; worse than H1. This is not RIFT or phase-congruency reproduction. |
| H3: geometry search extends H1 beyond translation | Multi-scale coarse-to-fine image registration | Planned image-only rotation/scale search and local refinement | Implemented and known-transform tested; development comparison is running. Must compare each geometry family separately. |
| H4: expanded all-light training plus bounded adaptation transfers better | [JPL lunar LoFTR study](https://seal.ae.gatech.edu/sites/default/files/2024-09/GeorgakisGeorgios.pdf), [MoonAnything](https://arxiv.org/html/2604.00682v1) | 384 training patches; all eligible Sun pairs; near-identity projection, singular values bounded [.75,1.25]; development-only checkpoint selection | Training completed on 11,727 lighting pairs; checkpoint 800 selected using 256 development descriptor pairs. Registration development evaluation is running. Descriptor retrieval is a surrogate, not a registration result. |
| H5: global pretrained context plus local geometric precision improves difficult matches | [RoMa, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.pdf) | Candidate future dense baseline/feature combination | Not implemented; no outcome claimed. |
| H6: diagnostics can reject wrong alignments without rejecting allowed motion | The old transported gate failed because its translation cutoff excluded the experiment's true shift | `research_gate.py`: development-only coverage, forward/reverse cycle, support, structural score and peak margin | Empirical policies frozen on the first development screen. No fresh-test performance or safety guarantee. |

## Decision protocol

1. Keep the full record of each development hypothesis, including poor results and applicability exclusions. Do not pool translation-only baselines into a rotation/scale superiority claim.
2. Compare candidate models and combinations on development terrain, with the same known geometry, masks, failures and negatives. Descriptor retrieval alone cannot select the final research conclusion.
3. Before any reserve run, save an immutable artifact binding dataset, source files, model bytes, selection rule, rejection policy, complete case plan and comparison methods. Choose the meaningful improvement criterion before seeing reserve outcomes.
4. Score every planned pair. Report raw success, coverage, accepted incorrect alignments, discarded correct alignments, non-overlap false acceptance, runtime and uncertainty grouped by overlapping footprints/shared acquisition.
5. Preserve reserve failures. Subsequent modifications return to development and require another untouched geographic evaluation. Expanding to different crater/polar terrain remains necessary for broader lunar-generalization claims.

Current evidence and unfinished requirements are in [iteration report](reports/RESEARCH_ITERATION_1.md) and [machine-readable record](reports/research_iteration_v1.json). No state-of-the-art improvement or real navigation accuracy has been established.
