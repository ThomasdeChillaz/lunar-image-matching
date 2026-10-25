# Research outreach notes

Use this factual summary after reviewing the report and code yourself:

I built a reproducible pilot benchmark comparing NCC, SIFT/RANSAC and pretrained DISK/LightGlue on published LROC repeat observations from three lunar regions, with a held-out south-polar scene. The pipeline preserves source provenance, masks nodata, separates exact-transform controls from repeat-observation diagnostics, and records failures as well as estimates. It also includes an independent-checkpoint annotation tool because the source map coordinates do not establish subpixel ground truth. The repository contains the measured results, failure examples and a short report.

Possible research question to discuss: which reference-generation and uncertainty protocol would make this small illumination-robust matching benchmark scientifically useful for a terrain-relative navigation group?

Do not describe map agreement as navigation accuracy, claim reproduction of LIMA, imply JPL endorsement, or claim a performance improvement that is not supported by the measured report. No message has been sent.
