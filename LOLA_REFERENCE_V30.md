# Reference terrain for an additional simulation test: still under investigation

The compact model has not improved consistently. A defensible next study also
needs fresh terrain and independently checked reference geometry. CRESENT-365
is a candidate **additional** simulation source; it does not replace or repair
the original missing twelve fresh LunarGeo footprint groups.

The CRESENT paper identifies the [MIT LOLA archive](https://imbrium.mit.edu/) as
its terrain source and describes terrain heights used to adjust crater rims.
See the [authors' paper, sections3.2–3.3](https://arxiv.org/html/2509.20748v1).
Its exact DEM product/version and rim-sampling implementation have not yet been
recovered. The current investigation therefore uses official data as a candidate
reference and records discrepancies, rather than assuming renderer agreement.

## What was actually acquired

One official northern development tile:
[LDEM_512_45N_90N_000_090.IMG](https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_512_45N_90N_000_090.IMG)
and its [detached PDS label](https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_512_45N_90N_000_090.LBL).
The complete file is2,123,366,400bytes, downloaded in66.60calendarseconds.
Local full-file SHA256:
`25b3774029238256dd41c204dff300892f957baddf102aec5f71d9a64b187152`.
HTTP size/version matched the earlier fixed byte-range probe. A publisher
checksum was not available/verified; this hash identifies our complete local
snapshot. The2.12GB file is ignored under `data/lola-reference-v30/` and is not
uploaded to the code repository. See
[acquisition evidence](reports/lola_reference_acquisition_v30.json).

PDS labels specify little-endian signed16-bit samples,0.5metre scaling,
1737400metre reference radius, pixel registration, east-positive longitude,
and the mean-Earth/polar-axis DE421 frame. **These map values are planetary
radii, not camera ray ranges or camera-axis depth.** The loader validates these
conventions, dimensions and projection offsets and rejects unsupported fields.
It checks exact HTTP206ranges before reading bodies, caps range sizes and
preserves label bytes including their original newline conventions.

Three official labels and eight actual height samples are retained in the small
`pds_height_fixtures_v30.json` fixture. An offline auditor independently checks
signed byte arithmetic and geographic byte offsets. Eleven focused controls
pass in the validated environment. These tests import no neural models and
make no network calls. The complete neural repository tests/fresh procedural
smoke are queued behind the existing training/audit/control pipeline; they have
not been rerun concurrently with fitting.

```sh
python -m pytest -q test_lunar_pds_height_v30.py
python verify_lunar_pds_v30.py --output /tmp/lola-range-replay.json
```

## A real discrepancy remains

Before sampling crater rims, the development probe fixed the first40 eligible
craters in already exposed **metadata-control frame0**, with finite500–30000m
axes and all360rim sampling positions within this tile. No optical camera image
was opened. The nominal released-axis interpretation gives:

| Development-only terrain check | Median absolute radius difference | Maximum absolute difference |
|---|---:|---:|
| Released axes used as rim semiaxes,360samples |311.68m|1125.09m|
| Exploratory alternative: halve both axes, same40craters |84.79m|1259.45m|

[Original protocol](reports/cresent_lola_rim_probe_v30_protocol.json),
[all original outcomes](reports/cresent_lola_rim_probe_v30.json),
[exploratory alternative protocol](reports/cresent_axis_alternative_v30_protocol.json),
[all alternative outcomes](reports/cresent_axis_alternative_v30_results.json).

The halved-axis hypothesis was declared **after seeing the nominal discrepancy**.
It improves the median while worsening the maximum; it neither confirms axis
semantics nor validates the map/renderer. No offset, transform, pose, matcher or
acceptance threshold is fitted, and neither interpretation is adopted as verified
reference depth. The original released conic/pose annotations remain unchanged.

Differences may involve the sampling convention, terrain version/interpolation,
axis meaning or frame/renderer registration. This probe does not identify a
unique cause or establish an error bound. Forty craters in one exposed metadata
frame are not forty independent terrain tests. Exact conic algebra alone cannot
prove optical crater-rim alignment or dense terrain visibility.

With the optional full map and the earlier separate CRESENT metadata capsule:

```sh
python probe_cresent_lola_rims_v30.py \
  --map-root data/lola-reference-v30 --output /tmp/lola-rim-replay.json
```

Source optical pixels decoded: **zero**. Matcher inferences: **zero**. Dense camera
depth created: **none**. Renderer agreement, footprint-disjoint fresh splits,
visibility and orbital accuracy remain unverified. LuSNAR7–9remain untouched.
This is measurable progress toward checking additional data, not a passed final
test or a model improvement. NASA/LOLA data credit: LRO LOLA team; CRESENT
metadata: McLeod, Chng, Rodda and Chin,
DOI10.1007/s42064-025-0294-0; CRESENT source metadata CC BY-NC4.0. Own code MIT.
