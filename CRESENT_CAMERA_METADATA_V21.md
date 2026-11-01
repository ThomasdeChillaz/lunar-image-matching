# A candidate fresh source: metadata controls only

The [CRESENT-365 v3 release](https://zenodo.org/records/23021341) includes15,283
camera-pose records and projected crater annotations. The pose text's published
MD5 was verified. Image and annotation ZIP directories were inspected by byte
ranges; their complete multi-gigabyte archive checksums were not locally
verified. Only the first PNG's33byte header was read, including its1024x1024
dimensions and IHDR checksum. **Zero raster pixels were decoded and zero
matching inferences were run.** This has not created a fresh localization test.

Annotation algebra indicates focal length886.8100134752659pixels, principal
point(512,512), and a60degree field of view. Pose text is Moon-centred camera
position in metres followed by a wxyz world-to-camera quaternion. A camera-to-
world transform needs the transposed rotation. These are metadata-derived
conventions, not measured optical calibration accuracy. The world annotation
header omits the rotation field present in its actual seven-column rows.

The original predeclared broad check on seven frames **failed**: tiny or very
flattened projected ellipses have inconsistent shapes. Its source, protocol,
report and failures are preserved. The discrepancy is concentrated in
unresolved objects; a unique numerical/renderer cause has not been proven.
Projected ellipse centres also cannot be treated as projections of the3D
crater centres under oblique viewing: the full conic must be projected.

A narrower geometric convention check was declared before retrieving five
different annotation frames:2,33,333,3333,9999. It keeps finite/in-frame ellipses
with both semiaxes at least2pixels and a surface disc facing the camera. All
6,940eligible ellipses pass the declared conic checks. This does **not** repair
the original broad gate. Surface-facing does not establish terrain visibility.
See `reports/cresent_camera_metadata_replay_v21.json` for independent replay.

The optional local fixture directory is `data/cresent-metadata-v21`. Its
manifest pins every retrieved text/header/protocol and three interpretation
source files. To replay after installing that separate metadata fixture:

```sh
python verify_cresent_metadata_v21.py \
  --root data/cresent-metadata-v21 --output /tmp/cresent-metadata-replay.json
```

Source metadata is CC BY-NC4.0; our interpretation code is MIT. No images or
external model weights belong in this metadata artifact. Tests without the
optional fixture still run the analytic geometry controls and skip its one
source-data integration check explicitly.

Before any localization study, reference DEM/depth, coordinate-frame
uncertainty, surface visibility, geographical footprint groups, overlap with
all historical exposures, source/model/case freezes and actual matching
software controls remain required. Exclude metadata-control image IDs
0,1,2,7,17,33,101,333,1000,3333,7000,9999,15282from any proposed untouched
image-matching reserve. Do not infer solar angles/timestamps from frame order.
This orbital-view simulation is a distinct additional task; it does not satisfy
the original requirement for twelve fresh LunarGeo footprint groups.
