# Additional northern terrain study: prospective source and control freeze

Repeated training has not delivered a consistent learned gain. This next data
investigation aims to supply separate terrain and exact3D simulation labels,
without relying on the unresolved CRESENT map/annotation registration. It is a
**new additional simulation study**; the original missing twelve fresh released
LunarGeo groups remain a recorded shortfall. Neither software preparation nor
geographical spacing constitutes an achieved model target.

## Source and split before any terrain decoding

Use the official [north-polar LOLA terrain products](https://pgda.gsfc.nasa.gov/products/98),
with10m grid spacing and the MOON_ME DE421 frame. The verified source header
is43280×43280float32, north polar stereographic, in metres above a1737400m
sphere. No northern source height pixels have been decoded in this phase.
Nominal grid spacing is not a claim of10m independently measured terrain detail;
the source includes interpolation and offers separate effective-resolution and
height-error products.

[north_render_prospective_plan_v31.json](reports/north_render_prospective_plan_v31.json)
contains every candidate position and full source-window extent, assigned
before patch extraction:

| Purpose | Separated candidate components |
|---|---:|
| Training |16|
| Model selection |8|
| Confidence calibration |4|
| Final test |12|
| Separate renderer/cost pilots |4|

All20.48km terrain/occluder supports are nonoverlapping, with deterministic
assignments across four projected strips. These are map supports, not known
independent missions or renderer batches. The historical optical/training
footprint audit remains required before certifying any candidate as fresh.
Missing terrain must be reported; no inpainting or performance-driven replacement
is allowed.

The plan specifies a10m core mesh with40m outer sampling, four cameras and six
actual rendered Sun directions at2°/5° elevations. It retains spherical lunar
curvature and a per-component local east/north/up frame with its transform to
the body frame. Constant diffuse reflectance, weak background illumination and
finite terrain boundaries are explicit simulation assumptions. They do not
reproduce every lunar optical or distant-shadow effect.

For the principal comparison, each evaluated component has the same12frozen
lighting/view pairs. The candidate final set has144positive pairs and144
nonoverlap pairs, with shared component dependence reported. Training can use
all108lighting/view pairs per training component. Negative partners must be
fixed within each split, and both endpoints tracked. Raw source/case hashes,
preprocessing and model/threshold freezes remain required before final inference.
Preparing this manifest does not pass the remaining calibration/final-inference
gates.

## Verify the renderer before using its labels

The official Intel Blender4.5.9package matches its current vendor SHA256 and
passes macOS deep/strict signature verification. Build hash8bf95cbd38d1 and the
relevant API fields are checked. This is an isolated local runtime; it does not
modify the validated matching environment or require paid compute. See
[runtime/header metadata proof](reports/north_renderer_runtime_v31.json).

The [independent projection check](reports/north_projection_control_v31.json)
compares the analytic north-polar conversion to GDAL on five fixed points:
maximum difference2.85×10^-14degrees. Six focused no-render geometry/triangle
controls pass. Actual rendering is still pending; these are software checks.

`blender_camera_control_v31.py` renders known planes and a solid box at128and512
native pixels. It records raw floating-point position/depth, every control's
errors, explicit native pixel centres and geometric silhouettes. Camera-axis
Z and ray range are tested separately. Dark regions do not become occlusion
labels. Failed controls preserve all files; their tolerances cannot be silently
relaxed after measurement.

`verify_blender_camera_control_v31.py` independently computes triangle/ray
intersections and recovers two analytic fixture poses with the unchanged common
AP3P estimator. This is not the95%oracle gate on actual development terrain;
that gate remains required once the terrain/cameras exist.

```sh
python -m pytest -q test_north_render_plan_v31.py

/path/to/Blender --background --factory-startup --threads 2 \
  --python blender_camera_control_v31.py -- \
  --output runs/north-v31-camera-control

python verify_blender_camera_control_v31.py \
  --run runs/north-v31-camera-control \
  --output reports/north_camera_control_audit_v31.json
```

Publish and byte-verify all these sources before the actual forward. The renderer
must wait for the existing training, its independent cohort/selected-model checks
and the queued full repository regression to finish. It is another serialized
two-thread CPU job, not concurrent paid or extra compute. Only after actual
camera/depth controls and oracle replay pass may the four declared pilot terrain
components be acquired/rendered. Their measured cost must fit the prospective
six-hour generation cap before full generation is allocated. No terrain pilot
or full dataset has been generated yet.

## Remaining scientific work

A new model must improve actual camera localization against the strongest
simple compact pipeline and completed modern profiles, under the original
acceptance/error and grouped-uncertainty targets. The northern simulation cannot
establish verified orbital accuracy or replace LuSNAR/laboratory transfer. Final
cases remain unused; no learned or SOTA result is claimed. All prior failed
studies and the CRESENT discrepancy stay preserved.

Source credit: Barker et al.(2025), *Large-Scale Roughness Properties of the Lunar
North and South Polar Regions as Measured by LOLA*, DOI10.3847/PSJ/adbc9d;
data DOI10.60903/gsfcpgda-lola-poles. Blender is separately licensed upstream;
its binary is not redistributed in this repository. Own new code is MIT.
