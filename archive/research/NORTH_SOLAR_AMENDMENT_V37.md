# Correct the primary lunar lighting before training

The expanded terrain acquisition completed: all 24 original development
supports, 600 verified encoded tiles (338,328,349 bytes) and 24 curved meshes.
The first render was stopped after a geometry check found that **96 of 144
site/light settings imply a solar body latitude outside a conservative lunar
solar envelope**. No matcher or fitting result was used for this decision.
Five completed stress components and all partial output remain preserved.

## Why the original angles are a stress test

[NASA's lunar solar geometry presentation](https://ntrs.nasa.gov/api/citations/20230000854/downloads/Minow_SciTech_2023%20-%20presentation%20-%20final.pdf)
places the lunar rotational axis about 1.5 degrees from the ecliptic normal.
Solar elevation and azimuth therefore cannot be chosen independently at a
polar site. Converting the old local directions to Moon-body coordinates
implies solar latitudes of 3.11–5.99 degrees for light 0 and 6.11–8.99 degrees
for light 3 across these development sites. They remain useful arbitrary
relighting stress settings, but cannot support a realistic lunar-Sun claim.
The screen uses a generous 1.6-degree bound, not fitted model outcomes.
[Full diagnostic](reports/north_solar_geometry_diagnostic_v37.json).

The earlier pilot still establishes numerical camera/depth/mesh and oracle
controls. Its lighting is now explicitly classified as photometric stress;
those controls did not establish orbital solar feasibility.

## Frozen correction

Use subsolar latitude +1.5 degrees and six fixed local hour angles:
-80, -50, -20, +20, +50 and +80 degrees. Derive the solar vector in Moon-body
coordinates, then transform it to each site's east/north/up frame. This couples
local elevation and azimuth and keeps these configurations above the mathematical
horizon. The necessary solar-latitude envelope passes at every prospective
site. This is an idealized geometry family, **not dated SPICE ephemerides** or
an independently characterized terrain-horizon model.

All 24 terrain supports, cameras, native pixel conventions, materials, image
sizes, primary lighting-index pairs, labels and pose criteria stay fixed.
The correction precedes all model inference and fitting on the new data.
Actual pinned Blender controls verify all 144 lamp orientations against their
body-coordinate vectors; maximum direction error is 3.79e-7. Three independent
solar tests, the six-light/split tests, the full 295-test suite (four optional
skips) and a fresh 100-row procedural smoke pass. These are data/software
checks, not improved model performance.

The original render allowance remains six hours. The interrupted render used
538.14 calendar seconds; the amendment retains its unused allowance and also
keeps the **original absolute deadline**. It does not reset a clock or claim
the old render completed. [Stop record](reports/north_v35_render_stop_v37.json),
[pre-inference solar contract](reports/north_solar_amendment_v37.json).

## Execution and fitting gates

Publish and byte-verify this source before restarting with a separate output
folder. Reuse the existing checked terrain meshes; no new regions or reserve
pixels are acquired. Run the corrected 576-image render, then the complete
72-geometry/243,360-label audit with the original tolerances and >=95% oracle
criterion. Calibration and final regions remain unused.

```sh
/path/to/pinned/Blender --background --factory-startup --threads 2 \
 --python-exit-code 1 --python render_north_development_v37.py --
python verify_north_development_v37.py
```

Prepared v38 fitting programs are published as **gated, not yet executed**.
After the full-data audit, prepare the versioned six-light bundle. The first
fitting action is a train-only 100-update cost pilot using the same 81,696-
parameter architecture, original v18 objective and optimizer. Its fine encoder
stays frozen. Independently retrain all 100 updates and compare model,
optimizer, losses, gradients and random states exactly before a full cohort.
A fixed 16-case baseline cost bank follows, using all eight selection components
and both positive/nonoverlap cases. Incomplete comparisons stay explicit.

```sh
python north_training_bundle_v38.py
python train_north_pilot_v38.py --publication-commit PUBLIC_COMMIT
python verify_north_pilot_v38.py
python benchmark_north_control_v38.py --publication-commit PUBLIC_COMMIT
```

The pilot reads only 16 training components. The baseline bank reads only eight
selection components. All six lights are sampled; original three-light and
72-case assumptions are not reused. Query geometry is training/scoring only;
inference accepts reference geometry and query intrinsics. Global body supports
control negative construction rather than identical local boxes.

Uniform diffuse material, weak constant ambient, a point Sun and finite terrain
halos remain simplified photometric assumptions. Original fresh released-
LunarGeo availability, external pretraining overlap, full acceptance/nonoverlap,
remaining author baselines, transfer, uncertainty, paper and compact-v3 targets
remain unresolved. The completed old 6,000-update study remains negative. No
new localization gain, SOTA or verified orbital accuracy is claimed; no email.
