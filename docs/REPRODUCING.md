# Reproduce the software and measured evidence

Use a complete checkout and an editable Python 3.12 installation. Install the core
dependencies and package as shown in the root README. Matching controls additionally
need the validated CPU PyTorch runtime. The complete historical environment remains
pinned in `requirements-lock.txt`; optional pretrained models have their own archived
dependency files and weight requirements.

## Fast checks without lunar inputs

```sh
lunar-match status
lunar-match audit
python -m pytest -q
lunar-match smoke --output runs/generated-control-new.json
```

The generated control deliberately supplies two camera/lighting proposals. It checks
whether a lower-ranked branch can win after real native correlations and AP3P. It
uses no lunar imagery and is not a physical sunlight simulation.

An additional generated check covers the full public command path: RGB16 files,
manifest validation, the image-only camera initializer, native matching and output
serialization. It supplies no true query camera to inference:

```sh
python tools/check_native_cli.py --output runs/generated-native-cli-new
```

This also uses generated patterns; its successful pose is software evidence.

The complete historical tests run in their own original relative-path namespace:

```sh
cd archive/research
python -m pytest -q
```

`make test-archive PYTHON=/absolute/path/to/python` does the same from the root. Use
an absolute interpreter path because the command changes its working directory.

## Independent replay of the actual experiment

Obtain the two clean-input parts from the
[input release](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-blind-v139-cost-gate),
the previous [saved-output capsule](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-blind-v142-development),
and the latest [beam result capsule](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-beam-v144-development).
Verify the checksum sidecars and the prerequisite digests in the latest capsule's
inventory. Extract all four ZIPs **into `archive/research/`**, keeping their internal
paths. Existing matching files must have identical bytes; do not replace differing
data to force a replay to pass.

```sh
lunar-match replay --output runs/saved-replay-new.json
```

This recomputes 56 geometry/Sun/precision cells, 24 historical control identities,
retained branch and field accounting, terrain intervals and the branch diagnosis.
It runs no matcher, fits no policy and needs no private launch authority. Without
the public input artifacts, a fresh checkout can run status/audit and software
controls but cannot reproduce the actual geometry metrics.

## Experimental native matching

The input manifest is a list of inference-only cases with fields `id`,
`reference_geometry`, `catalogue`, `query_image` and `query_intrinsics`. Every file
record pins `path` and SHA-256. Each catalogue item contains a distinct integer
`light_id`, a world-frame unit `sun_world` vector, and an `image` file record.

Relative file paths resolve from the manifest directory by default. Use
`--input-root` explicitly for another base. Historical manifests use the frozen
workspace root as that base:

```sh
lunar-match validate \
  --manifest archive/research/reports/solar-blind-v138/inference-inputs.json \
  --input-root archive/research --case-id blind138-00

lunar-match localize \
  --manifest archive/research/reports/solar-blind-v138/inference-inputs.json \
  --input-root archive/research \
  --case-id blind138-00 --policies beam_active --field-budget 128 \
  --output runs/my-uncalibrated-result.json
```

The public pose remains null. Add `--research-output` to inspect raw proposals.
`validate` decodes inputs and checks geometry without running a matcher.
The field budget counts branch-specific searches, with a maximum of 128 in total.
Output files must be new. This example is a user interface, not authorization for
an agent to bypass the original scientific compute admission or final-data freeze.
Published latency belongs to the frozen experiment. The new interface's additional
validation and process startup have not been benchmarked on lunar inference.

Original `python match_terrain.py ...` and `python localize_terrain.py ...` commands
remain compatibility entry points. They run the original archived scripts. Explicit
input paths are resolved from the caller when present, with historical repository
paths falling back to the archive; outputs remain relative to the caller. For
other historical commands, change into `archive/research/` first.
