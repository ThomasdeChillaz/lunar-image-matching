# Replaying the five-method development comparison

The [development release](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-five-arm-v115-results)
contains 482 unchanged members. They are the original worker summary,
completion seal, and 480 saved numerical prediction arrays. The archive
excludes workflow authorization, logs, model weights and the source images.
[Archive and run receipt](public-replay-receipt.json) ·
[Paired findings](../SOLAR_FIVE_ARM_RESULTS_V123.md)

## Set up exact replay

Use Linux ARM64 with Python 3.12.15. Check out a fresh directory at
`24046f9c34488fe2efd83464386b0ca8e46906e2`. Place these three files in a
separate directory and set `REPLAY_INPUTS` to its absolute path.

- `solar-five-arm-v115-replay-data.zip`, SHA-256
  `f19f372f46bd874e780b3b803e0e846765044968c9e87b555863594ddc5c6715`.
  [Development release](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-five-arm-v115-results)
- `solar-arm-v98-public.zip`, SHA-256
  `2d25dca8b56f86f10444289b7458aeed4ba9a6755bf1ff19fca7014afaa1dee0`.
  [Original public input archive](https://github.com/ThomasdeChillaz/lunar-image-matching/releases/tag/solar-arm-transport-v98)
- `replay_input_hashes_v115.json`, SHA-256
  `bd856d419adb42c40a0f5b4ff7ab452e1918a7280a74d96584382ecd53f54d09`.
  [Exact selected input list](replay_input_hashes_v115.json)

From the fresh checkout, verify the archives and extract just the selected
replay inputs. The input archive also contains files that the replay never
extracts or loads.

```sh
test "$(git rev-parse HEAD)" = 24046f9c34488fe2efd83464386b0ca8e46906e2
python - <<'PY'
from hashlib import file_digest, sha256
from pathlib import Path
from zipfile import ZipFile
import json
import os

inputs = Path(os.environ["REPLAY_INPUTS"])
digests = {
    "solar-five-arm-v115-replay-data.zip": "f19f372f46bd874e780b3b803e0e846765044968c9e87b555863594ddc5c6715",
    "solar-arm-v98-public.zip": "2d25dca8b56f86f10444289b7458aeed4ba9a6755bf1ff19fca7014afaa1dee0",
    "replay_input_hashes_v115.json": "bd856d419adb42c40a0f5b4ff7ab452e1918a7280a74d96584382ecd53f54d09",
}
for name, expected in digests.items():
    with (inputs / name).open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"Unexpected SHA-256 for {name}")

selected = json.loads((inputs / "replay_input_hashes_v115.json").read_text())
with ZipFile(inputs / "solar-arm-v98-public.zip") as base:
    for name, expected in selected.items():
        path = Path(name)
        data = path.read_bytes() if path.exists() else base.read("payload/" + name)
        if sha256(data).hexdigest() != expected:
            raise ValueError(f"Unexpected source bytes for {name}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)

with ZipFile(inputs / "solar-five-arm-v115-replay-data.zip") as replay:
    if len(set(replay.namelist())) != 482:
        raise ValueError("Expected the summary, seal and 480 saved arrays")
    replay.extractall(".")
PY

rg '^(numpy|opencv-python) @' requirements_arm_cp312_v113.txt > /tmp/lunar-v115-replay-wheels.txt
python -m pip install --no-deps --require-hashes -r /tmp/lunar-v115-replay-wheels.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  python -B verify_solar_matched_arm_v113.py runs/solar-five-arm-v115/comparison \
  --protocol reports/solar-five-arm-v115/comparison-protocol.json \
  --output independent_replay.json
cmp independent_replay.json "$REPLAY_INPUTS/independent_replay_v115.json"
```

For the last command, place the
[published exact replay receipt](independent_replay_v115.json) beside the
three inputs. The strict verifier may reject tiny projection differences
on x86. On the tested x86 runtime, one projected scalar differed by
1.42 × 10⁻¹⁴ native pixels and every scoring threshold decision reproduced.
[Cross-platform numerical check](portability_recheck_v123.json)

The separate [ARM audit job](https://github.com/ThomasdeChillaz/lunar-image-matching/actions/runs/41964097603)
also verified the original Actions container against its GitHub digest,
all individual file seals, and the official origin. The reduced release
supports the independent score replay above. Original workflow custody
uses the official Actions artifact.
