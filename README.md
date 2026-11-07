# Lunar image matching and camera localization

An independent research project studying how illumination ambiguity, correspondence
precision and geometric conditioning affect lunar camera localization.

The work includes a native-resolution matcher, controlled comparisons, saved-output
replay and a documented negative result. **The latest persistent-hypothesis method
does not improve pose recovery over the existing control.** The useful contribution
so far is a reproducible experiment and a concrete diagnosis of where matching fails.

[Research brief](docs/RESEARCH_BRIEF.md) · [Measured results](docs/RESULTS.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Reproduce](docs/REPRODUCING.md)

## Latest measured experiment

Seven methods were compared on the same eight native 512-pixel lunar-render pairs:
four positives and four nonoverlaps, covering three exposed positive terrain groups.

| Method | Correct raw poses | Wrong positive poses | Nonoverlap poses | Median positive seconds |
|---|---:|---:|---:|---:|
| Existing single hypothesis, no extra search | 1/4 | 0 | 0/4 | 21.77 |
| Single hypothesis, full field allowance | 1/4 | 2 | 0/4 | 50.17 |
| Persistent hypotheses, uniform selection | 1/4 | 0 | 0/4 | 50.35 |
| Persistent hypotheses, active selection | 1/4 | 0 | 0/4 | 54.52 |

Correct means rotation error at most 2° and translation error at most 1% of median
reference range. All eligible positives remain in the denominator. These are raw
development poses, not calibrated acceptances or independently verified orbital results.
The full seven-method table, criteria, uncertainty and failures are in [Results](docs/RESULTS.md).

The faint case retains the correct catalogue illumination but no useful camera
candidate. Two nearly uniform cases expose a second issue: accurate intermediate
camera fits can produce inaccurate final poses after discrete correspondence extraction.
These observations motivate work on camera coverage and the correspondence-to-pose handoff.

## Run the maintained workflow

Use Python 3.12. Installation is editable because the complete frozen research
workspace is part of this repository; a standalone wheel is not a research capsule.

```sh
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
lunar-match status
lunar-match audit
```

`status` reads the hash-checked measured results. `audit` verifies all 2,288 original
public files. Neither command runs a matcher, downloads data or needs model weights.

The generated native-matching control additionally needs the validated CPU neural
runtime, because the frozen import chain includes historical neural components:

```sh
python -m pip install torch==2.2.2 torchvision==0.17.2
lunar-match smoke
```

This control uses generated patterns and deliberately supplied camera hypotheses.
It is a software check, not evidence of lunar generalization. Follow
[Reproducing](docs/REPRODUCING.md) for saved geometry replay and native input matching.

## Repository map

| Location | Responsibility |
|---|---|
| `src/lunar_matching/` | Maintained CLI, input contracts, evidence checks, atomic output and isolated backend execution |
| `tests/` | Maintained interface and regression tests |
| `docs/` | Current research brief, architecture, reproduction and byte-preservation inventory |
| `archive/research/` | Original computational workspace, with every tracked file preserved byte-for-byte |
| `reports/` | Readable copies of previously published reports, preserving existing report links |

The historical backend is isolated in a child process. Importing the maintained
package performs no scientific run and does not inject historical module names into
the caller. Original `match_terrain.py` and `localize_terrain.py` command names remain
available as compatibility launchers. The archived versions retain the original code.

## Verification and research status

```sh
python -m pytest -q
make test-archive PYTHON=/absolute/path/to/python
lunar-match smoke
lunar-match replay
```

The published study independently replays all 56 method/case cells and 24 historical
control identities. Current cleanup verification is recorded in
[the verification receipt](docs/verification.json). Earlier benchmark results,
checkpoints and source/data freezes remain historical evidence. The original
[research targets](reports/research_requirements_status_v144.json) remain open where
unmet; the cleanup changes organization and input handling, not measured accuracy.
