# Camera-angle constraint on blind lunar pose proposals

The 52° off-nadir constraint removed the implausible initial poses without recovering a correct camera pose. The original positive banks contained 12 beams, all outside the cone. The constrained banks contained one beam inside the cone, which was wrong. The final estimator also returned a new raw pose on one nonoverlap render. The intervention failed the frozen paired gate.

| Outcome | Original v112 | Initial cone v113 |
|---|---:|---:|
| Correct pose anywhere in positive bank | 0/2 | 0/2 |
| Correct first-ranked positive pose | 0/2 | 0/2 |
| Positive candidates inside the cone | 0/12 | 1/1 |
| Raw beam on nonoverlap | 2/2 | 2/2 |
| Correct positive pose after final AP3P | 0/2 | 0/2 |
| Raw pose after final AP3P on nonoverlap | 0/2 | 1/2 |

The two positives repeat the same changed-light camera pair and the two negatives repeat the same nonoverlap camera pair. All rows use the same six original images and the same reference table, matcher checkpoint, shared-Sun policy, and pose threshold of 2° and 1% of median reference range. The constraint uses a simulator-family attitude bound from the declared camera offsets. It uses no per-image query attitude. The original final 3-pixel AP3P remains independent of the constraint. All reported outputs are raw and uncalibrated.

Across the two positive renders, the cone rejected 3,512 of 3,786 finite initial roots before scoring. The 48 later native AP3P calls returned zero poses. The cone therefore changed the search in the intended stage, while the matched cases remained correspondence-limited. This observation supports testing another source of correspondences before revisiting scattered-light ranking. Generalization requires new terrain. Gravity-aware absolute pose itself has published precedents, and actual use of attitude information needs separately characterized instrument inputs. [Gravity-aware absolute pose](https://sites.cs.ucsb.edu/~holl/pubs/Sweeney-2015-ISMAR.pdf)

## Reproduce the saved result

Extract both the [original v112 archive](solar-secondary-sentry-artifacts-v112.zip) and [v113 supplement](solar-pose-cone-artifacts-v113.zip) in a clean clone. The v113 archive SHA-256 is `70c93a83ddbc9bddaa51aed1a8efc2c16f4b565f1f8ed74df74383d2296d15b8`. It contains four saved blind banks, the six opaque observations, the exact compact checkpoint and the terminal seals. The independent scorer checks all blind inputs and hashes before joining pose truth. The command reproduced the published [saved scores](solar-pose-cone-replay-v113.json) byte for byte in a clean checkout.

```sh
run=runs/solar-pose-cone-v113
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python replay_solar_pose_cone_v113.py \
    --seal-sha256 "$(shasum -a 256 "$run/COMPLETE.json" | cut -d' ' -f1)" \
    --output "$run/replay-check.json"
```

The four blind calls took 43.09 s in total. The one-use canonical reservation charged the full 180 s. Calibration and final reserves were untouched. [Published dispatch](solar-pose-cone-dispatch-v113.json) · [Original scattered-light pilot](SOLAR_SECONDARY_BLIND_SENTRY_V112.md)
