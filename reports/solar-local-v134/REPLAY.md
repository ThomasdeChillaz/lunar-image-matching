# Reproduce the native diagnostic

Use the source tag and its development replay bundle. Extract the bundle into
the tagged repository root without changing existing files with different bytes.
The inventory lists every payload file, size and SHA256. Use the validated
environment in `requirements-lock.txt`.

Replay saved measurements and geometry scores without matcher inference:

```sh
python reports/solar-local-v134/replay_saved.py --output replay-v134.json
python reports/solar-local-v134/paired_analysis.py --output paired-replay-v134.json
python reports/solar-local-v134/conditioning_diagnostic.py --output conditioning-replay-v134.json
python verify_solar_smoke_v73.py runs/solar-controlled-eight-toggle-v133 --output procedural-replay-v134.json
```

Compare the first three outputs with the corresponding `results.json`,
`paired-results.json` and `conditioning-diagnostic.json` under this report folder.
Replay checks source and input hashes, the sealed predictions, all 640 score cells,
exact seven-control repetition and endpoint equality across coordinate frames.
The sensitivity diagnostic is explicitly post-hoc. Procedural wiring controls
are separate from lunar performance. Runtime is a saved workload measurement.

The bundle preserves the original v132 generated failure and malformed v133
prediction file. **Do not score or repair that truncated file into a result.**
Its complete inference loop failed at serialization; only v134 has valid sealed
twenty-method results. No failed charge or deadline was reset.

The research controllers require canonical compute admission and a new versioned
protocol for any fresh scientific inference. The replay bundle contains no
reservation journal, launch authority, neural weights, private correspondence, credentials,
or calibration/final imagery. Replaying saved outputs does not authorize a new
experiment. Public matching interfaces and their uncalibrated-null behavior stay
unchanged; no candidate is promoted by this release.
