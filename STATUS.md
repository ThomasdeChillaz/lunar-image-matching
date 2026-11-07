# Current status

The repository now separates a maintained interface from a byte-preserved research
archive. All 28 maintained tests and 1,535 archived tests pass. The generated native
CLI and independent 56-cell metric replay also pass. The scientific outcome is unchanged.

- Latest experiment: seven methods, four positives, four nonoverlaps; all methods
  recover the same one positive. Active beam adds no recovery advantage.
- Existing no-extra control: 21.77s median; active beam: 54.52s. Both return zero
  wrong positives and zero nonoverlap poses on this small roster.
- Saved-output replay verifies 56 method/case cells and 24 old-control identities.
- Data: three exposed positive terrain components; four calibration and twelve final
  terrain groups remain untouched.
- Research gaps: camera-candidate coverage, final correspondence/pose precision,
  genuinely fresh evaluation, calibrated acceptance and a clean modern comparison.

[Current results](docs/RESULTS.md) · [Cleanup verification](docs/verification.json) ·
[Original research requirements](reports/research_requirements_status_v144.json)

The complete earlier status log is preserved in archive/research/STATUS.md.
