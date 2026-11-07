# Development

Use an editable Python 3.12 environment. Change maintained interfaces in
`src/lunar_matching/` and put behavioral tests in `tests/`. Keep inference inputs,
geometric labels and acceptance decisions explicit. Use atomic exclusive writes
for results; create a new output rather than overwriting an experiment.

Run the maintained tests, archive byte audit, complete archived regression and a
fresh generated control. Replay the published metrics after evidence or interface
changes. Changes to the actual algorithm, data or scorer require a separately
versioned scientific study; the original accounting and untouched reserves persist.

`archive/research/` is immutable computational history. The public `reports/` mirror
keeps earlier links readable. New reports belong in `docs/` or a new versioned study.
Do not reformat historical scripts or modify a source hash to make a check pass.

Keep private emails, drafts, local execution tickets and credentials outside Git.
The archive is ignored for untracked additions; its existing tracked files remain
versioned. Review `git diff --cached` before committing. Commit normally without
rewriting previous scientific history.
