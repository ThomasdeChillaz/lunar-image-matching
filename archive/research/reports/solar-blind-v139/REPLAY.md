# Read-only cost-gate reproduction

From a checkout plus the checksummed clean-input capsule, run:

```sh
python reports/solar-blind-v139/replay_cost.py
```

This uses Python's standard library only. It verifies all 101 frozen source and
125 input files, then independently computes `12 * max(pair_seconds) + 30` from
both complete cost controls. The result is 652.51 seconds, exceeding even the
entire original 480-second cap before accounting for the pilot's own runtime.
It also checks the complete eight-case/six-method roster and public failure
receipt. No private ledger or execution ticket is required.

The run stopped before lunar inference. This script replays hashes and admission
arithmetic; it does not independently reconstruct the live generated pose checks
from saved pose arrays, which the cost-only record does not contain. Those checks
are enforced in the frozen engine and covered by its software tests. No new lunar
accuracy or learned-model evidence is created by running this replay.

The proposed v140 continuation preserves the failed run and unchanged methods;
its execution requires a separate user resource decision and public freeze.
