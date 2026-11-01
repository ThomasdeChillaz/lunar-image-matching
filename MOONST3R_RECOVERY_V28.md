# Explicit recovery amendment for the missing native case

The original four-calendar-hour MOONSt3R comparison is **incomplete:71/72**.
After a host clock/suspension discontinuity, its calendar elapsed time was
41,891seconds while completed case inference totalled3,239seconds. Its live
monotonic alarm had not stopped it. The actor invoked the existing timeout
handler and preserved all71 complete case files and the original protocol.
The precise cause of the clock divergence is not independently established.
Do not claim that this original experiment complied with its calendar cap or
completed72cases, and do not restart its expired deadline.

The user-approved project permits up to72CPUwall hours for pretrained
comparisons. Within that unchanged project allowance, this **separate**
amendment provides at most600 additional calendar seconds, including startup,
for the single missing case. It is an explicit resource-policy amendment,
not an uninterrupted completion of the first protocol. There is no paid
compute, model change, threshold fitting or accuracy-based case choice.

The contract in `reports/moonst3r_recovery_contract_v28.json` freezes all71
original file hashes and the one missing identity before any supplemental
inference. Publish and byte-verify the recovery code and contract first.
The new output directory receives byte-identical copies of the original rows;
the expired directory remains untouched. Same512native full-FOV content,
pretrained source/weights/runtime and common reference-geometry AP3P are used.
Query truth is scoring only; model-predicted3D remains excluded. UTC checks
before/after expensive operations also detect host-clock jumps that an alarm
alone may miss. All failed attempts remain preserved.

```sh
python recover_native_moonst3r_v28.py \
  --source-commit PUBLIC_COMMIT \
  --dependency-dir /path/to/optional/dependencies \
  --contract reports/moonst3r_recovery_contract_v28.json \
  --output runs/localization-v28-moonst3r-amended
python verify_moonst3r_recovery_v28.py \
  --run runs/localization-v28-moonst3r-amended \
  --output reports/native_moonst3r_amended_audit_v28.json
```

If this phase passes, report **amended two-phase72-case results**, with the
original expired71-case study disclosed. The audit checks exact preservation
and independently replays every native metric/common pose. Neither a merged
time statistic nor this resource amendment supports a paired CPU speed claim.
Exact pretrained training-data overlap is still unknown. All terrain is
previously exposed development; fresh final, calibrated acceptance/transfer
and SOTA remain unestablished. No completed amended accuracy result exists yet.
