# Full24 Direct payload preflight for a matched ARM comparison

The current v98 transport input inventory can accommodate the unchanged
full24 Direct reference matcher. The preflight checked all **768 original
native512 reference renders**, all **32 original camera and map identities**,
the historical source-published Direct profile and the exact closure of
required runtime evidence. The original Direct validator also replayed its
saved procedural controls. No query image was opened, no new model ran and
no budget was reserved.

| Logical payload component | Bytes |
|---|---:|
| Existing staged ARM preflight | 3,660,551,084 |
| Additional Direct members | 106,441,424 |
| Combined logical payload | 3,766,992,508 |
| Headroom below the original 4 GiB cap | 527,974,788 |

The increment contains 677 inputs and seven sources beyond the inherited
preflight. It is smaller than the standalone Direct closure because the
existing ARM payload already includes some reference images and geometry.
Every Direct closure member was hash-checked, role/path-checked and measured.
Every render header was checked for the original 512-pixel 8-bit content.
The new transport members passed the original v98 public text scan and v75
path/size limits. The previously frozen ARM-only members were checked for
presence and size. The staged base protocol and its private portable
reference evidence stay outside the handoff.

The archived v92 execution protocol has a historical machine binding. It
serves as a byte-for-byte proof of the original profile from the checked-out
public Git history. The worker needs the pinned v90 render inventory and
controls and does not need to transport the machine-bound v92 protocol. The
preflight intentionally creates no archive or signed execution artifact.
The normal source, upload, timer, budget and independent replay gates still
apply.

## Worker integration

Preserve the four-arm v98 workflow and its exact guard. Use a distinct
source-published scope if adding Direct. A five-arm qualification would use
the original eight v86 case IDs without editing them. For the separate
larger development comparison use the unchanged 96-, 64- or 48-case v103
lists and the original all-384 manifest as the eligibility authority.
Choose the largest list fitting the measured cap before any comparison
outcomes. The pilot and larger run must share actual worker/software,
profiles, source and coordinate policy. A four-arm pilot measures no
same-worker Direct cost. Record Direct cost with a separately qualified
worker run and never apply an Intel measurement to the ARM budget gate.

The versioned worker should keep one resident original `Backend(profile)`
per profile. Add the original Direct profile and its transport closure in the
freeze. In the Direct branch only, bind the per-case metadata adapter to the
resident backend before constructing the existing capture wrapper.

```python
if profile["backend"] == "direct_render":
    tick = time.monotonic()
    case_backend = InferenceResultCapture(bind_case(resident, case, protocol["input_sha256"]))
    loader.last_setup["direct_reference_identity_setup_seconds"] = time.monotonic() - tick
else:
    case_backend = capture
row, arrays = evaluate_case(case_backend, inp, case, scoring_loader=load_aligned_scoring)
row["raw_inference_result"], row["raw_inference_reporting"] = capture_case_report(case_backend, arrays)
```

The loaded `inp` stays unchanged for `evaluate_case`, so its solar provider
is null and query truth still first appears after inference. Only the
original Direct adapter sees the three reference digests required to check
its library. The other profiles keep the existing input and capture path.
Use the unchanged common pose solve and replay, and include the measured
per-case identity-binding cost in Direct's cold setup. Complete online
inference still includes all original image I/O, SIFT searches and pose
attempts. Historical reference render costs remain separate. Same-worker
relative latency uses the same metric on every arm. The existing process
memory measurement still supports only a worker-wide high-water report.

The worker guard currently rejects a fifth backend and its non-pilot
admission assumes all cases. A versioned guard must explicitly admit the
unchanged original profile and chosen frozen IDs, without relaxing
materialized-reference, admission, publication, resource or scorer
requirements. The preflight does not modify these historical guards.
