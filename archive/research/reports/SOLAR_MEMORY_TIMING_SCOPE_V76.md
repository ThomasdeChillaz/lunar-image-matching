# Memory and timing measurement scope

Runner-only memory and timing telemetry

resource.RUSAGE_SELF.ru_maxrss is normalized to bytes: Darwin raw bytes, Linux
raw KiB times1024. Unsupported platforms or resource errors are unavailable, not
zero. Values are worker-process LIFETIME high-water RSS up to each snapshot;
child-process RSS is excluded. Runtime, prior/current profiles, input data,
allocator history and caches remain included. There is no allocation-delta,
per-case RAM or isolated model-memory claim, and shared profiles cannot rank
model RAM. One-profile jobs are explicitly identified as standalone PROFILE
worker processes, still including runtime/data and still excluding children.

Snapshots occur at execution, before/after each profile, and immediately before
summary serialization; no batch sampler or subprocess is introduced. Failure
metadata retains the current profile's last snapshot. Cached-only profile counts
are explicit; prior software-control/cached peaks are never used as fresh worker
samples. Historical cached fields remain historical and are labelled accordingly.

The existing complete_localization_cold_seconds number is unchanged: model-
resident inference plus input/reference setup actually incurred under the reused
loader. It does not force cold inputs for every case. Model startup, independent
pose verification/scoring, offline preparation and cached timing history remain
separate. No numeric pose/timing formula, threshold, gate, budget, scientific
math source or learning-closure source is changed.

Apply worker_memory_telemetry_v75.patch and copy test_solar_worker_memory_v75.py.
The full runner copy is for review; merge the narrow patch into current root.
11 focused metadata/array/stub tests and66 combined evaluation/guard/protocol
regressions pass. Independent reviewer also passed all11 focused tests.
No actual neural forward, image, ray, sampler, scientific job or root edit was
performed for this package. Teacher activation must not wait for this telemetry.
