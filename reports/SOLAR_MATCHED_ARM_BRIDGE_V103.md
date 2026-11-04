# Exact full-reference Direct bridge for a future paired ARM worker

The existing DirectRenderAdapter requires a reference lookup object only to
check three original reference geometry and mesh digests. The ARM loader
correctly sets the live lookup to null because its physics are materialized.
The bridge proves the known small v80 lookup JSON against both the case and
execution allowlists and the original published Git object. It checks the
source camera, full reference map, approved reference key and original audit.
It exposes only those three digests inside an unchanged Direct call. The
generic evaluator continues to see its original materialized input with no
live lookup. It supplies no query Sun, depth, pose or visibility label.

Use a distinct future protocol, relay scope and worker source after the v98
cost pilot. Preserve the old runner and do not change v98 source/ticket scope.
Bind the bridge only for the Direct arm, keeping one resident Direct backend.
The following illustrates the hook inside the future runner's existing per-case
loop:

```python
if profile["backend"] == "direct_render":
    case_backend = InferenceResultCapture(bind_case(resident_backend, case, protocol["input_sha256"]))
else:
    case_backend = resident_capture
row, arrays = evaluate_case(case_backend, inp, case, scoring_loader=load_aligned_scoring)
```

The original Backend may wrap DirectRenderAdapter. Call the same original
native pose replay and raw-result capture for each case. Count reference
metadata binding in cold per-case setup or in the timed Direct call, and
record model load and resource qualification separately. Do not record an
online physics lookup. Keep the same 512 native image content, reference
geometry, pose estimator, coordinate conventions and 3-pixel limit.

The unchanged Direct profile gate requires its exact archived source and
library evidence, including all 768 sealed reference renders from 32 cameras.
The scope extension adds about 123 MB under that gate. Freeze the real
additional source, direct profile pins, raw reference geometry and library
inputs in the new protocol. Do not transport the entire old v91/v86 pilot
protocol when a smaller exact Direct evidence record supplies its lineage.
The v86 protocol has an old machine path and the v98 public transport
projection intentionally accepts only the known original v80 exception.

## Prospective cohort and compute

The current canonical baseline remainder is 35,879 seconds. The 1,800-second
v98 eight-case four-profile pilot is the first actual modern measurement.
Historical ARM native cases on a different bank reached 54.693 seconds for
LoMa-R and 53.678 seconds for RoMa v2. The two warm maxima alone project about
41,614 seconds across 384 cases, and about 62,421 under the 1.5× admission
factor. The remaining allowance and one-job 18,000-second limit cannot
support a two-model full-bank claim from those historical observations.

After exact ARM qualification, predeclare a paired development subset across
all eight groups, with positive/nonoverlap pairs retained and both negative
endpoints recorded. A 96-case design selects one motion and one roll camera
per group, one same-light and two changed-light conditions per camera, and
the matched nonoverlap for each positive. Select by geometry/metadata alone
and balance the six frozen lighting combinations across groups. If measured
startup-inclusive ARM case peaks and setup leave insufficient cap, declare
the smaller 64-case two-light version before new outcomes, or use 48 cases
with one camera and three conditions. Measure Direct on the same worker first
for its actual setup cost. Label any reduced comparison explicitly as a
development subset. Preserve the independent full 384-case Direct evidence.

The required common arms are frozen compact, SIFT, LoMa-R, RoMa v2, and the
original full24 Direct control. A separate same-worker even12 Direct arm
tests withheld lighting if the cost gate fits, with the existing v95 exact
filter and no truth input. Author model code and weights stay at their source
pins, including native BF16 support for the same pretrained ARM path, LoMa-R
internal504 and RoMa v2 fast512 settings. Accuracy and paired latency across
these methods can then be compared on the same worker. No acceptance,
calibration, final reserve, per-model memory rank or transfer result follows
from this development comparison.

Five fixture tests passed, including map/camera substitution, input hash and
public proof mismatch, prior injection, oversized metadata, and the immutable
three-digest boundary. The exact local and published lookup proof passed for
all eight real v86 metadata cases. No images, arrays, matchers or models ran.
