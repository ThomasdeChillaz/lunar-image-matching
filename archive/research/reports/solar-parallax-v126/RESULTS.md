# Native parallax selection result

Both policies recovered zero poses and retained zero native measurements
on the full 16-pair known-light development study. The 32.05-second run
completed prediction, scoring and saved-output replay. The two policies
used identical cell quotas and 12 attempts per nonempty case, yielding
144 attempts each. The near-uniform u025 condition had no initializer
and stayed in the denominator.

On the faint condition, 27 attempts across the policies passed the
coarse correlation score and failed the native-pixel fine score. All
27 fine scores were below the existing 0.55 threshold. Other attempts
failed at the coarse stage. On the bright control, 24 forward fine
matches reached reverse checking, which rejected them on support or
boundary rules. The experiment therefore supplies no measured selection
benefit. The first repair targets the native measurement stage.
[Matching-stage repair](../solar-local-v127/METHOD.md)

The conditions reuse four camera pairs on three development terrain
groups. All results use matching reference illumination. The original
feasibility rule failed, and neither calibrated acceptance nor final
terrain was evaluated.
