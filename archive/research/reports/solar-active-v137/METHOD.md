# Approved six-arm lunar development comparison

The user explicitly approved **up to 40 additional CPU wall-minutes** on
5 November 2026. This is a separate 2,400-second supplement. The exhausted original
ledger and its journal remain byte-identical; original training 13,493 seconds
and comparison 0 seconds are not reset or reassigned. No paid compute is used.

## Fixed scope and hypothesis

Run the prepared v136 sequential acquisition algorithm unchanged, with all four
equal-search controls and Direct. Use every v134 image pair: 16 positives and 16
certified nonoverlaps, four camera/illumination conditions on three exposed
terrain components. Preserve both independent rendering-noise variants per image
and every nearly uniform failure. No new geography, calibration or final reserve
is opened. This known-light diagnostic does not test estimated query Sun or the
complete learned A/B/C contribution.

The six fresh methods are:

- Direct: all 24 original reference-light renders plus the matched-light reference,
  SIFT 2,048 features, ratio 0.75, original top-eight proposal selection and final
  common reference-camera AP3P/refit.
- Joint fit without additional patches: the frozen v135 soft-correlation method
  followed by measured grid modes and common AP3P.
- Joint fit plus 32 additional patches, selected uniformly.
- The same budget, selected by reference-preview gradient energy.
- The same budget, selected by geometric hypothesis separation.
- The same budget, selected by robust predicted-observation separation: principal
  `active` method.

All use native 512-pixel content, reference-only geometry and the original six-match,
three-native-pixel final estimator. An internal soft camera remains a proposal;
it is not substituted for the final pose. Camera-conditioned image endpoints
are not independent landmarks. No acceptance rule is fitted.

Keep all twenty v134 saved matching controls unchanged. Fresh Direct endpoints,
pose and estimator must exactly repeat their frozen counterpart. On the eight
cases available from v135, the fresh no-extra result must repeat the frozen joint
result. A mismatch stops the job before scoring.

## Fresh inference and timing

Recompute the original geometry-based proposal bank and eight-candidate,
four-scale photometric initializer from each image pair. Its complete numerical
hypothesis list must equal v134's saved list. The saved camera is a check, not an
inference input. Reference geometry projections and Direct reference descriptors
may be cached as in the prior frozen implementation. Query features, query image
ranking, camera optimization, patch searches and poses are always computed fresh.
Bind caches to the exact reference arrays/settings; record cold misses and
reference preparation costs. The original offline physical-render preparation
remains historical cost, not a new zero-cost rendering claim.

All primary-arm online times include input decoding and fresh initialization.
Direct times include input/library decoding, reference cache misses, fresh query
features, matching and all final candidate estimations. Whole-worker memory is
reported separately and must not become a per-method ranking. Counterbalance the
four acquisition policies across noise combinations and alternate Direct's order.

Query depth, true query pose and true query Sun are never inference inputs.
Load scoring geometry only after the full 32-case predictions are sealed. Replay
all 26 methods x 32 pairs = 832 score cells independently. Keep failures and
rejections in the positive denominator. Negative analysis accounts for both
terrain endpoints.

## Cost gate and enforcement

- **120 seconds:** generated native flat/layered full-pipeline controls, all six
  fresh arms, fresh initialization and the complete 32-extra-search budget for
  every acquisition policy. Direct gets 24 generated gamma-pattern templates to
  exercise extraction; these are not sunlight simulations or independent terrain.
- **1,800 seconds:** the full 32-pair comparison, sealing, scoring and independent
  replay, only if the generated controls pass and the cost estimate fits.
- **480 seconds:** remaining recovery envelope. It is not a free retry; using it
  needs a separately published diagnosis and prospective protocol.

Forecast 1.5 times 32 times the worst complete generated pair, plus 45 seconds for
guards, saved-control copying, scoring and I/O. This is a conservative measured
estimate, not a mathematical maximum over optimizer convergence. Controller and
child watchdogs enforce the original fixed deadline. A shared exclusive lease
prevents overlapping scientific jobs. Use two CPU threads per job; OpenCV uses
one. Failed jobs keep their entire fixed charge and original deadline.

The private supplementary ledger contains the one approved invocation identity;
only its hash and non-sensitive allowance summary are public. It is separate from
the original canonical journal. No ledger, reservation journal or launch ticket
is included in a replay package. New job/version names cannot reset its total.

## Preliminary signal and unchanged full targets

The principal active arm must recover at least two more correct raw positives
than every completed cached/no-extra/Direct control, add no wrong positive or
nonoverlap outputs versus Direct, and recover more positives than each equal-
budget acquisition control. Correct means rotation<=2 degrees and translation
<=1% of median reference range, using the unchanged independent simulation scorer.

This diagnostic cannot establish 1% deployment risk or a general state-of-the-art
ranking. The original targets remain: >=70% correctly accepted poses,>=10-point
accepted improvement with a positive terrain-group interval, empirical accepted
error/nonoverlap each<=1%, modern accuracy/efficiency value, full ablations/seeds,
transfer and untouched final evaluation. Publish negative findings if the rule
fails. No model promotion, email or paper submission is automatic.
