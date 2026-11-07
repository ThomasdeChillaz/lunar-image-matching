"""Prospective patch/exposure selection with a fixed sensor model and budget."""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.special import gammainc, gammaincc


HEIGHT_VARIANTS_M = (0., -5., 5., -10., 10., -20., 20.)
POLICIES = ("fixed_uniform", "fixed_gradient", "fixed_hypothesis", "joint")


@dataclass(frozen=True)
class Sensor:
    relative_to_electrons_per_second: float
    dark_electrons_per_second: float
    read_noise_electrons: float
    full_well_electrons: int
    seconds_per_tick: float

    def __post_init__(self):
        values = (self.relative_to_electrons_per_second, self.dark_electrons_per_second,
                  self.read_noise_electrons, self.seconds_per_tick)
        if (not all(math.isfinite(v) for v in values) or min(values) < 0
                or not all(values[i] > 0 for i in (0, 2, 3))
                or not isinstance(self.full_well_electrons, int) or self.full_well_electrons < 1):
            raise ValueError("Sensor rates, read noise, integration unit and integer full well are required")


@dataclass(frozen=True)
class Budget:
    total_ticks: int
    frames: int
    allowed_ticks: tuple[int, ...]

    def __post_init__(self):
        ticks = (self.total_ticks, self.frames, *self.allowed_ticks)
        if not all(type(t) is int and t > 0 for t in ticks) or not self.allowed_ticks:
            raise ValueError("Positive integer integration and frame budgets are required")
        if tuple(sorted(set(self.allowed_ticks))) != self.allowed_ticks:
            raise ValueError("Durations must be sorted and unique")
        if self.total_ticks % self.frames or self.fixed_ticks not in self.allowed_ticks:
            raise ValueError("Equal fixed duration must be feasible for all control frames")

    @property
    def fixed_ticks(self):
        return self.total_ticks // self.frames


def clipped_poisson_moments(lam, well, read_sigma):
    """Return charge mean, shot+read variance and saturation probability.

    Charge is min(Poisson(lam), well). Read noise is independent and added
    after clipping the collected charge. Truncated Poisson moments give
    the charge moments without iterating over individual electron counts.
    """
    lam = np.asarray(lam, float)
    if not np.isfinite(lam).all() or (lam < 0).any():
        raise ValueError("Poisson expectation must be finite and nonnegative")
    # The Poisson recurrence shifts truncated first and factorial second
    # moments to ordinary Poisson CDFs. Cost is independent of full-well size.
    saturated = gammainc(well, lam)
    first = lam*gammaincc(well-1, lam) if well >= 2 else np.zeros_like(lam)
    factorial_second = lam*(lam*gammaincc(well-2, lam)) if well >= 3 else np.zeros_like(lam)
    mean = np.clip(first + well*saturated, 0., well)
    variance = np.maximum(first + factorial_second + well**2*saturated - mean**2, 0.) + read_sigma**2
    return mean, variance, saturated


def separation_table(irradiance, valid, sensor, allowed_ticks):
    """Return worst-case standardised mean gaps for H×N×7×P inputs."""
    levels = np.where(valid, irradiance, 0.)
    complete = valid.all(axis=2)
    pairs = [(a, b) for a in range(len(levels)) for b in range(a+1, len(levels))]
    shared = np.array([complete[a] & complete[b] for a, b in pairs], dtype=bool)
    if not pairs:
        shared = np.empty((0, *complete.shape[1:]), bool)
    n_actions = levels.shape[1]
    increments = np.zeros((n_actions, len(allowed_ticks), len(pairs)))
    saturation = np.zeros((n_actions, len(allowed_ticks)))
    has_pred = complete.any(axis=(0, 2))
    rate = sensor.relative_to_electrons_per_second*levels + sensor.dark_electrons_per_second
    for column, duration in enumerate(allowed_ticks):
        mu, variance, saturated = clipped_poisson_moments(
            rate * (duration*sensor.seconds_per_tick), sensor.full_well_electrons, sensor.read_noise_electrons,
        )
        low, high = mu.min(axis=2), mu.max(axis=2)
        high_variance = variance.max(axis=2)
        max_sat = np.where(complete, saturated.max(axis=2), 0.).max(axis=(0, 2))
        saturation[:, column] = np.where(has_pred, max_sat, np.nan)
        for p, (a, b) in enumerate(pairs):
            gap = np.maximum(0., np.maximum(low[a]-high[b], low[b]-high[a]))
            standardised = gap**2 / (high_variance[a] + high_variance[b])
            increments[:, column, p] = np.where(shared[p], standardised, 0.).sum(axis=-1)
    return pairs, shared, increments, saturation


def plan_exposures(irradiance, valid, weights, patch_xy, reference_gradient, sensor, budget, policy):
    """Plan from reference-side predictions and current hypothesis weights only.

    Each action is an exposure duration plus one predeclared fixed-size ROI
    from that frame. Geometry, Sun and radiometry are computed upstream for
    each *hypothesized* state. No camera observations enter this routine.
    """
    levels = np.asarray(irradiance, float)
    mask = np.asarray(valid, bool)
    if (levels.ndim != 4 or levels.shape[2] != len(HEIGHT_VARIANTS_M) or min(levels.shape) < 1
            or mask.shape != levels.shape or not np.isfinite(levels[mask]).all() or (levels[mask] < 0).any()):
        raise ValueError("Need H×N×7×P nonnegative reference irradiance with explicit coverage")
    weights = np.asarray(weights, float)
    points, gradient = np.asarray(patch_xy, float), np.asarray(reference_gradient, float)
    hypotheses, actions = levels.shape[:2]
    if (weights.shape != (hypotheses,) or not np.isfinite(weights).all() or (weights <= 0).any()
            or not math.isfinite(float(weights.sum()))):
        raise ValueError("Finite positive weights are required for surviving hypotheses")
    if (actions < budget.frames or points.shape != (actions, 2) or not np.isfinite(points).all()
            or ((points < 0) | (points >= 512)).any() or len(np.unique(points, axis=0)) != actions
            or gradient.shape != (actions,) or not np.isfinite(gradient).all() or (gradient < 0).any()):
        raise ValueError("Unique native patch centres and reference gradients must support every frame")
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    weights = weights / weights.sum()
    durations = budget.allowed_ticks
    pairs, shared, increments, saturation = separation_table(levels, mask, sensor, durations)
    pair_weights = np.array([weights[a]*weights[b] for a, b in pairs])
    # The surrogate is bounded per pair, so another frame has diminishing
    # value when the pair is already well separated by the earlier selections.
    accumulated = np.zeros(len(pairs))
    cell_xy = points.astype(int) // 128
    cells = cell_xy[:, 1]*4 + cell_xy[:, 0]
    visits = np.zeros(16, dtype=int)
    unused = np.ones(actions, bool)
    records = []
    left = budget.total_ticks

    @lru_cache(maxsize=None)
    def feasible(frames, ticks):
        if not frames:
            return ticks == 0
        if ticks < frames*durations[0] or ticks > frames*durations[-1]:
            return False
        return any(feasible(frames-1, ticks-t) for t in durations)

    fixed_column = durations.index(budget.fixed_ticks)
    for frames in range(budget.frames, 0, -1):
        available = np.flatnonzero(unused)
        feasible_columns = ([j for j, t in enumerate(durations) if feasible(frames-1, left-t)]
                            if policy == "joint" else [fixed_column])
        scores = (-np.expm1(-increments/4)) @ (pair_weights * np.exp(-accumulated/4))
        if policy in ("fixed_hypothesis", "joint"):
            action, column = min(
                ((i, j) for i in available for j in feasible_columns),
                key=lambda ij: (-scores[ij], visits[cells[ij[0]]], cells[ij[0]],
                                abs(durations[ij[1]]-left/frames), ij[0], durations[ij[1]]),
            )
        else:
            best = lambda i: (-gradient[i] if policy == "fixed_gradient" else 0, visits[cells[i]], cells[i], i)
            action, column = min(available, key=best), fixed_column
        duration = durations[column]
        prob = saturation[action, column]
        records.append(dict(action=int(action), duration_ticks=duration,
                            incremental_expected_separation=float(scores[action, column]),
                            max_supported_saturation_probability=float(prob) if np.isfinite(prob) else None))
        left -= duration
        unused[action] = False
        visits[cells[action]] += 1
        accumulated += increments[action, column]
    if left != 0:
        raise RuntimeError("Every policy must spend exactly the declared integration and frame budget")

    pair_diagnostics = []
    for p, (a, b) in enumerate(pairs):
        common_pixels = shared[p].sum(axis=1)
        bound = increments[..., p].max()
        status = ("separable_some_action" if bound > 1e-12 else
                  "unresolved_with_missing_coverage" if not shared[p].all() else "unresolved_on_complete_atlas")
        pair_diagnostics.append(dict(hypotheses=[a, b], status=status,
                                     actions_with_shared_pixels=int((common_pixels > 0).sum()),
                                     actions_with_complete_shared_pixels=int(shared[p].all(axis=1).sum()),
                                     max_single_frame_expected_separation=float(bound),
                                     accumulated_expected_separation=float(accumulated[p])))
    score = float((-np.expm1(-accumulated/4)) @ pair_weights)
    return dict(policy=policy, integration_ticks=budget.total_ticks,
                integration_seconds=budget.total_ticks*sensor.seconds_per_tick, frames=budget.frames,
                pixels_per_frame=levels.shape[-1], records=records,
                expected_separation=score, pair_diagnostics=pair_diagnostics)
