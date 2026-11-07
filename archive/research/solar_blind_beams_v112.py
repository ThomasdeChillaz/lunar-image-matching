"""Opt-in research capture of the frozen v84 blind solver's final candidate bank."""

import ast
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import math
from pathlib import Path

import numpy as np


_SOURCE_SEALS = {
    "solar_inference_cache_v84.py": "732627a68f9422533e0e633df78526600c6703d75e6674f603b1cfd0c102b769",
    "solar_assignment_v63.py": "ebb84f4e04a1b037f427bdba4266fa776fdb1c32f7ba0518eaa24c5dbc592861",
    "solar_frontend_v63.py": "cdbd20fd07b3d8582e82d27e56b10295e7f2c5671b21927aa74790a72a4407c0",
}
_EARLY_REASONS = {
    "blank_or_uninformative_image", "insufficient_reference_surface", "no_candidate_assignments",
}
_SELECTION = ast.dump(ast.parse("best = max(candidates or beams, key=lambda b: b['score'])").body[0])


@dataclass(frozen=True)
class BlindBeamV112:
    source_beam_index: int
    pose: tuple[tuple[float, ...], ...]
    sun_reference_enu: tuple[float, float, float]
    score: float
    sun_index: int


@dataclass(frozen=True)
class BlindBeamBank:
    candidates: tuple[BlindBeamV112, ...]
    capture_reached: bool
    reason: str | None

    def to_dict(self):
        return dict(schema="solar_blind_beams_v112", track="estimated_sun",
                    pose_frame="camera_to_reference_scene_enu", sun_frame="reference_scene_enu",
                    capture_reached=self.capture_reached, reason=self.reason,
                    candidates=[asdict(candidate) for candidate in self.candidates])


def _snapshot(graph, beams, candidates, best):
    """Copy only the exact original eligible poses and their final Sun vectors."""
    if not candidates:
        return BlindBeamBank((), True, "no_post_filter_pose_candidates")
    beam_indices = {id(beam): index for index, beam in enumerate(beams)}
    scores = [float(beam["score"]) for beam in candidates]
    if (not all(math.isfinite(score) for score in scores)
            or not 0 < len(candidates) <= len(beams) <= 8
            or len({id(beam) for beam in candidates}) != len(candidates)
            or any(id(beam) not in beam_indices for beam in candidates)):
        raise ValueError("Unexpected final blind candidate bank")
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    if candidates[order[0]] is not best:
        raise ValueError("Observed order disagrees with the frozen blind winner")
    result = []
    for i in order:
        beam = candidates[i]
        pose = np.asarray(beam["pose"], dtype=float)
        index = beam["sun_index"]
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], (0., 0., 0., 1.), rtol=0., atol=1e-8)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), rtol=0., atol=1e-5)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1., rtol=0., atol=1e-5)
                or not isinstance(index, (int, np.integer)) or not 0 <= index < len(graph.sun_vectors)
                or not graph.solar_materialized_mask[index]):
            raise ValueError("Invalid blind pose or uncomputed final Sun state")
        sun = np.asarray(graph.sun_vectors[index], dtype=float)
        if (sun.shape != (3,) or not np.isfinite(sun).all()
                or not np.isclose(np.linalg.norm(sun), 1., rtol=0., atol=1e-6)):
            raise ValueError("Final Sun must be an exact unit reference-ENU vector")
        result.append(BlindBeamV112(beam_indices[id(beam)], tuple(tuple(map(float, row)) for row in pose),
                                   tuple(map(float, sun)), scores[i], int(index)))
    return BlindBeamBank(tuple(result), True, None)


class _ObserveSelection(ast.NodeTransformer):
    def __init__(self):
        self.matches = 0

    def visit_Assign(self, node):
        if ast.dump(node) != _SELECTION:
            return node
        self.matches += 1
        callback = ast.parse("_capture_v112(graph, beams, candidates, best)").body[0]
        return [node, ast.copy_location(callback, node)]


@lru_cache(maxsize=1)
def _observed_program(source_path):
    """Compile two frozen function bodies in a private namespace, with one observer."""
    source = Path(source_path)
    for name, checksum in _SOURCE_SEALS.items():
        if sha256(source.with_name(name).read_bytes()).hexdigest() != checksum:
            raise ValueError("Frozen observer dependency changed: " + name)
    tree = ast.parse(source.read_bytes(), filename=str(source))
    solver, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "solve_v84"]
    frontend, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SolarMatcherV84"]
    localize, = [n for n in frontend.body if isinstance(n, ast.FunctionDef) and n.name == "localize"]
    observer = _ObserveSelection()
    solver = observer.visit(solver)
    if observer.matches != 1:
        raise ValueError("Expected one exact final pre-AP3P selection seam")
    program = ast.Module(body=[solver, localize], type_ignores=[])
    return compile(ast.fix_missing_locations(program), str(source), "exec")


def localize_with_blind_beams(matcher, reference, query, geometry, query_intrinsics, *,
                             response=False, active=False, budget=128, policy="active",
                             reference_sun=None, final_estimator=None, solar_lookup=None):
    """Run the exact v84 shared-Sun frontend and capture eligible pre-AP3P beams.

    This research-only entry point has no known-query-Sun or query truth
    parameter. The ordinary result and final estimator behavior are retained.
    Only the frozen v84 implementation is supported.
    """
    import solar_inference_cache_v84 as frozen

    if type(matcher) is not frozen.SolarMatcherV84:
        raise ValueError("The blind observer requires the exact frozen SolarMatcherV84")
    captured = []

    def observe(graph, beams, candidates, best):
        if captured:
            raise ValueError("Multiple final blind candidate captures")
        captured.append(_snapshot(graph, beams, candidates, best))

    namespace = dict(vars(frozen), _capture_v112=observe)
    exec(_observed_program(frozen.__file__), namespace)
    result = namespace["localize"](
        matcher, reference, query, geometry, query_intrinsics, solar_mode="shared",
        response=response, active=active, budget=budget, policy=policy,
        reference_sun=reference_sun, final_estimator=final_estimator, solar_lookup=solar_lookup,
    )
    if not captured:
        reason = result.get("reason")
        if reason not in _EARLY_REASONS:
            raise ValueError("Frozen solver unexpectedly bypassed the blind candidate capture")
        return result, BlindBeamBank((), False, reason)
    return result, captured[0]
