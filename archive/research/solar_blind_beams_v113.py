"""Private research-only integration of the v113 cone into v84 initial proposals."""

import ast
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np

import solar_blind_beams_v112 as v112
import solar_pose_cone_v113 as cone


_OBSERVER_SHA256 = "2ff9b1eb0a4ed202f7334d5c32abe7ee9ca65ebc537338cd1651a1385f3b7880"
_CONE_SHA256 = "23b903d807bacd805e6e5d307ad5567a4246492440b3121c9ae611b9565727f4"
_SEAMS = {
    "import": "from solar_pose_proposals_v73 import propose",
    "native_ap3p": "native_pose, info = estimate(w, q, intrinsics, dict(solver='ap3p', iterations=allocation, reclassify=True))",
    "native_lm": "b['pose'] = _lm_update(native_pose, w, q, intrinsics)",
}
_SIGNATURES = {key: ast.dump(ast.parse(code).body[0]) for key, code in _SEAMS.items()}
_INITIAL_COUNTERS = ("iterations_requested", "minimal_sets_executed", "roots_evaluated",
                     "roots_rejected_off_nadir", "lm_refits_rejected_off_nadir")


class _IntegrateCone(v112._ObserveSelection):
    def __init__(self):
        super().__init__()
        self.integrations = dict.fromkeys(_SEAMS, 0)

    def visit_ImportFrom(self, node):
        if ast.dump(node) != _SIGNATURES["import"]:
            return node
        self.integrations["import"] += 1
        return ast.copy_location(ast.parse("propose = _v113_initial_proposal").body[0], node)

    def visit_Assign(self, node):
        signature = ast.dump(node)
        for key, expression in (("native_ap3p", "native_pose"), ("native_lm", "b['pose']")):
            if signature == _SIGNATURES[key]:
                self.integrations[key] += 1
                callback = ast.parse(f"_v113_observe_native({key!r}, {expression})").body[0]
                return [node, ast.copy_location(callback, node)]
        return super().visit_Assign(node)


@lru_cache(maxsize=1)
def _cone_observed_program(source_path):
    source = Path(source_path)
    pins = {**v112._SOURCE_SEALS, "solar_blind_beams_v112.py": _OBSERVER_SHA256,
            "solar_pose_cone_v113.py": _CONE_SHA256}
    for name, expected in pins.items():
        if sha256(source.with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen cone integration dependency changed: " + name)
    if (Path(v112.__file__).resolve() != source.with_name("solar_blind_beams_v112.py").resolve()
            or Path(cone.__file__).resolve() != source.with_name("solar_pose_cone_v113.py").resolve()):
        raise ValueError("Cone and observer modules must load from the pinned source directory")
    tree = ast.parse(source.read_bytes(), filename=str(source))
    solver, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve_v84"]
    frontend, = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SolarMatcherV84"]
    localize, = [node for node in frontend.body if isinstance(node, ast.FunctionDef) and node.name == "localize"]
    observer = _IntegrateCone()
    solver = observer.visit(solver)
    if observer.matches != 1 or set(observer.integrations.values()) != {1}:
        raise ValueError("Expected the exact frozen initial, native and final selection seams")
    program = ast.Module(body=[solver, localize], type_ignores=[])
    return compile(ast.fix_missing_locations(program), str(source), "exec")


def _off_nadir(pose):
    if pose is None:
        return None
    matrix = np.asarray(pose, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    return bool(-matrix[2, 2] < cone._MIN_NADIR_DOT - cone._EPS)


def localize_with_blind_cone_v113(matcher, reference, query, geometry, query_intrinsics, *,
                                  response=False, active=False, budget=128, policy="active",
                                  reference_sun=None, final_estimator=None, solar_lookup=None):
    """Return the ordinary result, pre-final beam bank and cone-stage counters.

    Only the quantization-aware v73 root search/refit is constrained. Later
    native v84 pose calls and the original final AP3P can leave the cone. Their
    results are observed without filtering. Shared estimated-Sun is fixed.
    """
    import solar_inference_cache_v84 as frozen

    if type(matcher) is not frozen.SolarMatcherV84 or matcher.pose_initialization != "quantization_aware":
        raise ValueError("Cone research path requires the exact quantization-aware SolarMatcherV84")
    captures = []
    diagnostics = dict(
        schema="solar_blind_cone_diagnostics_v113", cone_degrees=cone.CONE_DEGREES,
        initial_v73=dict(calls=0, returned_poses=0, **dict.fromkeys(_INITIAL_COUNTERS, 0)),
        native_v84={stage: dict(calls=0, returned_poses=0, off_nadir_poses=0, invalid_poses=0)
                    for stage in ("native_ap3p", "native_lm")},
    )

    def initial(world, pixels, intrinsics, iterations):
        pose, info = cone.propose_with_simulator_cone(world, pixels, intrinsics, iterations)
        record = diagnostics["initial_v73"]
        record["calls"] += 1
        record["returned_poses"] += int(pose is not None)
        for key in _INITIAL_COUNTERS:
            record[key] += int(info[key])
        return pose, info

    def observe_native(stage, pose):
        record = diagnostics["native_v84"][stage]
        record["calls"] += 1
        if pose is not None:
            record["returned_poses"] += 1
            outside = _off_nadir(pose)
            record["invalid_poses"] += int(outside is None)
            record["off_nadir_poses"] += int(outside is True)

    def capture(graph, beams, candidates, best):
        if captures:
            raise ValueError("Multiple final blind candidate captures")
        captures.append(v112._snapshot(graph, beams, candidates, best))

    namespace = dict(vars(frozen), _v113_initial_proposal=initial, _v113_observe_native=observe_native,
                     _capture_v112=capture)
    exec(_cone_observed_program(frozen.__file__), namespace)
    result = namespace["localize"](
        matcher, reference, query, geometry, query_intrinsics, solar_mode="shared",
        response=response, active=active, budget=budget, policy=policy,
        reference_sun=reference_sun, final_estimator=final_estimator, solar_lookup=solar_lookup,
    )
    if captures:
        bank = captures[0]
    else:
        reason = result.get("reason")
        if reason not in v112._EARLY_REASONS:
            raise ValueError("Frozen solver unexpectedly bypassed the blind candidate capture")
        bank = v112.BlindBeamBank((), False, reason)
    diagnostics.update(
        pre_final_beam_poses=len(bank.candidates),
        pre_final_off_nadir_poses=sum(_off_nadir(beam.pose) is True for beam in bank.candidates),
        ordinary_final_off_nadir=_off_nadir(result.get("pose")),
        ordinary_final_iterations_reported=result.get("final_estimator_iterations"),
    )
    return result, bank, diagnostics
