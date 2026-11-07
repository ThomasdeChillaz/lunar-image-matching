"""Procedural simulator-family cone for the frozen v73 initial AP3P proposer."""
import ast
from functools import lru_cache
from hashlib import sha256
import math
from pathlib import Path

import cv2
import numpy as np


V73_SHA256 = "7b0b649b5310dc08ff3f9f2c61e8035cff2a9dbfae0994cc238808352151ff73"
CONE_DEGREES = 52.
_MIN_NADIR_DOT = math.cos(math.radians(CONE_DEGREES))
_EPS = 16*np.finfo(float).eps


def within_simulator_cone(world_to_camera_rvec):
    """Use the original world-to-camera convention and reference ENU down."""
    vector = np.asarray(world_to_camera_rvec, dtype=float)
    if vector.size != 3 or not np.isfinite(vector).all():
        return False
    rotation = cv2.Rodrigues(vector.reshape(3, 1))[0]
    return bool(-rotation[2, 2] >= _MIN_NADIR_DOT - _EPS)


def _signature(code):
    return ast.dump(ast.parse(code).body[0])


_ROOT_COUNTER = _signature("info['roots_evaluated'] += 1")
_REFIT_SCORE = _signature("score = quality(errors(obj, pixels, intrinsics, rv, tv))")


class _AddConeChecks(ast.NodeTransformer):
    def __init__(self):
        self.initializers = 0
        self.root_checks = 0
        self.refit_checks = 0

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "info":
            self.initializers += 1
            diagnostic = ast.parse(
                "info.update(simulator_off_nadir_cone_degrees=52., roots_rejected_off_nadir=0, "
                "lm_refits_rejected_off_nadir=0)"
            ).body[0]
            return [node, ast.copy_location(diagnostic, node)]
        if ast.dump(node) == _REFIT_SCORE:
            self.refit_checks += 1
            guard = ast.parse(
                "if not _within_simulator_cone(rv):\n"
                "    info['lm_refits_rejected_off_nadir'] += 1\n"
                "    break"
            ).body[0]
            return [ast.copy_location(guard, node), node]
        return node

    def visit_AugAssign(self, node):
        if ast.dump(node) != _ROOT_COUNTER:
            return node
        self.root_checks += 1
        guard = ast.parse(
            "if not _within_simulator_cone(rv):\n"
            "    info['roots_rejected_off_nadir'] += 1\n"
            "    continue"
        ).body[0]
        return [node, ast.copy_location(guard, node)]


@lru_cache(maxsize=1)
def _specialized_program(source_path):
    raw = Path(source_path).read_bytes()
    if sha256(raw).hexdigest() != V73_SHA256:
        raise ValueError("Frozen v73 AP3P source differs")
    tree = ast.parse(raw, filename=str(source_path))
    proposal, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "propose"]
    guard = _AddConeChecks()
    proposal = guard.visit(proposal)
    if (guard.initializers, guard.root_checks, guard.refit_checks) != (1, 1, 1):
        raise ValueError("Expected the exact v73 minimal-root and LM scoring seams")
    return compile(ast.fix_missing_locations(ast.Module(body=[proposal], type_ignores=[])),
                   str(source_path), "exec")


@lru_cache(maxsize=1)
def _proposer():
    import solar_pose_proposals_v73 as v73

    namespace = dict(vars(v73), _within_simulator_cone=within_simulator_cone)
    exec(_specialized_program(v73.__file__), namespace)
    return namespace["propose"]


def propose_with_simulator_cone(world, pixels, intrinsics, iterations=128):
    """Return an uncalibrated initial v73 pose with a simulator attitude bound.

    Sample allocation and generation, the residual objective, and the maximum
    LM refit count are unchanged. This does not call or change final AP3P.
    """
    return _proposer()(world, pixels, intrinsics, iterations)
