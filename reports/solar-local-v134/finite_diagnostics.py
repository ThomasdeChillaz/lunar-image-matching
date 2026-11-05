"""Represent unavailable rejected-peak margins as null, without changing decisions."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"solar-local-v133"))
import reference_frame as frame

original_direction = frame.native.direction


def direction(*args, **kwargs):
    result = original_direction(*args, **kwargs)
    margin = result.get("margin")
    if margin is not None and not np.isfinite(margin):
        # Infinity denotes no finite alternative peak. The frozen matcher has
        # already rejected this measurement. Never sanitize an accepted match
        # or conceal a different unexpected nonfinite scientific quantity.
        if result["query_xy"] is not None or not np.isposinf(margin):
            raise ValueError("Unexpected nonfinite accepted measurement diagnostic")
        return dict(result, margin=None, unavailable_peak_comparison=True)
    return result


def install():
    frame.native.direction = direction
    frame.native.control.pipeline.run = frame.run
