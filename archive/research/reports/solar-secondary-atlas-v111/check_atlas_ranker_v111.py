"""Compose the reference-only diffuse atlas and published v111 ranker on a toy scene."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import types

import numpy as np

from solar_secondary_atlas_v111 import TerrainPatches, Visibility, build_atlas, project_atlas


RANKER = Path("reports/solar-low-frequency-ranking-v111/solar_low_frequency_ranking_v111.py")
RANKER_COMMIT = "375bf6afe0d0416c6e8ea0c584e37dcddef882f3"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def published_ranker(repository, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Require an explicit full published ranker commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main"],
                   cwd=repository, timeout=5, check=True)
    source = subprocess.check_output(["git", "show", f"{commit}:{RANKER}"], cwd=repository, timeout=5)
    module = types.ModuleType("published_solar_low_frequency_ranking_v111")
    exec(compile(source, f"{commit}:{RANKER}", "exec"), module.__dict__)
    return module.rerank_lunar_candidates, commit, sha(source)


def fixture(rerank):
    x, y = np.meshgrid(np.arange(-3.5, 4.), np.arange(-3.5, 4.))
    receivers = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    reflector = np.array([5.5, .3, 1.25])
    centers = np.vstack([receivers, reflector])
    normals = np.tile([0., 0., 1.], (len(centers), 1))
    normals[-1] = [-1., 0., 0.]
    mapped = np.ones(len(centers), dtype=bool)
    terrain = TerrainPatches(centers, normals, np.ones(len(centers)),
                             np.full(len(centers), .5), mapped, mapped)
    solar = np.full(len(centers), Visibility.BLOCKED)
    solar[-1] = Visibility.CLEAR
    mutual = np.full((len(centers), len(centers)), Visibility.BLOCKED)
    mutual[-1, :-1] = mutual[:-1, -1] = Visibility.CLEAR
    atlas = build_atlas(terrain, [-.6, 0., .8], solar, mutual)

    size = 128
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 4.
    intrinsics = np.array([[128., 0., 63.5], [0., 128., 63.5], [0., 0., 1.]])
    candidates = [pose.copy(), pose]
    candidates[0][0, 3] = .75
    rendered = [project_atlas(atlas, candidate, intrinsics, shape=(size, size))
                for candidate in candidates]

    # Generate an independent detector observation from the analytic one-bounce
    # formula on a horizontal unit lattice. The true fixture pose stays here.
    py, px = np.indices((size, size))
    surface_x = np.floor((px - 63.5) / 32) + .5
    surface_y = np.floor((63.5 - py) / 32) + .5
    dx, dy, dz = reflector[0] - surface_x, reflector[1] - surface_y, reflector[2]
    query = .003 + 1.8 * (.5**2 * .6 * dx * dz / (np.pi**2 * (dx**2 + dy**2 + dz**2)**2))

    result = rerank(query, np.stack([item.radiance for item in rendered]),
                    np.stack([item.valid for item in rendered]), [0, 1])
    if (result["primary"]["order"] != [1, 0] or result["primary"]["reason"] is not None
            or result["residual_after_plane"]["order"] != [1, 0]
            or result["common_pixels"] != size * size):
        raise ValueError("Composed atlas and ranker fixture did not recover the synthetic target")
    return dict(
        procedural_only=True, real_inputs=False, patch_count=len(centers), detector_shape=[size, size],
        candidate_world_x_m=[float(item[0, 3]) for item in candidates], original_order=[0, 1],
        supported_pixels=[int(item.valid.sum()) for item in rendered], ranking=result,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--ranker-commit", default=RANKER_COMMIT)
    args = parser.parse_args()
    rerank, commit, ranker_hash = published_ranker(args.repo_root.resolve(), args.ranker_commit)
    result = fixture(rerank)
    print(json.dumps(dict(**result, ranker_commit=commit, ranker_sha256=ranker_hash,
                          atlas_sha256=sha(Path(__file__).with_name("solar_secondary_atlas_v111.py").read_bytes()),
                          composition_source_sha256=sha(Path(__file__).read_bytes())), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
