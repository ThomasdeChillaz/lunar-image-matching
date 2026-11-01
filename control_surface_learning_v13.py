"""Actual selected-weight analytic controls for the surface-trained cohort."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from coarse_pipeline_v5 import infer as initialize
from compact_lunar import CompactMatcher
from dataset import sha256
from lunargeo_dataset import write_new
from native_refinement_v7 import NativeRefiner
from surface_learning_v13 import refine_surface
from synthetic import texture


def control(root, output):
    root = Path(root); torch.set_num_threads(2); cv2.setNumThreads(1)
    manifest = json.loads(Path('reports/localization_training_data_v5.json').read_text())
    coarse = CompactMatcher(manifest['coarse_model']); image = texture(733, 512)
    query = np.rot90(image, 1).copy(); blank = np.zeros_like(image); depth = np.full(image.shape, 10.)
    k = np.array([[400., 0, 255.5], [0, 400., 255.5], [0, 0, 1.]])
    initial, _ = initialize(coarse, coarse.model, image, query, depth, k, np.eye(4), k)
    no_pose, attempts = initialize(coarse, coarse.model, blank, blank, depth, k, np.eye(4), k)
    if initial is None or no_pose is not None:
        raise ValueError('Fixed coarse analytic initializer failed')
    invalid = attempts[0].copy(); invalid['pose'] = None
    expected_rotation = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
    results = []
    for folder in sorted(root.glob('*/training.json')):
        meta = json.loads(folder.read_text()); path = folder.parent/'selected.pt'
        model = NativeRefiner(meta['variant'])
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        selected = refine_surface(model, image, query, depth, k, np.eye(4), k, initial)
        rejected = refine_surface(model, blank, blank, depth, k, np.eye(4), k, invalid)
        pose = selected['pose']; rotation = float(np.linalg.norm(pose[:3, :3]-expected_rotation))
        translation = float(np.linalg.norm(pose[:3, 3]))
        passed = rotation < .01 and translation < .05 and rejected['pose'] is None
        results.append(dict(experiment=meta['experiment'], seed=meta['seed'], selected_step=meta['selected_step'],
            checkpoint_sha256=sha256(path), passed=bool(passed), rotation_matrix_error=rotation,
            translation_error_m=translation, blank_pose_null=rejected['pose'] is None,
            estimated_cam2world=pose.tolist()))
    if len(results) != (3 if json.loads((root/'protocol.json').read_text())['pilot'] else 5):
        raise ValueError('Missing selected-model control')
    result = dict(passed=all(r['passed'] for r in results), runs=results, image_seed=733,
        source_sha256=sha256(__file__), scope='Actual native512px quarter-turn/blank software controls; not lunar validation.')
    write_new(output, result); print(json.dumps(result, indent=2))
    if not result['passed']:
        raise ValueError('Selected model analytic control failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args(); control(args.run, args.output)
