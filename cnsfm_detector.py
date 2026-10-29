# SPDX-License-Identifier: GPL-3.0-only
"""Optional YOLOv9 inference adapter using the authors' crater checkpoint.

This entry point integrates GPL-3.0 YOLOv9; that optional integration retains
GPL-3.0 terms. Upstream code and pretrained weights are downloaded separately.
All input bytes and source revisions are checked before model deserialization.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import pickletools
import subprocess
import sys
import types
import zipfile
import cv2
import numpy as np
from dataset import load_pair, sha256, validate_manifest
from landmark_review import fingerprint

YOLO_REVISION = '5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff'
WEIGHT_SHA256 = '498a647d5c80d7a5a9c4f151c264c98f762db3a0f1574ff05f5ae8041bbfd860'
ALLOWED_GLOBALS = {
    '__builtin__ set', 'collections OrderedDict', 'torch FloatStorage', 'torch HalfStorage',
    'torch LongStorage', 'torch Size', 'torch._utils _rebuild_parameter', 'torch._utils _rebuild_tensor_v2',
    *('models.common '+name for name in ['ADown', 'CBFuse', 'CBLinear', 'Concat', 'Conv', 'DFL',
        'RepConvN', 'RepNBottleneck', 'RepNCSP', 'RepNCSPELAN4', 'SP', 'SPPELAN', 'Silence']),
    'models.yolo DetectionModel', 'models.yolo DualDDetect',
    'torch.nn.modules.activation SiLU', 'torch.nn.modules.batchnorm BatchNorm2d',
    'torch.nn.modules.container ModuleList', 'torch.nn.modules.container Sequential',
    'torch.nn.modules.conv Conv2d', 'torch.nn.modules.linear Identity', 'torch.nn.modules.pooling MaxPool2d',
    'torch.nn.modules.upsampling Upsample'}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module+' '+name not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError('Unexpected checkpoint global: '+module+' '+name)
        return super().find_class(module, name)


def load_detector(source, weights):
    source, weights = Path(source).resolve(), Path(weights).resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != YOLO_REVISION or sha256(weights) != WEIGHT_SHA256:
        raise ValueError('Detector source/checkpoint differs from pinned inputs')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', '--quiet', 'HEAD'], check=True)
    with zipfile.ZipFile(weights) as archive:
        data = archive.read(next(n for n in archive.namelist() if n.endswith('/data.pkl')))
        for op, argument, position in pickletools.genops(data):
            if op.name == 'STACK_GLOBAL' or (op.name == 'GLOBAL' and argument not in ALLOWED_GLOBALS):
                raise ValueError('Checkpoint contains unexpected serialized classes')
    os.environ['YOLOv5_AUTOINSTALL'] = 'false'
    sys.path.insert(0, str(source))
    import torch
    from utils.general import non_max_suppression
    # Restricted loading of this exact pinned checkpoint; no arbitrary globals.
    module = types.ModuleType('restricted_cnsfm_checkpoint')
    module.Unpickler = RestrictedUnpickler
    module.load, module.loads = pickle.load, pickle.loads
    checkpoint = torch.load(weights, map_location='cpu', pickle_module=module)
    model = checkpoint['model'].float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.set_num_threads(1)
    return model, torch, non_max_suppression


def detect(image, valid, model, torch, nms):
    # Official dual-head inference output, fixed public detect_dual defaults.
    resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).float()[None, None].repeat(1, 3, 1, 1)/255
    with torch.inference_mode():
        prediction = model(tensor)[0][1]
        boxes = nms(prediction, conf_thres=.25, iou_thres=.45, max_det=1000)[0].numpy()
    entries = []
    for x1, y1, x2, y2, score, category in boxes:
        x, y = (x1+x2)/2*.8, (y1+y2)/2*.8
        if not 0 <= x <= 511 or not 0 <= y <= 511 or not valid[int(round(y)), int(round(x))]:
            continue
        entries.append(dict(x=float(x), y=float(y), width=float((x2-x1)*.8), height=float((y2-y1)*.8),
            score=float(score), category=int(category), detection_id=len(entries)+1))
    return entries


def run(manifest, source, weights, output):
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    assets = validate_manifest(manifest, '.')
    model, torch, nms = load_detector(source, weights)
    if len(model.names) != 1:
        raise ValueError('Expected single crater class; inspect released model before using multiclass outputs')
    output_assets = []
    for asset in manifest['assets']:
        pair = next(p for p in manifest['pairs'] if p['region_id'] == asset['region_id'])
        pair = dict(pair, reference=asset['asset_id'], query=asset['asset_id'])
        image, _, valid, _, grid = load_pair(pair, assets, '.')
        image[~valid] = 0
        detections = detect(image, valid, model, torch, nms)
        output_assets.append(dict(product_id=asset['product_id'], region_id=asset['region_id'],
            common_bounds_m=pair['common_bounds_m'], image_bytes_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            mask_bytes_sha256=hashlib.sha256(valid.tobytes()).hexdigest(), detections=detections))
        print(asset['product_id'], len(detections), 'craters', flush=True)
    result = dict(schema_version=1, source_manifest_sha256=fingerprint(manifest), image_size=512,
        yolo_revision=YOLO_REVISION, weight_sha256=WEIGHT_SHA256,
        source_sha256=sha256(__file__), class_names=model.names,
        parameters=dict(confidence=.25, nms_iou=.45, inference_size=640, max_detections=1000),
        protocol='Redetect craters on the same masked 512px map-grid images as all baseline methods. Single whole-image YOLO pass with fixed public defaults; author native patch tiling is not reproduced.',
        environment={name:importlib.metadata.version(name) for name in ['torch', 'torchvision', 'numpy', 'opencv-python', 'pandas', 'PyYAML', 'IPython', 'scipy', 'seaborn']},
        warning='All detections are algorithm outputs. They are never independent landmark references.', assets=output_assets)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='expanded_solar_manifest.json')
    p.add_argument('--yolo-source', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--output', default='cnsfm_detector_outputs.json')
    args = p.parse_args()
    run(json.loads(Path(args.manifest).read_text()), args.yolo_source, args.weights, args.output)
