"""Create a blank, zoomable review packet bound to exact source images and grids."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import cv2
from dataset import load_pair, validate_manifest


def fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()


def create(manifest, root, output):
    assets = validate_manifest(manifest, root)
    pairs = []
    for pair in manifest['pairs']:
        a, b, ma, mb, grid = load_pair(pair, assets, root)
        images = []
        for image in (a, b):
            ok, data = cv2.imencode('.png', image)
            if not ok:
                raise ValueError('PNG encoding failed')
            images.append('data:image/png;base64,'+base64.b64encode(data).decode())
        pairs.append(dict(pair_id=pair['pair_id'], reference_product=pair['reference'],
            query_product=pair['query'], images=images))
    payload = dict(source_manifest_sha256=fingerprint(manifest), pairs=pairs)
    template = Path(__file__).with_name('landmark_review_template.html').read_text()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('__REVIEW_DATA__', json.dumps(payload)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='lunar_manifest.json')
    p.add_argument('--root', default='.')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    create(json.loads(Path(args.manifest).read_text()), args.root, args.output)
