"""Fixed train/selection diagnostic of the failed native no-match head."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import sha256
from localization_refiner_v3 import decode_peak
from lunargeo_dataset import write_new
from native_refinement_v7 import NativeRefiner, joint_logits
from train_balanced_localization_v3 import load_bundle


def rank_auc(scores, positive):
    scores = np.asarray(scores, float); positive = np.asarray(positive, bool)
    if scores.shape != positive.shape or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('Finite paired ranking inputs required')
    a, b = scores[positive], scores[~positive]
    if not len(a) or not len(b):
        return None
    return float(((a[:, None] > b).sum()+.5*(a[:, None] == b).sum())/(len(a)*len(b)))


def run(folder, output):
    folder = Path(folder)
    complete = json.loads((folder/'COMPLETE.json').read_text())
    meta = json.loads((folder/'training.json').read_text())
    if (meta['variant'] != 'dustbin' or sha256(folder/'final.pt') != complete['final_sha256']
            or sha256(folder/'training.json') != complete['training_sha256']):
        raise ValueError('Verified completed failed dustbin checkpoint required')
    for p, h in meta['source_sha256'].items():
        if sha256(p) != h:
            raise ValueError('Training implementation changed')
    torch.set_num_threads(2); cv2.setNumThreads(1)
    model = NativeRefiner('dustbin')
    model.load_state_dict(torch.load(folder/'final.pt', map_location='cpu', weights_only=True)); model.eval()
    manifest, tiles, data = load_bundle('reports/localization_training_data_v5.json')
    rng = np.random.default_rng(20261109); rows = []
    for c in manifest['cases']:
        if c['split'] not in ('train', 'model_selection'):
            raise ValueError('Only train/selection pixels permitted')
        z = data[c['tile_id']]; key = c['key']; p, q = z[f'p{key}'], z[f'q{key}']
        target = z[f'target{key}']-np.rint(q)+16
        valid = z[f'visible{key}'] & ((target >= 0) & (target <= 32)).all(1)
        for positive in (True, False):
            eligible = np.flatnonzero(valid == positive)
            if not len(eligible):
                continue
            indices = rng.choice(eligible, min(4, len(eligible)), replace=False)
            rp, qp, xy = [], [], []
            for i in indices:
                rc, qc = np.rint(p[i]).astype(int), np.rint(q[i]).astype(int)
                rp.append(cv2.getRectSubPix(z['images'][2*c['light_a']], (27, 27), tuple(map(float, rc))).astype(np.float32)/255)
                qp.append(cv2.getRectSubPix(z['images'][2*c['light_b']+1], (57, 57), tuple(map(float, qc))).astype(np.float32)/255)
                xy.append(p[i]-rc+13)
            with torch.inference_mode():
                local, null = model(torch.from_numpy(np.stack(rp))[:, None], torch.from_numpy(np.stack(qp))[:, None],
                                    torch.tensor(np.array(xy), dtype=torch.float32))
                no_match = null.sigmoid().numpy()
                reject = (joint_logits(local, null).argmax(1) == 1089).numpy()
                peak = decode_peak(local).numpy()+16
            for j, i in enumerate(indices):
                error = float(np.linalg.norm(peak[j]-target[i])) if positive else None
                rows.append(dict(pair_id=c['pair_id'], split=c['split'], component=c['component'],
                    proposal_index=int(i), window_has_visible_correspondence=positive,
                    decoded_error_px=error, precise_match=bool(positive and error <= 2),
                    no_match_probability=float(no_match[j]), joint_reject=bool(reject[j])))
    stats = {}
    for split in ('train', 'model_selection'):
        part = [r for r in rows if r['split'] == split]
        scores = [1-r['no_match_probability'] for r in part]
        stats[split] = dict(samples=len(part), components=len({r['component'] for r in part}),
            window_validity_auc=rank_auc(scores, [r['window_has_visible_correspondence'] for r in part]),
            decoded_two_pixel_reliability_auc=rank_auc(scores, [r['precise_match'] for r in part]),
            joint_rejected=sum(r['joint_reject'] for r in part),
            precise_before_rejection=sum(r['precise_match'] for r in part),
            groups={str(v): dict(samples=len(group), median_no_match_probability=float(np.median([r['no_match_probability'] for r in group])))
                    for v in (True, False) if (group := [r for r in part if r['window_has_visible_correspondence'] == v])})
    result = dict(scope='Unaugmented fixed train/selection coarse-proposal diagnostic of a failed final checkpoint; not pose acceptance or independent test accuracy.',
        checkpoint_sha256=complete['final_sha256'], source_sha256=sha256(__file__),
        manifest_sha256=sha256('reports/localization_training_data_v5.json'), seed=20261109,
        sampling='At most4 positives and4 negatives per frozen case; no augmentation or threshold fitting. Repeated samples are dependent.',
        results=stats, rows=rows, calibration_or_test_pixels_loaded=0)
    write_new(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.model, a.output)
