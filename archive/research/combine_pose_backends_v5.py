"""Simple SIFT-first engineering baseline; no correctness labels in selection."""
import argparse
import gzip
import json
from pathlib import Path
from dataset import sha256
from lunargeo_dataset import write_new


def choose(sift,compact):
    if sift is not None and sift['pose'] is not None:return sift,'sift_pose_exists'
    if compact is not None and compact['pose'] is not None:return compact,'compact_pose_fallback'
    return None,'both_pose_estimators_failed'


def run(output):
    root=Path(output)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);source=Path('runs/localization-v5-axial-correction/results.json.gz')
    protocol=dict(scope='Post-correction development-only engineering fallback; no new learned model or calibrated acceptance.',
        source_sha256=sha256(__file__),input_sha256=sha256(source),
        choice='Use native SIFT pose whenever its reference-geometry estimator succeeds; otherwise use the four-orientation compact proposal.',
        warning='Selected from development evidence. No negative-pair/final validation or1% risk claim.',
        future_comparison='Evaluate this fixed fallback equally for frozen and newly trained coarse candidates; do not claim a learning gain against a weaker baseline.')
    write_new(root/'protocol.json',protocol)
    with gzip.open(source,'rt') as f:original=json.load(f)
    sift={r['pair_id']:r for r in original if r['method']=='sift_native' and r['solver']=='ap3p_reclassify'}
    cnn={r['pair_id']:r for r in original if r['method']=='quarter_turn_bank' and r['selected']};rows=[]
    for key,a in sift.items():
        selected,reason=choose(a,cnn.get(key))
        rows.append(dict(pair_id=key,tile_id=a['tile_id'],component=a['component'],light_a=a['light_a'],light_b=a['light_b'],
            selected_backend=None if selected is None else selected['method'],reason=reason,
            pose=None if selected is None else selected['pose'],pose_correct=bool(selected and selected['pose_correct']),
            strict_pose_correct=bool(selected and selected['strict_pose_correct'])))
    counts=dict(cases=len(rows),correct=sum(r['pose_correct'] for r in rows),strict=sum(r['strict_pose_correct'] for r in rows),
        estimated=sum(r['pose'] is not None for r in rows),wrong=sum(r['pose'] is not None and not r['pose_correct'] for r in rows))
    result=dict(scope=protocol['scope'],results=counts,correct_raw_pose_rate=counts['correct']/len(rows),
        wrong_fraction_of_estimated=counts['wrong']/max(counts['estimated'],1),
        note='No pair-level acceptance policy: these are raw estimates, not correctly accepted coverage.',rows=rows)
    write_new(root/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args().output)
