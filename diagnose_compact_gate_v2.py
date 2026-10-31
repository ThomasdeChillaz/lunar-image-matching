"""Post-study diagnosis of the unchanged old gate; no inference or fitting."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from compact_research import controlled_inputs
from confidence import diagnostics
from dataset import sha256
from lunargeo_dataset import write_new
from matchers import Match
from research_quality_gate import accepts,features
from structural_refinement import AlignmentObjective


def diagnose(output):
    source='runs/compact-v2-original-test/results.json'
    rows=json.loads(Path(source).read_text())
    manifest=json.loads(Path('reports/compact_dataset_v2.json').read_text())
    tiles={t['tile_id']:t for t in manifest['tiles']};cache={}
    policy_path='reports/reserve_protocol_v1.json'
    policy=json.loads(Path(policy_path).read_text())['policy']
    counters=Counter();by_light={};details=[]
    for row in rows:
        if row['negative']:continue
        a,b,ma,mb,_=controlled_inputs(row,tiles,cache)
        forward=Match(matrix=None if row['transform'] is None else np.array(row['transform']),
            reference_points=np.array(row['reference_points']),query_points=np.array(row['query_points']),
            support=row['support'])
        reverse=Match(matrix=None if row['reverse_transform'] is None else np.array(row['reverse_transform']))
        quality=AlignmentObjective(a,b,ma,mb).quality(forward.matrix)
        evidence=dict(estimated=forward.matrix is not None,
            selection_quality={} if quality is None else dict(quality,agreement_fraction=0.,agreeing_other_families=[]),
            **diagnostics(forward,reverse,a.shape))
        assert accepts(evidence,policy)==row['original_frozen_accepted']
        values=features(evidence)
        checks=dict(estimated=evidence['estimated'],reverse_estimated=evidence['reverse_estimated'],
            finite=bool(np.isfinite(values).all()),global_correlation=values[0]>=policy['min_global'],
            patch_correlation=values[1]>=policy['min_patch'],lit_support=values[2]>=policy['min_lit'],
            geometric_support=values[3]>=policy['min_support'],family_agreement=values[4]>=policy['min_agreement'],
            reverse_cycle=values[5]<=policy['max_cycle'])
        failed=[name for name,passed in checks.items() if not passed]
        assert bool(not failed)==row['original_frozen_accepted']
        counters.update(failed)
        if failed==['global_correlation']:counters['global_correlation_only']+=1
        light=f"{row['light_a']}-{row['light_b']}"
        counts=by_light.setdefault(light,dict(positives=0,correct_raw=0,old_accepted=0,new_accepted=0))
        counts['positives']+=1;counts['correct_raw']+=bool(row['correct'])
        counts['old_accepted']+=bool(row['original_frozen_accepted']);counts['new_accepted']+=bool(row['accepted'])
        details.append(dict(pair_id=row['pair_id'],correct_raw=row['correct'],failed_checks=failed))
    result=dict(scope='Post-study explanation on consumed controlled test outcomes. No new model inference, threshold fit or rule change.',
        source_results_sha256=sha256(source),old_policy_sha256=sha256(policy_path),
        old_policy=policy,positives=len(details),failed_check_counts=dict(counters),
        note='Failed-check counts can overlap. The global-only count isolates cases failing no other gate condition.',
        by_light_pair=by_light,cases=details)
    write_new(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    diagnose(parser.parse_args().output)
