"""Completion lineage plus the unchanged independent v59 geometry audit.

The base geometry audit is retained separately. Its scores, thresholds and
sampling are unchanged; the new checks only establish original/new lineage.
No matcher inference or fitting occurs in this program.
"""
import argparse,json,hashlib
from pathlib import Path
from north_camera_completion_v62 import verify_completion_lineage
from verify_north_camera_v59 import audit

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(root,schedule,contract,base_output,output):
    base_output,output=Path(base_output),Path(output)
    if base_output.exists()or output.exists():raise FileExistsError('Preserve every independent audit')
    lineage=verify_completion_lineage(root,contract)
    base=audit(root,schedule,base_output)
    result=dict(base,completion_lineage=lineage,completion_contract_sha256=sha(contract),
        base_geometry_audit_path=str(base_output),base_geometry_audit_sha256=sha(base_output),
        completion_audit_source_sha256=sha(__file__),original_render_deadline_met=False,
        scope='Missing-only completion lineage plus unchanged independent full geometry/pose audit. Original1800second rendering deadline remains failed. Four already-used pilot supports; no fitting, fresh geography, model gain or orbital-accuracy claim.')
    with output.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k]for k in ('passed','oracle_success','geometrically_occluded_samples','occlusion_training_supported','completion_lineage')},indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','schedule','contract','base-output','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.root,a.schedule,a.contract,a.base_output,a.output)
