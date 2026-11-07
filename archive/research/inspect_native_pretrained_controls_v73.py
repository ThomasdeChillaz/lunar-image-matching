"""Read-only saved-control replay; never constructs or forwards a network."""
import argparse,json
from pathlib import Path
from native_pretrained_adapters_v73 import native_profile,validate_native_profile,sha


def run(repository_root,output):
    root=Path(repository_root).resolve();rows=[]
    for method,profile,run_root in(
        ('matchanything','native512','localization-v17-matchanything-native-control'),
        ('moonst3r','common_native512_full_field_of_view','localization-v19-moonst3r-native-control')):
        path=root/'runs'/run_root;item=native_profile(method,method,profile,path,root)
        try:
            gate=validate_native_profile(item)
            status=dict(gate_eligible=True,control_replay=gate['control_replay'],evidence_sha256=gate['evidence_sha256'])
        except (ValueError,KeyError,OSError)as error:
            status=dict(gate_eligible=False,reason=type(error).__name__+': '+str(error))
        rows.append(dict(method=method,author_profile=profile,existing_actual_control_run=str(path),**status))
    value=dict(scope='Read-only current-source, saved-array and common-pose replay of preserved ACTUAL pretrained software controls. No new neural forward or lunar measurement.',
        rows=rows,runtime_assets_loaded=False,new_actual_controls_executed=False,lunar_images_loaded=0,full_dataset_geometry_gate_checked=False,comparison_admitted=False,lunar_performance_claim=False,
        inspector_sha256=sha(__file__),adapter_sha256=sha(Path(__file__).with_name('native_pretrained_adapters_v73.py')))
    with Path(output).open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.repository_root,a.output),indent=2))
