"""Bounded actual safe-weight MOONSt3R native forward/pose software control."""
import argparse
import json
from pathlib import Path
import resource
import signal
from time import perf_counter
import platform

from dataset import sha256
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import controls
from native_moonst3r_v19 import NativeMOONSt3R

SOURCES=['native_moonst3r_v19.py','control_native_moonst3r_v19.py','moonst3r_model_lock_v19.json',
    'native_baseline_pose_v7.py','localization_robust_pose_v4.py','localization_pose_refit_v3.py',
    'lunar_geometry_v5.py','lunar_geometry_v2.py','localization_diagnostics_v3.py','contrastive_terrain.py',
    'synthetic.py','loftr_matcher.py','eloftr_matcher.py','dataset.py','lunargeo_dataset.py']


def run(output,dependency_dir,max_seconds=600):
    root=Path(output)
    if root.exists():raise FileExistsError('Preserve all baseline control attempts')
    root.mkdir(parents=True);started=perf_counter()
    def timeout(*args):raise TimeoutError('MOONSt3R actual pretrained control exceeded declared CPU wall cap')
    old=signal.signal(signal.SIGALRM,timeout);signal.alarm(max_seconds)
    write_new(root/'protocol.json',dict(scope='Actual pretrained native512 software gate; no lunar inference or accuracy claim.',
        cpu_threads=2,max_seconds=max_seconds,calibration_or_test_pixels_loaded=0,
        source_sha256={p:sha256(p) for p in SOURCES}))
    try:
        import cv2
        import torch
        torch.set_num_threads(2);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
        matcher=NativeMOONSt3R(dependency_dir=dependency_dir,threads=2);write_new(root/'provenance.json',matcher.provenance)
        result=controls(matcher);write_new(root/'controls.json',result)
        value=dict(passed=result['passed'],actual_native_pretrained_forward_attempted=True,controls=result,
            elapsed_seconds=perf_counter()-started,peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)),
            source_sha256=sha256(__file__),scope='Actual known-transform and invalid-input software gate, not lunar accuracy, independent renderer transfer or isolated model latency.')
        write_new(root/'summary.json',value)
        if not result['passed']:raise ValueError('Actual MOONSt3R software control failed; no lunar comparison authorized by this gate')
        print(json.dumps({k:v for k,v in value.items() if k!='controls'},indent=2));return value
    except Exception as exc:
        write_new(root/'FAILURE.json',dict(passed=False,error_type=type(exc).__name__,error=str(exc),elapsed_seconds=perf_counter()-started,
            completed_control_saved=(root/'controls.json').exists(),scope='Incomplete or failed actual baseline gate. Do not infer lunar performance.'))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--dependency-dir')
    parser.add_argument('--max-seconds',type=int,default=600);args=parser.parse_args()
    if not 60<=args.max_seconds<=600:raise ValueError('Bounded60–600sec actual control required')
    run(args.output,args.dependency_dir,args.max_seconds)
