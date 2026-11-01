"""Bounded actual pretrained native-image gate; no lunar/calibration/test pixels."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import resource
import shutil
import signal
from time import perf_counter

import cv2

from dataset import sha256
from lunargeo_dataset import write_new
from native_baseline_pose_v7 import controls
from native_matchanything_v17 import NativeMatchAnything

SOURCES=['control_native_matchanything_v17.py','native_matchanything_v17.py',
    'matchanything_matcher.py','eloftr_matcher.py','loftr_matcher.py',
    'matchanything_model_lock.json','native_baseline_pose_v7.py','synthetic.py',
    'localization_diagnostics_v3.py','localization_robust_pose_v4.py',
    'localization_pose_refit_v3.py','lunar_geometry_v5.py','lunar_geometry_v2.py']


def run(output,dependency_dir,profile,threads,max_seconds):
    root=Path(output)
    if root.exists():raise FileExistsError('Preserve completed/failed pretrained controls')
    root.mkdir(parents=True);(root/'source').mkdir();cv2.setNumThreads(1)
    protocol=dict(scope='Actual pretrained native512px software/cost controls, not lunar localization performance.',
        profile=profile,cpu_threads=threads,max_seconds=max_seconds,source_sha256={p:sha256(p) for p in SOURCES},
        lunar_images_loaded=0,calibration_or_test_pixels_loaded=0,
        timing='Concurrent local cost pilot; not isolated inference latency or an efficiency comparison.',
        started_utc=datetime.now(timezone.utc).isoformat())
    write_new(root/'protocol.json',protocol)
    for source in SOURCES:shutil.copy2(source,root/'source'/source)
    def timeout(*args):raise TimeoutError('Pretrained control exceeded the declared wall budget')
    previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(max_seconds);started=perf_counter()
    try:
        matcher=NativeMatchAnything(dependency_dir=dependency_dir,profile=profile,threads=threads)
        write_new(root/'provenance.json',matcher.provenance)
        result=controls(matcher);write_new(root/'controls.json',result)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        summary=dict(passed=result['passed'],scope=protocol['scope'],controls=result,
            elapsed_seconds=perf_counter()-started,peak_process_rss_bytes=int(rss if platform.system()=='Darwin' else rss*1024),
            protocol_sha256=sha256(root/'protocol.json'),controls_sha256=sha256(root/'controls.json'),
            source_sha256=sha256(__file__),lunar_images_loaded=0,calibration_or_test_pixels_loaded=0)
        for source,digest in protocol['source_sha256'].items():
            if sha256(source)!=digest:raise ValueError('Executed control source changed')
        write_new(root/'summary.json',summary);print(json.dumps(summary,indent=2));return summary
    except Exception as exc:
        write_new(root/'FAILED.json',dict(type=type(exc).__name__,message=str(exc),elapsed_seconds=perf_counter()-started))
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--dependency-dir');parser.add_argument('--profile',choices=['native512','training_size832'],default='native512')
    parser.add_argument('--threads',type=int,choices=[1,2],default=2);parser.add_argument('--max-seconds',type=int,default=600);args=parser.parse_args()
    if not 30<=args.max_seconds<=3600:raise ValueError('Bounded pretrained control budget required')
    run(args.output,args.dependency_dir,args.profile,args.threads,args.max_seconds)
