"""A bounded, model-free inventory of the frozen model-selection image allowlist.

Prepare/publish first. Run only in a free serial CPU slot. No reservation or
refund is performed here. Existing model/data contracts remain unchanged.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time

CAP_SECONDS = 120
CLEANUP_RESERVE_SECONDS = 5
WORK_SECONDS = CAP_SECONDS-CLEANUP_RESERVE_SECONDS
EXPECTED_CASES = 384
EXPECTED_POSITIVES = 192
EXPECTED_NEGATIVES = 192
EXPECTED_IMAGES = 224
LIGHT_PAIRS = [(0,0),(16,16),(0,17),(16,1),(8,17),(8,1)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def utc(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Explicit UTC launch/deadline timestamps required')
    return result.astimezone(timezone.utc)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def safe_path(repo, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or not p.parts or any(x in ('..', '.') for x in p.parts):
        raise ValueError('Strict repository-relative input paths required')
    candidate = repo.joinpath(*p.parts)
    if not candidate.resolve().is_relative_to(repo) or any(repo.joinpath(*p.parts[:i]).is_symlink() for i in range(1,len(p.parts)+1)):
        raise ValueError('Symlink/escaping input path forbidden')
    return candidate


def manifest_scope(args):
    repo = Path(args.repo_root).resolve()
    path = safe_path(repo, args.manifest)
    if sha(path) != args.manifest_sha256:
        raise ValueError('Frozen selection manifest changed')
    m = json.loads(path.read_text()); cases = m['cases']
    if not m['complete'] or m['split'] != 'model_selection' or m['calibration_or_final_access'] is not False:
        raise ValueError('Complete model-selection-only manifest required')
    if (len(cases) != EXPECTED_CASES or m['positive_cases'] != EXPECTED_POSITIVES or m['negative_cases'] != EXPECTED_NEGATIVES
            or len(m['terrain_groups']) != 8 or m['generated_selection_unit_count'] != 32
            or m['eligible_selection_unit_count'] != 32 or sorted(map(tuple,m['light_pairs'])) != sorted(LIGHT_PAIRS)):
        raise ValueError('This frozen inventory requires all original384 cases/32 cameras/eight groups')
    units = {r['unit_index']: r for r in m['generated_selection_units']}
    eligible = set(m['eligible_unit_indices']); pins = {}; uses = {}
    if len(eligible) != 32 or len(units) != 32 or len({c['case_id'] for c in cases}) != EXPECTED_CASES:
        raise ValueError('Original complete case/camera denominator required')
    for c in cases:
        if c['split'] != 'model_selection' or type(c['positive']) is not bool or (c['light_a'],c['light_b']) not in LIGHT_PAIRS:
            raise ValueError('No training/calibration/final or unlisted light pair allowed')
        for role, unit_key, component_key, light_key in [('reference','unit_index','component','light_a'),('query','query_unit_index','query_component','light_b')]:
            index = c[unit_key]; u = units[index]
            if index not in eligible or not u['eligible'] or u['split'] != 'model_selection' or u['component'] != c[component_key]:
                raise ValueError('Both endpoints must belong to the32 selection-camera inventory')
            relative = c[role]
            expected = f"{u['source_output']}/{role}/light-{c[light_key]}.png"
            if relative != expected:
                raise ValueError('Image is not an exact manifest-approved selection role/light path')
            safe_path(repo,relative)
            digest = c['input_sha256'][relative]
            if relative in pins and pins[relative] != digest:
                raise ValueError('Conflicting image hashes')
            pins[relative] = digest
            uses.setdefault(relative,[]).append(dict(case_id=c['case_id'],role=role,component=c[component_key],unit_index=index,light=c[light_key]))
    if len(pins) != EXPECTED_IMAGES or sum(c['positive'] for c in cases) != EXPECTED_POSITIVES:
        raise ValueError('Every frozen image/case must remain in the inventory')
    return repo, m, cases, pins, uses


def group_rows(cases, key):
    groups = {}
    for row in cases: groups.setdefault(str(key(row)),[]).append(row)
    result = {}
    for name, rows in sorted(groups.items()):
        pos = [r for r in rows if r['positive']]
        rejected = sum(r['current_guard_triggered'] for r in pos)
        result[name] = dict(all_cases=len(rows),positive_cases=len(pos),negative_cases=len(rows)-len(pos),
            positive_guard_triggered=rejected,positive_guard_passed=len(pos)-rejected,
            software_guard_only_positive_coverage_ceiling=None if not pos else (len(pos)-rejected)/len(pos),
            reference_trigger_cases=sum(r['reference_guard_triggered'] for r in rows),
            query_trigger_cases=sum(r['query_guard_triggered'] for r in rows),
            both_trigger_cases=sum(r['reference_guard_triggered'] and r['query_guard_triggered'] for r in rows))
    return result


def worker(args):
    # The controller starts this fresh process with all thread limits set.
    import numpy as np
    import PIL
    from PIL import Image
    if os.environ.get('SOLAR_INVENTORY_PARENT_PID') != str(os.getppid()):
        raise ValueError('Worker requires its startup-inclusive external watchdog parent')
    if sha(Path(__file__)) != args.source_sha256:
        raise ValueError('Worker source bytes changed')
    started = datetime.now(timezone.utc); tick = time.monotonic()
    repo, m, cases, pins, uses = manifest_scope(args)
    images = {}; output = Path(args.output)
    for index, relative in enumerate(sorted(pins)):
        path = safe_path(repo,relative)
        if sha(path) != pins[relative]: raise ValueError('Image bytes changed: '+relative)
        with Image.open(path) as image:
            source_mode = image.mode
            # EXACTLY the current evaluation loader's conversion; no resize.
            pixels = np.asarray(image.convert('L')).copy()
        if pixels.shape != (512,512) or pixels.dtype != np.uint8:
            raise ValueError('Native512 uint8 source content required')
        histogram = np.bincount(pixels.reshape(-1),minlength=256)
        probabilities = histogram[histogram>0].astype(np.float64)/pixels.size
        std = float(pixels.std())  # NumPy population std, as frozen guard.
        images[relative] = dict(source_sha256=pins[relative],source_mode=source_mode,decoded_dtype=str(pixels.dtype),
            decoded_shape=list(pixels.shape),decode_policy="PIL.convert('L'), identical to current loader; no resize",
            population_std=std,current_guard_std_less_than_1=bool(std<1),exact_constant=bool(pixels.min()==pixels.max()),
            intensity_min=int(pixels.min()),intensity_max=int(pixels.max()),unique_value_count=int(np.count_nonzero(histogram)),
            empirical_histogram_entropy_bits=float(-np.sum(probabilities*np.log2(probabilities))),
            histogram_counts=histogram.tolist(),uses=uses[relative])
        print(json.dumps(dict(images_completed=index+1,total_images=len(pins),path=relative)),flush=True)
    rows=[]
    for c in cases:
        row={k:c[k]for k in ['case_id','positive','unit_index','query_unit_index','component','query_component','light_a','light_b','reference','query']}
        row.update(reference_guard_triggered=images[c['reference']]['current_guard_std_less_than_1'],
                   query_guard_triggered=images[c['query']]['current_guard_std_less_than_1'])
        row['current_guard_triggered'] = row['reference_guard_triggered'] or row['query_guard_triggered']
        row['trigger_pattern'] = ('both' if row['reference_guard_triggered'] and row['query_guard_triggered'] else
            'reference_only' if row['reference_guard_triggered'] else 'query_only' if row['query_guard_triggered'] else 'neither')
        rows.append(row)
    positive=[r for r in rows if r['positive']]; failures=sum(r['current_guard_triggered'] for r in positive)
    report=dict(complete=True,started_utc=started.isoformat(),completed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(Path(__file__)),manifest=args.manifest,manifest_sha256=args.manifest_sha256,cpu_threads=2,
        software=dict(numpy=np.__version__,pillow=PIL.__version__,python=sys.version.split()[0]),
        all_cases=len(rows),positive_cases=len(positive),negative_cases=len(rows)-len(positive),unique_images=len(images),
        positive_cases_blocked_by_current_guard=failures,positive_cases_not_blocked_by_current_guard=len(positive)-failures,
        current_software_guard_only_positive_coverage_ceiling=(len(positive)-failures)/len(positive),
        all_cases_remain_eligible=True,model_forwards=0,optimizer_updates=0,new_training_targets=0,
        calibration_or_final_access=False,training_images_opened=0,scoring_geometry_or_labels_opened=0,
        images=images,cases=rows,
        breakdowns=dict(reference_region=group_rows(rows,lambda r:r['component']),
            query_region=group_rows(rows,lambda r:r['query_component']),
            light_pair=group_rows(rows,lambda r:f"{r['light_a']}->{r['light_b']}"),
            reference_region_and_light_pair=group_rows(rows,lambda r:f"{r['component']}:{r['light_a']}->{r['light_b']}"),
            reference_light=group_rows(rows,lambda r:r['light_a']),query_light=group_rows(rows,lambda r:r['light_b']),
            unit=group_rows(rows,lambda r:r['unit_index']),changed_light=group_rows(rows,lambda r:r['light_a']!=r['light_b'])),
        scope='Descriptive image statistics and the unchanged std<1 SOFTWARE guard only. Not physical observability, solar/camera visibility labels, information-theory bounds or model accuracy.',
        limitations=['Low variance does not imply no recoverable information. Exact constancy concerns these quantized pixels only.',
            'Histogram entropy ignores spatial organization and is not a matching/localization information bound.',
            'Brightness never supplies physical shadow or occlusion labels.',
            'All192 positive cases stay in the denominator; this inventory excludes no data and selects no model/checkpoint.',
            'Repeated views/lights/images are not independent terrain trials.'],worker_calendar_seconds=time.monotonic()-tick)
    write(output/'inventory.json',report)


def controller(args):
    started = datetime.now(timezone.utc); tick=time.monotonic()
    if sha(Path(__file__)) != args.source_sha256: raise ValueError('Prepared source bytes changed')
    latest=utc(args.latest_start_utc); original_deadline=utc(args.absolute_deadline_utc)
    deadline=min(started+timedelta(seconds=CAP_SECONDS),original_deadline)
    work_deadline=min(started+timedelta(seconds=WORK_SECONDS),original_deadline-timedelta(seconds=CLEANUP_RESERVE_SECONDS))
    if started>latest or started>=work_deadline: raise TimeoutError('Declared inventory dispatch lease/work deadline expired; no reset')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    pending=[]; child=None; complete=False; failure=None
    def interrupted(signum,frame): pending.append(signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def work_remaining():
        return min(WORK_SECONDS-(time.monotonic()-tick),(work_deadline-datetime.now(timezone.utc)).total_seconds())
    def total_remaining():
        return max(0.,min(CAP_SECONDS-(time.monotonic()-tick),(deadline-datetime.now(timezone.utc)).total_seconds()))
    try:
        # The child imports NumPy/Pillow, hashes/decodes images and writes its
        # report within115s, retaining5s cleanup INSIDE the total120s allowance.
        argv=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            VECLIB_MAXIMUM_THREADS='2',NUMEXPR_NUM_THREADS='2',SOLAR_INVENTORY_PARENT_PID=str(os.getpid()))
        with (output/'worker.log').open('x') as log:
            if pending or work_remaining()<=0:
                raise TimeoutError('No child launch after the declared work deadline')
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
            while child.poll() is None:
                if pending:raise InterruptedError('Stop requested; preserve partial inventory artifacts')
                remaining=work_remaining()
                if remaining<=0:raise TimeoutError('Inventory work deadline reached; cleanup remains inside total120s')
                time.sleep(min(.1,remaining))
        if pending or work_remaining()<=0:raise TimeoutError('Inventory completed after its work deadline or was interrupted')
        if child.returncode:raise subprocess.CalledProcessError(child.returncode,argv)
        report=json.loads((output/'inventory.json').read_text())
        if (not report['complete'] or report['all_cases']!=EXPECTED_CASES or report['unique_images']!=EXPECTED_IMAGES
                or report['source_sha256']!=args.source_sha256 or report['manifest_sha256']!=args.manifest_sha256
                or sha(Path(__file__))!=args.source_sha256):
            raise ValueError('Incomplete inventory cannot be sealed')
        if pending or work_remaining()<=0:
            raise TimeoutError('Inventory validation completed after its work deadline')
        write(output/'COMPLETE.json',dict(inventory_sha256=sha(output/'inventory.json'),source_sha256=args.source_sha256,
            manifest_sha256=args.manifest_sha256,scope='Image-statistic inventory only; not model/scientific success'))
        if pending or work_remaining()<=0:
            raise TimeoutError('Inventory seal completed after its work deadline; failed receipt overrides any partial seal')
        complete=True
    except BaseException as exc:
        failure=dict(error_type=type(exc).__name__,error=str(exc))
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=min(2.,total_remaining()))
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=min(2.,total_remaining()))
        raise
    finally:
        write(output/'watchdog.json',dict(complete=complete,source_sha256=args.source_sha256,manifest_sha256=args.manifest_sha256,
            started_utc=started.isoformat(),finished_utc=datetime.now(timezone.utc).isoformat(),
            original_latest_start_utc=latest.isoformat(),original_absolute_deadline_utc=original_deadline.isoformat(),
            effective_deadline_utc=deadline.isoformat(),work_deadline_utc=work_deadline.isoformat(),
            cap_seconds=CAP_SECONDS,work_cap_seconds=WORK_SECONDS,calendar_seconds=time.monotonic()-tick,
            child_pid=None if child is None else child.pid,child_returncode=None if child is None else child.returncode,
            failure=failure,cpu_threads=2,automatic_retry=False,refunds=False,
            budget_policy='Parent-owned admission/charges; this utility reserves or refunds nothing.',
            cleanup_inside_cap_seconds=CLEANUP_RESERVE_SECONDS,cleanup_outside_cap_seconds=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo-root','manifest','manifest-sha256','source-sha256','output','latest-start-utc','absolute-deadline-utc']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:worker(args)
    else:controller(args)


if __name__=='__main__':main()
