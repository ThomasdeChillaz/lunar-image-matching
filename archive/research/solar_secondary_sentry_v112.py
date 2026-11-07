"""Fresh v112 blind candidate and common reference-coverage sentry."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from time import monotonic

sys.path.insert(0, str(Path.cwd()))
import numpy as np

from render_solar_clean_radiance_v106 import BLENDER_BUILD, BLENDER_SHA, digest, read, write
from run_solar_secondary_v112 import JOURNAL, LEDGER, ROOT, checked, profile, roster, utc

DIR = ROOT / 'blind-sentry'
REQUEST = DIR / 'reference-only-request.json'
INPUT_SEAL = DIR / 'PREPARED.json'


def asset(path):
    return dict(path=str(path), sha256=digest(path))


def open_asset(item):
    return checked(item['path'], item['sha256'])


def request():
    spec = read(open_asset(read(INPUT_SEAL)['request']))
    if set(spec) != {'reference', 'slots', 'query_intrinsics', 'profile'}:
        raise ValueError('Reference, opaque observations and camera intrinsics only')
    if set(spec['reference']) != {'mesh', 'geometry', 'sun', 'latitude_deg', 'maximum_height_m'}:
        raise ValueError('Reference-only metadata schema')
    if any(set(slot) != {'id', 'reference_exr', 'query_exr'} for slot in spec['slots']):
        raise ValueError('Opaque input slots cannot carry query labels')
    if [r['id'] for r in spec['slots']] != ['r00', 'r01', 'r02', 'r03']:
        raise ValueError('Exactly four predeclared opaque sentry observations')
    return spec


def prepare(plan):
    for path, checksum in plan['predecessor_sha256'].items():
        checked(path, checksum)
    design = roster()
    capture = read(ROOT/'sentry'/'COMPLETE.json')
    lookup = {(r['view_id'], r['replicate']): r for r in capture['renders']}
    expected = {(r['view_id'], r['replicate']) for r in design['early_coverage_stop']['acquire_first_view_seed_pairs']}
    if len(lookup) != 6 or set(lookup) != expected:
        raise ValueError('Sealed six-render inventory differs')
    refs = design['endpoints'][0]
    audit, = [r for r in read(ROOT/'gate.json')['endpoints'] if r['endpoint'] == refs['endpoint']]
    if not audit['eligible'] or audit['map_mesh_sha256'] != refs['map_mesh_sha256']:
        raise ValueError('Reference mesh must match its independent geometry audit')
    camera = [r for r in read(ROOT/'geometry'/'COMPLETE.json')['cameras']
              if r['endpoint'] == refs['endpoint'] and r['role'] == 'reference']
    reference, = camera
    intrinsics = refs['camera_pair']['reference']['intrinsics']
    if any(e['camera_pair']['query']['intrinsics'] != intrinsics for e in design['endpoints']):
        raise ValueError('Sentry requires common declared camera intrinsics')
    inputs = DIR / 'inputs'
    inputs.mkdir(parents=True)

    def copy_input(item, target):
        with open_asset(item).open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        if digest(target) != item['sha256']:
            raise ValueError('Opaque copy changed input bytes')
        return asset(target)

    observations = {}
    slots, assignments = [], []
    ordered = {c['case_id']: c for c in design['cases']}
    for case_id in design['early_coverage_stop']['fixed_case_ids']:
        case = ordered[case_id]
        for replicate in (0, 1):
            slot = {'id': f'r{len(slots):02d}'}
            for role in ('reference', 'query'):
                key = case[f'{role}_view_id'], replicate
                if key not in observations:
                    observations[key] = copy_input(lookup[key], inputs/f'o{len(observations):02d}.exr')
                slot[f'{role}_exr'] = observations[key]
            slots.append(slot)
            assignments.append(dict(id=slot['id'], case_id=case_id, replicate=replicate))
    if len(observations) != 6:
        raise ValueError('Opaque handoff must preserve all six render identities')
    write(REQUEST, dict(reference=dict(mesh=dict(path=refs['map_mesh'], sha256=refs['map_mesh_sha256']),
                                       geometry=copy_input(reference, inputs/'reference.npz'),
                                       sun=refs['lighting'][0]['sun_enu'], latitude_deg=refs['site']['centre_latitude_deg'],
                                       maximum_height_m=audit['scene_maximum_height_m']),
                        query_intrinsics=intrinsics, slots=slots, profile=profile()))
    # Only the downstream seal may open the case mapping, after both blind arms are written.
    index = DIR/'scoring-index.json'
    write(index, assignments)
    write(INPUT_SEAL, dict(request=asset(REQUEST), scoring_index=asset(index), observation_files=6,
                           query_labels_in_request=False))


def renderer_identity():
    import bpy
    if (tuple(bpy.app.version[:3]) != (4, 5, 9) or bpy.app.build_hash.decode() != BLENDER_BUILD
            or digest(bpy.app.binary_path) != BLENDER_SHA):
        raise ValueError('Exact v109 Blender build required')


def reference(deadline):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from solar_map_v63 import grid_envelope, reference_solar_table
    from solar_sun_search_v68 import sun_search_catalogue

    start = monotonic()
    renderer_identity()
    item = request()['reference']
    with np.load(open_asset(item['mesh']), allow_pickle=False) as mesh:
        vertices, faces = mesh['vertices'], mesh['faces']
    with np.load(open_asset(item['geometry']), allow_pickle=False) as geometry:
        depth, intrinsics, pose = (geometry[k] for k in ('axial_depth', 'intrinsics', 'cam2world'))
    bvh = BVHTree.FromPolygons(vertices.tolist(), faces.tolist(), all_triangles=True, epsilon=0.)
    if item['maximum_height_m'] != float(vertices[:, 2].max()):
        raise ValueError('Reference scene maximum differs from independent audit')
    envelope = grid_envelope(vertices, faces, scene_maximum_height=item['maximum_height_m'])
    rays = 0

    def raycast(origin, direction, limit):
        nonlocal rays
        if rays % 4096 == 0:
            ensure_time(deadline)
        rays += 1
        point, normal, _, _ = bvh.ray_cast(Vector(origin), Vector(direction), float(limit))
        return None if point is None else (np.asarray(point, float), np.asarray(normal, float))

    catalogue = sun_search_catalogue(item['latitude_deg'])
    suns = np.array([x['sun_enu'] for x in catalogue])
    axis = 49.5 + 16*np.rint(np.linspace(0, 25, 16)).astype(int)
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    table, reasons = reference_solar_table(np.column_stack([xx.ravel(), yy.ravel()]),
                                          intrinsics, pose, suns, envelope, raycast)
    path = DIR / 'reference-table.npz'
    save_npz(path, depth=depth, intrinsics=intrinsics, cam2world=pose, reference_sun=item['sun'], sun_vectors=suns,
             sun_parameters=np.array([[x['hour_angle_deg'], x['subsolar_latitude_deg']] for x in catalogue]),
             coarse_sun_mask=np.array([x['coarse'] for x in catalogue], bool),
             solar_materialized_mask=np.ones(len(suns), bool), **table)
    write(DIR/'reference-table.json', dict(archive=asset(path), rays=rays, columns=len(suns),
                                         reason_counts=reasons, elapsed_seconds=monotonic()-start))


def save_npz(path, **arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def normalized_u8(image):
    luminance = np.maximum(np.asarray(image, float), 0)
    rank = (199*luminance.size + 199)//200 - 1
    white = float(np.partition(luminance.ravel(), rank)[rank])
    if not np.isfinite(luminance).all():
        raise ValueError('Finite luminance required')
    if white <= 0 or luminance.min() == luminance.max():
        return np.zeros(luminance.shape, np.uint8)
    return np.floor(np.minimum(luminance/white, 1.)*255. + .5).astype(np.uint8)


def blind(deadline):
    import os
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    import cv2
    import torch
    from localize_terrain import jsonable, load_matcher
    from replay_raw_acquisition_v109 import decode, luminance
    from solar_blind_beams_v112 import localize_with_blind_beams
    from solar_frontend_v63 import ReferenceGeometry

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec = request()
    prof = spec['profile']
    cfg = prof['configuration']
    checkpoint = checked(prof['checkpoint'], cfg['checkpoint_sha256'])
    matcher = load_matcher(checkpoint, cfg, allow_untrained=True)
    table = ReferenceGeometry.load(open_asset(read(DIR/'reference-table.json')['archive']))
    array = {}
    output = DIR/'banks'
    output.mkdir()
    for slot in spec['slots']:
        ensure_time(deadline)
        for role in ('reference_exr', 'query_exr'):
            entry = slot[role]
            if entry['path'] not in array:
                raw = luminance(decode(open_asset(entry).read_bytes(), exr_size=512))
                array[entry['path']] = (normalized_u8(raw), raw)
        reference_u8 = array[slot['reference_exr']['path']][0]
        query_u8, query_raw = array[slot['query_exr']['path']]
        tick = monotonic()
        ordinary, bank = localize_with_blind_beams(matcher, reference_u8, query_u8, table,
                                                   np.asarray(spec['query_intrinsics']), response=cfg['response'],
                                                   active=cfg['active'], budget=cfg['budget'], policy=cfg['policy'])
        item = dict(id=slot['id'], blind_bank=bank.to_dict(), compact_final_pose=jsonable(ordinary.get('pose')),
                    compact_final_reason=ordinary.get('reason'), elapsed_seconds=monotonic()-tick)
        raw_path = output/f'{slot["id"]}-luminance.npz'
        save_npz(raw_path, query_luminance=query_raw)
        item['query_luminance'] = asset(raw_path)
        write(output/f'{slot["id"]}.json', item)
    if len(array) != 6:
        raise ValueError('Decode exactly six independent optical inputs once')


def project(deadline):
    renderer_identity()
    import sys
    for folder in ('solar-secondary-mesh-v112', 'solar-secondary-atlas-v111'):
        sys.path.insert(0, str(Path.cwd()/'reports'/folder))
    from solar_secondary_atlas_v111 import project_atlas
    from solar_secondary_mesh_v112 import FiniteScene, mesh_array_sha256, prepare_blender_reference

    spec = request()
    with np.load(open_asset(spec['reference']['mesh']), allow_pickle=False) as mesh:
        vertices, faces = mesh['vertices'], mesh['faces']
    start = monotonic()
    maximum = spec['reference']['maximum_height_m']
    if maximum != float(vertices[:, 2].max()):
        raise ValueError('Atlas finite-scene maximum differs from independent audit')
    scene = FiniteScene(mesh_array_sha256(vertices, faces), maximum)
    binding = prepare_blender_reference(vertices, faces, tiles_per_axis=16, diffuse_albedo=.5,
                                        epsilon_m=.001, max_ray_distance_m=50000., finite_scene=scene)
    output = DIR/'predictions'
    output.mkdir()
    atlases = {}
    for slot in spec['slots']:
        ensure_time(deadline)
        name = slot['id']
        bank_path = DIR/'banks'/f'{name}.json'
        bank = read(bank_path)['blind_bank']
        candidates = bank['candidates']
        reason = None
        rendered = []
        started = monotonic()
        if not candidates:
            reason = 'no_eligible_blind_beams'
        elif any(row['sun_reference_enu'][2] <= 0 for row in candidates):
            reason = 'below_horizon_candidate_sun'
        else:
            for row in candidates:
                ensure_time(deadline)
                sun = tuple(row['sun_reference_enu'])
                if sun not in atlases:
                    atlases[sun] = binding.bind_sun(sun)
                solar = atlases[sun]
                intrinsics = spec['query_intrinsics']
                both = project_atlas(solar.atlas, row['pose'], intrinsics, channel='combined')
                direct = project_atlas(solar.atlas, row['pose'], intrinsics, channel='direct')
                rendered.append((both, direct, solar))
        path = output/f'{name}.npz'
        stats = []
        if rendered:
            save_npz(path, combined=np.stack([b.radiance for b, _, _ in rendered]),
                     direct=np.stack([d.radiance for _, d, _ in rendered]),
                     reference_valid=np.stack([b.valid & d.valid for b, d, _ in rendered]))
            for combined, direct, solar in rendered:
                atlas = solar.atlas
                stats.append(dict(combined_pixels=int(combined.valid.sum()), direct_pixels=int(direct.valid.sum()),
                                  solar_ray_states=solar.reason_counts, solar_rays=solar.traced_rays,
                                  missing_pairs=int(atlas.missing_pairs.sum()),
                                  excessive_form_factors=int(atlas.excessive_form_factor.sum()),
                                  direct_known=int(atlas.direct_known.sum()), secondary_known=int(atlas.secondary_known.sum())))
        write(output/f'{name}.json', dict(bank=asset(bank_path), prediction=asset(path) if rendered else None,
                                         unavailable=reason, candidates=stats, elapsed_seconds=monotonic()-started))
    write(output/'reference.json', dict(atlases=len(atlases), patch_count=len(binding.terrain.centers),
                                       mutual_rays=binding.mutual_rays,
                                       mutual_states=np.bincount(binding.mutual_states.ravel(), minlength=3).tolist(),
                                       total_wall_seconds=monotonic()-start))


def seal():
    sys.path.insert(0, str(Path.cwd()/'reports'/'solar-low-frequency-ranking-v111'))
    from solar_low_frequency_ranking_v111 import rerank_lunar_candidates

    output = DIR/'ranks'
    output.mkdir()
    ranked = []
    for slot in request()['slots']:
        name = slot['id']
        path = DIR/'predictions'/f'{name}.json'
        prediction = read(path)
        bank_record = read(open_asset(prediction['bank']))
        candidates = bank_record['blind_bank']['candidates']
        distinct = len({tuple(np.ravel(row['pose'])) for row in candidates})
        row = dict(id=name, prediction=asset(path), bank=prediction['bank'], distinct_poses=distinct,
                   original_order=list(range(len(candidates))), coverage_pass=False)
        if prediction['unavailable']:
            row.update(reason=prediction['unavailable'], rankings=None)
        else:
            with np.load(open_asset(prediction['prediction']), allow_pickle=False) as z:
                valid, direct, combined = (z[key] for key in ('reference_valid', 'direct', 'combined'))
            blank = np.zeros((512, 512))
            support_only = rerank_lunar_candidates(blank, combined, valid, row['original_order'])
            covered = len(support_only['active_quadrants']) >= 3
            row['support'] = {k: support_only[k] for k in ('common_pixels', 'retained_blocks', 'retained_pixels',
                                                         'quadrant_blocks', 'active_quadrants')}
            row['coverage_pass'] = distinct >= 2 and covered
            row['reason'] = ('candidate_starved' if distinct < 2 else
                             None if covered else 'insufficient_shared_quadrant_support')
            with np.load(open_asset(bank_record['query_luminance']), allow_pickle=False) as z:
                query = z['query_luminance']
            row['rankings'] = dict(combined=rerank_lunar_candidates(query, combined, valid, row['original_order']),
                                   direct=rerank_lunar_candidates(query, direct, valid, row['original_order']))
        path = output/f'{name}.json'
        write(path, row)
        ranked.append(asset(path))
    # Seal primary ranks and reference-only support before opening case identity.
    write(output/'COMPLETE.json', dict(ranks=ranked, scoring_truth_opened=False))
    assign = read(open_asset(read(INPUT_SEAL)['scoring_index']))
    design = roster()
    target = design['early_coverage_stop']['positive_gate_case_id']
    evaluated = {read(open_asset(row))['id']: read(open_asset(row)) for row in ranked}
    gates = [dict(**identity, coverage_pass=evaluated[identity['id']]['coverage_pass'],
                  reason=evaluated[identity['id']]['reason']) for identity in assign if identity['case_id'] == target]
    if len(gates) != 2:
        raise ValueError('Both positive seeds required for coverage decision')
    write(DIR/'COMPLETE.json', dict(complete=True, sentry_pass=all(g['coverage_pass'] for g in gates),
                                   positive_gates=gates, rank_seal=asset(output/'COMPLETE.json'),
                                   calibration_or_final_access=False, query_pose_scored=False,
                                   gate_uses_only_blind_candidate_diversity_and_reference_masks=True))


def ensure_time(deadline):
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError('Original v112 phase deadline elapsed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('prepare', 'reference', 'blind', 'project', 'seal'))
    parser.add_argument('--dispatch', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--publication-commit', required=True)
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    from datetime import timedelta
    from render_solar_raw_acquisition_v109 import publication
    from run_solar_secondary_v112 import dispatch_path
    if args.root != str(ROOT) or args.dispatch != dispatch_path('sentry'):
        raise ValueError('Require the fixed v112 sentry root and dispatch')
    deadline = utc(args.deadline_utc)
    ensure_time(deadline)
    plan = read(args.dispatch)
    publication(args.dispatch, plan, args.publication_commit)
    journal = read(JOURNAL)
    own, = [r for r in journal['reservations'] if r['job_id'] == digest(args.dispatch) + '/secondary-v112']
    if (journal['ledger_sha256'] != plan['metadata_sha256'][LEDGER] or own['seconds'] != 1200
            or own['category'] != 'baseline' or own['refund_on_failure'] is not False
            or deadline != utc(own['reserved_utc']) + timedelta(seconds=1200)):
        raise ValueError('Require the original one-use sentry reservation')
    if args.step == 'prepare':
        prepare(plan)
    elif args.step == 'seal':
        seal()
    else:
        {'reference': reference, 'blind': blind, 'project': project}[args.step](deadline)


if __name__ == '__main__':
    main()
