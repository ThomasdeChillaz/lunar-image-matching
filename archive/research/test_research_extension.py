import copy
import io
import json
from pathlib import Path
import numpy as np
import pytest
from confidence import diagnostics, accepts, calibrate, independent_correctness
from matchers import Match
from solar_metadata import local_sun, parse_product_page
from remote_archive import HTTPRangeReader
from dataset import validate_manifest
from landmark_review import fingerprint
from import_review import apply_reviews


def test_geometry_diagnostics_find_inconsistent_reverse_and_clustered_support():
    points = np.array([[10, 10], [11, 10], [10, 11], [11, 11]], float)
    first = Match(np.array([[1., 0, 20], [0, 1, 0]]), reference_points=points, query_points=points+[20, 0])
    correct_reverse = Match(np.array([[1., 0, -20], [0, 1, 0]]))
    wrong_reverse = Match(np.array([[1., 0, 20], [0, 1, 0]]))
    good = diagnostics(first, correct_reverse, (512, 512))
    bad = diagnostics(first, wrong_reverse, (512, 512))
    assert good['cycle_p95_px'] == 0
    assert bad['cycle_p95_px'] == 40
    assert good['coverage_reference'] < .00001
    assert good['inlier_residual_p95_px'] == 0


def labelled_row(correct, translation, cycle=0.):
    return dict(split='dev', label_source='exact_geometry', correct=correct, estimated=True,
        reverse_estimated=True, method='sift', scale_min=1., scale_max=1., rotation_deg=0.,
        translation_fraction=translation, cycle_p95_px=cycle, coverage_reference=.3, coverage_query=.3)


def test_calibration_uses_dev_labels_and_preserves_rejection_tradeoff():
    rows = [labelled_row(True, .02), labelled_row(False, .5, 20.)]
    result = calibrate(rows)
    assert result['development_counts']['accepted_correct'] == 1
    assert result['development_counts']['accepted_incorrect'] == 0
    assert accepts(rows[0], result['policy']) and not accepts(rows[1], result['policy'])
    contaminated = copy.deepcopy(rows)
    contaminated[0]['split'] = 'test'
    with pytest.raises(ValueError, match='development'):
        calibrate(contaminated)
    contaminated = copy.deepcopy(rows)
    contaminated[0]['label_source'] = 'map_grid'
    with pytest.raises(ValueError, match='Independent'):
        calibrate(contaminated)


def test_uncertain_checkpoint_errors_do_not_become_false_labels():
    assert independent_correctness(1., .5, 3.) is True
    assert independent_correctness(5., .5, 3.) is False
    assert independent_correctness(3., 1., 3.) is None
    assert independent_correctness(None, 1., 3.) is None


def test_solar_direction_uses_local_geography_not_image_orientation():
    assert np.allclose(local_sun(0, 0, 0, 90), [90, 0], atol=1e-8)
    assert np.allclose(local_sun(0, 0, 45, 0), [0, 45], atol=1e-8)
    assert np.allclose(local_sun(0, 0, 0, -90), [270, 0], atol=1e-8)
    assert np.allclose(local_sun(0, 0, 0, 180)[1], -90, atol=1e-8)


def test_archive_page_parser_requires_identity_and_distinct_fields():
    labels = ['Sub solar latitude', 'Sub solar longitude', 'Incidence angle', 'North azimuth', 'Sub solar azimuth']
    html = '/M123RE' + ''.join(f'<tr><td>{label}</th><td>{value}</td></tr>' for value, label in enumerate(labels))
    parsed = parse_product_page(html, 'M123RE')
    assert parsed['Sub solar longitude'] == 1 and parsed['North azimuth'] == 3
    with pytest.raises(ValueError, match='identity'):
        parse_product_page(html, 'M456RE')


def test_range_reader_validates_server_response_and_handles_cross_block_reads(monkeypatch):
    content = b'abcdefghijk'
    class Response(io.BytesIO):
        status = 206
    def fetch(request, timeout):
        first, last = map(int, request.headers['Range'].split('=')[1].split('-'))
        response = Response(content[first:last+1])
        response.headers = {'Content-Range':f'bytes {first}-{last}/{len(content)}'}
        return response
    monkeypatch.setattr('urllib.request.urlopen', fetch)
    reader = HTTPRangeReader('https://example.org/data.zip', len(content), block_size=4)
    reader.seek(3)
    assert reader.read(7) == b'defghij'
    reader.seek(-3, 2)
    assert reader.read() == b'ijk'
    def invalid(request, timeout):
        response = Response(content)
        response.status = 200
        response.headers = {}
        return response
    monkeypatch.setattr('urllib.request.urlopen', invalid)
    with pytest.raises(ValueError, match='honor'):
        HTTPRangeReader('https://example.org/data.zip', len(content)).read(1)


def test_disk_range_cache_returns_only_requested_bytes_and_survives_new_reader(monkeypatch, tmp_path):
    content = b'abcdefghijklmnopqrstuvwxyz'
    class Response(io.BytesIO):
        status = 206
    calls = []
    def fetch(request, timeout):
        first, last = map(int, request.headers['Range'].split('=')[1].split('-'))
        calls.append((first, last))
        response = Response(content[first:last+1])
        response.headers = {'Content-Range':f'bytes {first}-{last}/{len(content)}'}
        return response
    monkeypatch.setattr('urllib.request.urlopen', fetch)
    reader = HTTPRangeReader('https://example.org/data.zip', len(content), block_size=8, cache_dir=tmp_path)
    reader.seek(3)
    assert reader.read(2) == b'de'
    other = HTTPRangeReader('https://example.org/data.zip', len(content), block_size=8, cache_dir=tmp_path)
    assert other.read(3) == b'abc'
    assert len(calls) == 1


def test_expanded_manifest_has_all_321_geographically_separated_pairs():
    manifest = json.loads(Path('expanded_solar_manifest.json').read_text())
    validate_manifest(manifest)
    assert len(manifest['assets']) == 42 and len(manifest['pairs']) == 321
    assert {r:sum(p['region_id'] == r for p in manifest['pairs']) for r in ['S1', 'S2', 'S3']} == dict(S1=45, S2=45, S3=231)
    assert all(a['solar_geometry']['source_sha256'] and np.isfinite(a['solar_azimuth_deg']) for a in manifest['assets'])
    assert all(p['reference_type'] == 'map_grid' for p in manifest['pairs'])


def manual_review(manifest, reviewer='Independent reviewer A'):
    points = [dict(landmark_id=f'fixture {i}', reference=[40+(i%5)*90, 50+(i//5)*350],
        query=[42+(i%5)*90, 53+(i//5)*350], reference_uncertainty_px=1., query_uncertainty_px=1.) for i in range(10)]
    return dict(schema_version=2, source_manifest_sha256=fingerprint(manifest),
        coordinate_system='512px common-grid pixel centers', reviewer=reviewer, reviewer_type='human',
        manual_blind_review_attested=True, reviewed_utc='2026-10-26T20:00:00Z',
        pairs={'S1_01':dict(points=points, unscorable=False)})


def test_reviews_require_source_binding_and_manual_attestation_without_mutating_original():
    manifest = json.loads(Path('lunar_manifest.json').read_text())
    review = manual_review(manifest)
    result = apply_reviews(manifest, review)
    assert manifest['pairs'][0]['reference_type'] == 'map_grid'
    assert result['pairs'][0]['reference_validation_status'] == 'reviewed'
    assert result['pairs'][0]['reference_uncertainty_px'] == pytest.approx(np.sqrt(2))
    review['reviewer_type'] = 'generated'
    with pytest.raises(ValueError, match='human'):
        apply_reviews(manifest, review)
    review = manual_review(manifest)
    review['source_manifest_sha256'] = '0'*64
    with pytest.raises(ValueError, match='exact source'):
        apply_reviews(manifest, review)


def test_second_review_disagreement_increases_reference_uncertainty():
    manifest = json.loads(Path('lunar_manifest.json').read_text())
    first = manual_review(manifest)
    second = manual_review(manifest, 'Independent reviewer B')
    second['pairs']['S1_01']['points'][0]['query'][0] += 5
    result = apply_reviews(manifest, first, second)
    assert result['pairs'][0]['reference_uncertainty_px'] == 5
    assert result['pairs'][0]['interreviewer_disagreement_px'][0] == 5
    with pytest.raises(ValueError, match='distinct'):
        apply_reviews(manifest, first, first)


def test_review_can_record_unscorable_pairs_without_inventing_landmarks():
    manifest = json.loads(Path('lunar_manifest.json').read_text())
    review = manual_review(manifest)
    review['pairs']['S1_01'] = dict(unscorable=True, reason='No ten stable shared landmarks visible', points=[])
    result = apply_reviews(manifest, review)
    assert result['pairs'][0]['reference_validation_status'] == 'unscorable'
    assert result['pairs'][0]['reference_type'] == 'map_grid'
