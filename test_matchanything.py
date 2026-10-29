import os
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get('RUN_MATCHANYTHING') != '1', reason='Pinned author source/weights and optional dependencies required')
def test_original_matchanything_native_and_enlarged_known_geometry():
    from matchanything_matcher import MatchAnythingMatcher
    from synthetic import texture, make_pair
    from geometry import evaluation_points, endpoint_error
    image = texture(42, 128)
    query, truth = make_pair(image, dict(angle=12, dx=6, dy=-4, gain=.7, gamma=1.2, noise=.005), 43)
    for factor in (1, 2, 4):
        matcher = MatchAnythingMatcher(dependency_dir=os.environ.get('ELOFTR_DEPENDENCY_DIR'), factor=factor)
        result = matcher(image, query)
        error = endpoint_error(result.matrix, truth, evaluation_points(image.shape, truth))
        assert error is not None and error < 1
        assert matcher(np.zeros_like(image), np.zeros_like(image)).reason == 'low_texture'
        assert matcher(image, image, np.zeros_like(image, bool), None).reason == 'no_valid_support'
        with pytest.raises(ValueError, match='128px'):
            matcher(image[:64, :64], image[:64, :64])
