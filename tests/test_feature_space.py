from unittest import TestCase

import numpy as np
import numpy.testing as npt

from tests.testing_utils import (TEST_SEGMENTS, FEATURE_SPACE_RETURN,
                                 MAX_FEATURE, MIN_FEATURE)
from thyro.feature_space import FeatureSpace


class TestFeatureSpace(TestCase):
    def setUp(self):
        self.feature_space = FeatureSpace(TEST_SEGMENTS, MIN_FEATURE, MAX_FEATURE)

    def test_len(self):
        self.assertEqual(len(self.feature_space), 3)

    def test_shape(self):
        self.assertEqual(self.feature_space.shape, (3, 2))

    def test_output(self):
        test = np.array([feature_vector for feature_vector in self.feature_space])
        control = FEATURE_SPACE_RETURN

        npt.assert_equal(test, control)

    def test_output_shape(self):
        test = np.array([feature_vector for feature_vector in self.feature_space])
        self.assertEqual(test.shape, self.feature_space.shape)

    def test_segment_type_error(self):
        with self.assertRaises(TypeError):
            FeatureSpace('a', MIN_FEATURE, MAX_FEATURE)
