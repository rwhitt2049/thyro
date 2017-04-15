from unittest import TestCase

import numpy as np
import numpy.testing as npt
from tiko.dataset import LabeledDataSet

from tests.testing_utils import (mock_feature_space, FEATURE_SPACE_RETURN,
                                 TEST_FEATURE_NAMES, TEST_LABELS, TEST_TARGETS)

MOCKED_FEATURE_SPACE, MFS_RETURN_VALUES = mock_feature_space()


class TestSimpleDataSet(TestCase):
    def setUp(self):
        self.dataset = LabeledDataSet(MOCKED_FEATURE_SPACE, TEST_FEATURE_NAMES, TEST_LABELS)

    def test_dataset_data(self):
        test = self.dataset.data(sparse=False)
        npt.assert_equal(test, list(FEATURE_SPACE_RETURN))

    def test_dataset_feature_names(self):
        test = self.dataset.feature_names
        self.assertEqual(test, TEST_FEATURE_NAMES)

    def test_dataset_labels(self):
        test = self.dataset.labels
        self.assertEqual(test, TEST_LABELS)

    def test_dataset_target_names(self):
        test = self.dataset.target_names
        npt.assert_equal(test, np.array(['test_label']))

    def test_targets(self):
        test = self.dataset.targets
        npt.assert_equal(test, TEST_TARGETS)

# TODO test various ways to specify label
# TODO test feature_domain
