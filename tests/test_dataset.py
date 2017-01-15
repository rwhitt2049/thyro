from unittest import TestCase

import numpy as np
import numpy.testing as npt
from thyro.dataset import DataSet

from tests.testing_utils import mock_feature_space, FEATURE_SPACE_RETURN, TEST_FEATURE_NAMES, TEST_LABELS, TEST_TARGETS

MOCKED_FEATURE_SPACE, MFS_RETURN_VALUES = mock_feature_space()


class TestDataSet(TestCase):
    def setUp(self):
        self.dataset = DataSet(MOCKED_FEATURE_SPACE, TEST_FEATURE_NAMES,  TEST_LABELS)

    def test_dataset_data(self):
        test = self.dataset.data
        control = np.array(FEATURE_SPACE_RETURN)

        npt.assert_equal(test, control)

    def test_dataset_feature_names(self):
        test = self.dataset.feature_names

        npt.assert_equal(test, TEST_FEATURE_NAMES)

    def test_dataset_labels(self):
        test = self.dataset.labels
        npt.assert_equal(test, TEST_LABELS)

    def test_targets(self):
        test = self.dataset.targets
        npt.assert_equal(test, TEST_TARGETS)
