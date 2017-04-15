from collections import namedtuple
from unittest.mock import MagicMock, PropertyMock

import numpy as np

from tiko.features import create_feature

ReturnsFS = namedtuple('ReturnsFS', 'len shape segments features iter')

np.random.seed(10)

TEST_SEGMENTS = [slice(2, 5), slice(6, 9), slice(12, 15)]

TEST_SIGNAL = np.random.randint(1, 10, 15)
MIN_FEATURE = create_feature(TEST_SIGNAL, 'min')
MAX_FEATURE = create_feature(TEST_SIGNAL, 'max')

FEATURE_SPACE_RETURN = np.array([(MIN_FEATURE(seg), MAX_FEATURE(seg)) for seg in TEST_SEGMENTS])

TEST_FEATURE_NAMES = ['test_signal_min', 'test_signal_max']
TEST_LABELS = ['test_label'] * len(TEST_SEGMENTS)
TEST_TARGETS = np.array([0] * 3)


def mock_feature_space():
    mocked_fs = MagicMock()

    mocked_len = 3
    mocked_fs.__len__.return_value = mocked_len

    mocked_shape = (3, 2)
    shape = PropertyMock(return_value=mocked_shape)
    type(mocked_fs).shape = shape

    segments = PropertyMock(return_value=TEST_SEGMENTS)
    type(mocked_fs).segments = segments

    _features = (MIN_FEATURE, MAX_FEATURE)
    features = PropertyMock(return_value=_features)
    type(mocked_fs).features = features


    mocked_fs.__iter__.return_value = FEATURE_SPACE_RETURN

    return_values = ReturnsFS(mocked_len, mocked_shape, TEST_SEGMENTS, _features, FEATURE_SPACE_RETURN)

    return mocked_fs, return_values
