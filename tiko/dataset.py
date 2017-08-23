import abc

import numpy as np
import pandas as pd

__all__ = ['extract_features', 'concat']


def gen_feature_space(features, segments):
    for segment in segments:
        yield tuple(feature(segment) for feature in features)


def extract_features(features, segments=None, labels=None):
    if isinstance(labels, str):
        labels = [labels] * len(segments)
    if segments is None:
        segments = [slice(None, None)]
    feature_names = [feature.name for feature in features]
    categorical_features = [feature.is_categorical for feature in features]
    feature_space = list(gen_feature_space(features, segments))

    index = pd.RangeIndex(len(feature_space))
    df = pd.DataFrame(feature_space, index=index, columns=feature_names)

    if labels:
        return LabeledDataSet(feature_names, labels, categorical_features, df)
    else:
        # return UnlabeledDataSet(feature_space, categorical_features)
        raise NotImplemented('currently only unlabeled data supported')


class DataSet(abc.ABC):
    def __init__(self, data, categorical_features):
        self.categorical_features = categorical_features
        self._data = data

    def __getattr__(self, item):
        return getattr(self._data, item)

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return repr(self._data)

    def __dir__(self):
        return super().__dir__() + dir(self._data)


class UnlabeledDataSet(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class LabeledDataSet(DataSet):
    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels


def concat(*datasets):
    # check all same type
    # check categorical features are the same
    # call pd.concat

    if datasets:
        categorical_features = datasets[0].categorical_features
    else:
        categorical_features = []

    df = pd.concat(datasets)
    return UnlabeledDataSet(df, categorical_features)
