import abc

import pandas as pd

__all__ = ['extract_features']


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
        return UnlabeledDataSet(feature_space, categorical_features)


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

    def __bool__(self):
        return bool(len(self._data))


class UnlabeledDataSet(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)


class LabeledDataSet(DataSet):
    def __init__(self, labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels
