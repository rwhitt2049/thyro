from abc import ABCMeta, abstractmethod
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.preprocessing import LabelEncoder

from thyro.feature_space import FeatureSpace
from thyro.features import create_feature

__all__ = ['LabeledDataSet']


class BaseDataSet(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, feature_space):
        self.feature_space = feature_space

    @lru_cache()
    def data(self, sparse=True):
        if sparse:
            return lil_matrix(self.feature_space)
        else:
            return np.array(self.feature_space)

    @abstractmethod
    def as_dataframe(self):
        pass


class UnlabeledDataSet(BaseDataSet):
    def __init__(self, feature_space, feature_names, nominal_features=None):
        super().__init__(feature_space)
        self.feature_names = feature_names
        self.nominal_features = nominal_features

    def as_dataframe(self):
        index = pd.RangeIndex(len(self.feature_space))
        return pd.DataFrame(self.data(sparse=False), index, self.feature_names)


class LabeledDataSet(BaseDataSet):
    def __init__(self, feature_space, feature_names, labels, nominal_features=None):
        super().__init__(feature_space)
        self.feature_names = feature_names
        self.nominal_features = nominal_features
        self._labels = labels
        self._encoder = LabelEncoder()
        self.targets = self._encoder.fit_transform(self.labels)

    @property
    def labels(self):
        # TODO Simplify, consequences of EAFP?

        if isinstance(self._labels, str):
            return [self._labels] * len(self.feature_space)
        elif isinstance(self._labels, list):
            if len(self._labels) != len(self.feature_space):
                raise ValueError('If labels is specified as a sequence, it\'s length'
                                 'must equal the number of feature vectors in feature_space')
            else:
                return self._labels
        else:
            raise TypeError('Labels must be sequence of strings or a string.')

    @property
    def target_names(self):
        return self._encoder.classes_

    def as_dataframe(self, include_targets=True, include_labels=True):
        index = pd.RangeIndex(len(self.feature_space))

        df = pd.DataFrame(self.data(sparse=False), index, self.feature_names)

        if include_targets:
            df.insert(0, 'targets', self.targets)

        if include_labels:
            df.insert(0, 'labels', self.labels)

        return df


def dedupe(seq):
    value = []
    uniques = set()
    set_add = uniques.add
    for item in seq:
        if item not in uniques:
            value.append(item)
        set_add(item)

    return value


# EXPERIMENTAL FEATURE
def get_dataset(config, data, segments=None, user_features=None, default_statistics=None, labels=None):
    import warnings; warnings.warn('THIS IS EXPERIMENT AND SUBJECT TO CHANGE SIGNIFICANTLY')
    # TODO rename to extract_features, extract_featureset... ?
    if user_features is None:
        user_features = []

    nominal_features = [feature.is_nominal for feature in user_features]
    feature_names = [feature.name for feature in user_features]

    for settings in config:
        statistics = settings.get('statistics', default_statistics)
        statistics.extend(default_statistics)
        uniq_stats = dedupe(statistics)

        _data = data[settings['signal_name']]

        for stat in uniq_stats:
            feature_name = '_'.join(
                [settings.get('display_name', settings['signal_name']), stat])
            user_features.append(create_feature(_data, stat))
            nominal_features.append(settings.get('is_nominal', False))
            feature_names.append(feature_name)

    feature_space = FeatureSpace(*user_features, segments=segments)

    feature_space_array = np.array(list(feature_space))

    if labels is None:
        dataset = UnlabeledDataSet(feature_space_array, feature_names,
                                   nominal_features=nominal_features)
    else:
        dataset = LabeledDataSet(feature_space_array, feature_names, labels=labels,
                                 nominal_features=nominal_features)

    return dataset
