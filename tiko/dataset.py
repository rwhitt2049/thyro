from abc import ABCMeta, abstractmethod
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.preprocessing import LabelEncoder

# TODO - lighten up dependencies, make both scipy and sklearn optionl
# Probably will need to do some try/except monkey patchery?

from tiko.feature_space import FeatureSpace
from tiko.features import create_feature

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
        # TODO - push nominal_features default specification up one level of abstraction to extract_features
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
def extract_features(config, data, segments=None, user_features=None, default_statistics=None, labels=None, **feature_factory):
    """
    
    config structure
    
    list of dicts
    dicts with keys
        signal_name: str
        display_name: str
        statistics: list of str
        is_nominal: bool
        apply_default_statistics: bool
    
    Args:
        config: 
        data: 
        segments: 
        user_features: 
        default_statistics: 
        labels: 

    Returns:

    """
    import warnings; warnings.warn('THIS IS EXPERIMENT AND SUBJECT TO CHANGE SIGNIFICANTLY')
    # TODO rename to extract_features, extract_featureset... ?

    if not hasattr(data, '__getitem__'):
        raise TypeError('Your data must have a defined __getitem__ method')

    if user_features is None:
        user_features = []

    if segments is None:
        segments = [slice(None, None)]

    features = user_features
    nominal_features = [feature.is_nominal for feature in user_features]
    feature_names = [feature.name for feature in user_features]

    for settings in config:
        signal_name = settings['signal_name']
        signal = data[signal_name]
        statistics = settings.get('statistics', [])
        apply_default_statistics = settings.get('apply_default_statistics', True)
        is_nominal = settings.get('is_nominal', False)
        display_name = settings.get('display_name', signal_name)

        if apply_default_statistics:
            statistics += list(set(default_statistics) - set(statistics))
        elif not statistics:
            raise ValueError('No statistics defined for %s' % signal_name)

        for statistic in statistics:
            if statistic in feature_factory:
                feature = create_feature(signal, feature_factory[statistic])
            else:
                feature = create_feature(signal, statistic)

            features.append(feature)
            feature_names.append('_'.join([display_name, statistic]))
            nominal_features.append(is_nominal)

    feature_space = FeatureSpace(*user_features, segments=segments)

    feature_space_array = np.array(list(feature_space))

    if labels is None:
        dataset = UnlabeledDataSet(feature_space_array, feature_names,
                                   nominal_features=nominal_features)
    else:
        dataset = LabeledDataSet(feature_space_array, feature_names, labels=labels,
                                 nominal_features=nominal_features)

    return dataset
