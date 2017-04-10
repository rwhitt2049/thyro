import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from thyro.feature_space import FeatureSpace
from thyro.features import create_feature

__all__ = ['LabeledDataSet', 'get_dataset']


# think about what needs to happen if you concat a bunch of datasets.
# check if feature_names are all the same
# Append labels end to end in order
# append feature_domain end to end in order
# Feature space.... this is currently assumed to be a FeatureSpace generator,
# make a genertor of generators

# feature_space = (feature_fector for feature_space in feature_spaces
#                                     for feature_vector in feature_space)


class LabeledDataSet:
    def __init__(self, feature_space, feature_names, labels, nominal_features=None):
        self.feature_space = feature_space
        self.feature_names = feature_names
        self.nominal_features = nominal_features
        self._labels = labels
        self._encoder = LabelEncoder()
        self.targets = self._encoder.fit_transform(self.labels)

    @property
    def data(self):
        # TODO Cache?
        # TODO return as a sparse matrix with option to return as numpy array
        return np.array(list(self.feature_space))

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

    def as_dataframe(self, include_targets=True, include_labels=True, annotations=None):
        # TODO Remove annotations
        index = pd.RangeIndex(len(self.feature_space))

        if isinstance(annotations, pd.DataFrame):
            annotations.reindex(index=index)

        if include_labels:
            label_df = pd.DataFrame(self.labels, index, ['label'])
        else:
            label_df = None

        if include_targets:
            target_df = pd.DataFrame(self.targets, index, ['targets'])
        else:
            target_df = None

        data_df = pd.DataFrame(self.data, index, self.feature_names)

        dfs = filter(lambda x: False if x is None else True, [label_df, target_df, data_df])

        return pd.concat(dfs, axis=1, copy=False)


def dedupe(seq):
    value = []
    uniques = set()
    set_add = uniques.add
    for item in seq:
        if item not in uniques:
            value.append(item)
        set_add(item)

    return value


def get_dataset(config, data, user_features=None, default_operations=None, label=None):
    # TODO rename to extract_features, extract_featureset... ?
    if user_features is None:
        user_features = []

    domain = [feature.domain for feature in user_features]
    feature_names = [feature.name for feature in user_features]

    for settings in config:
        operations = settings.get('operations', default_operations)
        operations.extend(default_operations)
        uniq_ops = dedupe(operations)

        _data = data[settings['signal_name']]

        for op in uniq_ops:
            feature_name = '_'.join(
                [settings.get('display_name', settings['signal_name']), op])
            user_features.append(create_feature(_data, op))
            domain.append(settings.get('categorical', False))
            feature_names.append(feature_name)

    feature_space = FeatureSpace([slice(10, 50), slice(100, 200)], *user_features)

    dataset = LabeledDataSet(feature_space, feature_names, labels=label, feature_domain=domain)

    return dataset
