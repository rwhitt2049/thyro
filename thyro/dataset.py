import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# think about what needs to happen if you concat a bunch of datasets.
# check if feature_names are all the same
# Append labels end to end in order
# append feature_domain end to end in order
# Feature space.... this is currently assumed to be a FeatureSpace generator,
# make a genertor of generators

# feature_space = (feature_fector for feature_space in feature_spaces
#                                     for feature_vector in feature_space)


class DataSet:
    def __init__(self, feature_space, feature_names, labels=None, feature_domain=None):
        self.feature_space = feature_space
        self.feature_names = feature_names
        self._labels = labels
        self.feature_domain = feature_domain

    @property
    def data(self):
        #cache this?
        return np.array(list(self.feature_space))

    @property
    def labels(self):
        if self._labels is None:
            return ['None'] * len(self.feature_space)
        elif isinstance(self._labels, str):
            return [self._labels] * len(self.feature_space)
        elif isinstance(self._labels, list):
            return self._labels
        else:
            raise TypeError('Labels must None if unlabelled or be a string, '
                            'or a list of strings if labelled')

    @property
    def target_names(self):
        return self._encoder.classes_

    @property
    def _encoder(self):
        # Need to cache
        le = LabelEncoder()
        # If caching, should probably just do fit_transform
        le.fit(self.labels)
        return le

    @property
    def targets(self):
        # Need to cache
        return self._encoder.transform(self.labels)

    def as_dataframe(self, include_targets=True, include_labels=True, annotations=None):
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
