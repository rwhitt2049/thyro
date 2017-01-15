import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, feature_space, feature_names, labels=None):
        self.feature_space = feature_space
        self.feature_names = feature_names
        self._labels = labels

    @property
    def data(self):
        return np.array(list(self.feature_space))

    @property
    def labels(self):
        if self.labels is None:
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
        return set(self.labels)

    # @property
    # def targets(self):
    #     return np.array() # encoded labels

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
