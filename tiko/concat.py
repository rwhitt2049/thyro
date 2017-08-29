import itertools

import pandas as pd

from .dataset import UnlabeledDataSet, LabeledDataSet


def homogeneous_type(seq):
    i_seq = iter(seq)
    first_type = type(next(i_seq))
    return first_type if all((type(x) is first_type) for x in i_seq) else False


def homogeneous_attribute(seq):
    i_seq = iter(seq)
    first_attribute = next(i_seq)
    return first_attribute if all(x == first_attribute for x in i_seq) else False


def concat(*datasets):
    homogeneous_feature_type = homogeneous_type(datasets)
    if not homogeneous_feature_type:
        raise TypeError('All datasets must be of a homogeneous type.')

    homogeneous_feature_names = homogeneous_attribute(
        [dataset.feature_names for dataset in datasets]
    )
    if not homogeneous_feature_names:
        raise ValueError('All datasets must have homogeneous feature_names.')

    homogeneous_nominal_features = homogeneous_attribute(
        [dataset.categorical_features for dataset in datasets]
    )
    if not homogeneous_nominal_features:
        raise ValueError('All datasets must have homogeneous nominal_features.')

    data = pd.concat(datasets)

    if homogeneous_feature_type is UnlabeledDataSet:
        new_dataset = UnlabeledDataSet(data,
                                       homogeneous_nominal_features)
    elif homogeneous_feature_type is LabeledDataSet:
        all_labels = [dataset.labels for dataset in datasets]
        new_labels = list(itertools.chain.from_iterable(all_labels))
        new_dataset = LabeledDataSet(new_labels,
                                     data,
                                     homogeneous_nominal_features)
    else:
        raise TypeError('Can\'t concatenate this type of DataSet')

    return new_dataset
