import itertools

import numpy as np

from .dataset import UnlabeledDataSet, LabeledDataSet


def homogeneous_type(seq):
    iseq = iter(seq)
    first_type = type(next(iseq))
    return first_type if all((type(x) is first_type) for x in iseq) else False


def homogeneous_attribute(seq):
    iseq = iter(seq)
    first_attribute = next(iseq)
    return first_attribute if all(x == first_attribute for x in iseq) else False


def concat(*datasets):

    homogeneous_feature_type = homogeneous_type(datasets)
    if not homogeneous_feature_type:
        raise TypeError('All datasets must be of a homogeneous type.')

    homogeneous_feature_names = homogeneous_attribute([dataset.feature_names for dataset in datasets])
    if not homogeneous_feature_names:
        raise TypeError('All datasets must have homogeneous feature_names.')

    homogeneous_nominal_features = homogeneous_attribute([dataset.nominal_features for dataset in datasets])
    if not homogeneous_nominal_features:
        raise TypeError('All datasets must have homogeneous nominal_features.')

    new_feature_space = np.vstack([dataset.feature_space for dataset in datasets])

    if homogeneous_feature_type is UnlabeledDataSet:
        new_dataset = homogeneous_feature_type(new_feature_space, homogeneous_feature_names,
                                               homogeneous_nominal_features)
    elif homogeneous_feature_type is LabeledDataSet:
        all_labels = [dataset.labels for dataset in datasets]
        new_labels = list(itertools.chain.from_iterable(all_labels))
        new_dataset = homogeneous_feature_type(new_feature_space, homogeneous_feature_names,
                                               new_labels, homogeneous_nominal_features)
    else:
        raise TypeError('Don\'t know how to concatenate this type of DataSet')

    return new_dataset
