from .dataset import UnlabeledDataSet, LabeledDataSet
import itertools


def homogeneous_type(seq):
    iseq = iter(seq)
    first_type = type(next(iseq))
    return first_type if all((type(x) is first_type) for x in iseq) else False


def homogeneous_attribute(seq):
    iseq = iter(seq)
    first_feature_names = next(iseq)
    return first_feature_names if all(x == first_feature_names for x in iseq) else False


def concat(*datasets):

    if homogeneous_type(datasets):
        new_dataset = homogeneous_type(datasets)
    else:
        raise TypeError('All datasets must be of a homogeneous type.')

    homogeneous_feature_names = homogeneous_attribute([dataset.feature_names for dataset in datasets])
    if not homogeneous_feature_names:
        raise TypeError('All datasets must have homogeneous feature_names.')

    homogeneous_nominal_features = homogeneous_attribute([dataset.nominal_features for dataset in datasets])
    if not homogeneous_nominal_features:
        raise TypeError('All datasets must have homogeneous nominal_features.')

    new_feature_space = (feature_vector for dataset in datasets
                                            for feature_vector in dataset.feature_space)

    if new_dataset is UnlabeledDataSet:
        return new_dataset(new_feature_space, homogeneous_feature_names, homogeneous_nominal_features)
    elif new_dataset is LabeledDataSet:
        all_labels = [dataset.labels for dataset in datasets]
        new_labels = list(itertools.chain.from_iterable(all_labels))
        return new_dataset(new_feature_space, homogeneous_feature_names, new_labels, homogeneous_nominal_features)
    else:
        raise TypeError('Don\'t know how to concatenate this type of Dataset')
