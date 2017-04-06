import numpy as np
from collections import Callable
from abc import ABCMeta, abstractmethod


FACTORY = {
    'min': np.min,
    'max': np.max,
    'mean': np.mean,
}


class ABFeature(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, data, operation):
        self.data = data
        self.operation = operation

    @abstractmethod
    def __call__(self, item=slice(None, None)):
        return self.operation(self.data[item])


class Feature(ABFeature):
    def __init__(self, data, operation):
        super().__init__(data, operation)

    def __call__(self, item=slice(None, None)):
        return self.operation(self.data[item])


def get_feature(data, operation):
    if isinstance(operation, str):
        try:
            _function = FACTORY[operation]
        except KeyError:
            raise NotImplementedError('%s is not implemented' % operation)
    elif isinstance(operation, Callable):
        _function = operation
    else:
        raise TypeError('%s must be a callable or a string name for a default operation.' % operation)

    def _call(item=slice(None, None)):
        return _function(data[item])

    return _call
