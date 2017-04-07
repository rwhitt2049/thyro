from abc import ABCMeta, abstractmethod
from collections import Callable

import numpy as np

__all__ = ['Feature', 'create_feature', 'BaseFeature']


FACTORY = {
    'min': np.min,
    'max': np.max,
    'mean': np.mean,
}


def operation_factory(operation):
    if isinstance(operation, str):
        try:
            _operation = FACTORY[operation]
        except KeyError:
            raise NotImplementedError('%s is not implemented' % operation)
    elif isinstance(operation, Callable):
        _operation = operation
    else:
        raise TypeError('%s must be a callable or a string name for a default operation.' % operation)

    return _operation


class BaseFeature(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self):
        pass


class Feature(BaseFeature):
    def __init__(self, data, operation, name, is_categorical=False):
        self.data = data
        self.name = name
        self._operation = operation
        if isinstance(is_categorical, bool):
            self.is_categorical = is_categorical
        else:
            raise TypeError('is_categorical must be a bool')

    @property
    def operation(self):
        # TODO cache
        return operation_factory(self._operation)

    def __call__(self, item=slice(None, None)):
        return self.operation(self.data[item])


def create_feature(data, operation):
    _operation = operation_factory(operation)

    def _call(item=slice(None, None)):
        return _operation(data[item])

    return _call
