from abc import ABCMeta, abstractmethod
from collections import Callable

import numpy as np

__all__ = ['Feature', 'create_feature', 'BaseFeature']


STATISTIC_FACTORY = {
    'min': np.min,
    'max': np.max,
    'mean': np.mean,
}


def statistic_factory(statistic):
    if isinstance(statistic, str):
        try:
            _statistic = STATISTIC_FACTORY[statistic]
        except KeyError:
            raise NotImplementedError('%s is not implemented' % statistic)
    elif isinstance(statistic, Callable):
        _statistic = statistic
    else:
        raise TypeError('%s must be a callable or a string name for a default statistic.' % statistic)

    return _statistic


class BaseFeature(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self):
        pass


class Feature(BaseFeature):
    def __init__(self, data, statistic, name, is_nominal=False):
        self.data = data
        self.name = name
        self._statistic = statistic
        if isinstance(is_nominal, bool):
            self.is_nominal = is_nominal
        else:
            raise TypeError('is_nominal must be a bool')

    @property
    def operation(self):
        # TODO cache
        return statistic_factory(self._statistic)

    def __call__(self, item=slice(None, None)):
        return self.operation(self.data[item])


def create_feature(data, statistic):
    _operation = statistic_factory(statistic)

    def _call(item=slice(None, None)):
        return _operation(data[item])

    return _call
