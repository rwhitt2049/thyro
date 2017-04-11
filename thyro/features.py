from abc import ABCMeta, abstractmethod

from .stats import statistic_factory

__all__ = ['Feature', 'create_feature', 'BaseFeature']


class BaseFeature(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, item):
        pass


class Feature(BaseFeature):
    def __init__(self, data, statistic, name, is_nominal=False):
        self.data = data
        self.name = name
        self.statistic = statistic_factory(statistic)
        if isinstance(is_nominal, bool):
            self.is_nominal = is_nominal
        else:
            raise TypeError('is_nominal must be a bool')

    def __call__(self, item):
        return self.statistic(self.data[item])


def create_feature(data, statistic):
    _operation = statistic_factory(statistic)

    def _call(item):
        return _operation(data[item])

    return _call
