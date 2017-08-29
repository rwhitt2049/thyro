import abc
import typing

from .stats import statistic_factory

__all__ = ['Feature', 'feature']


class BaseFeature(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, item):
        pass


class Feature(BaseFeature):
    def __init__(self, data, statistic, name, is_categorical=False):
        """

        Args:
            data (``numpy.ndarray``):
            statistic (``Callable``):
            name (``str``):
            is_categorical (``bool``):
        """

        self.data = data
        self.name = name
        self.is_nominal = is_categorical
        self.statistic = statistic

    def __call__(self, item):
        """

        Args:
            item (``slice``):

        Returns:


        """
        return self.statistic(self.data[item])


def feature(data, statistic, name, is_categorical=False):
    if isinstance(statistic, str):
        statistic = statistic_factory[statistic]

    try:
        assert isinstance(statistic, typing.Callable)
    except AssertionError as e:
        raise TypeError('statistic must be a string name of a valid stat or a callable')

    return Feature(data, statistic, name, is_categorical)
