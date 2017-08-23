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
