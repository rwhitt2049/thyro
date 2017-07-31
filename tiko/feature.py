from abc import ABCMeta, abstractmethod
import typing

from .stats import statistic_factory

__all__ = ['Feature', 'BaseFeature']


class BaseFeature(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, item):
        pass

# TODO: Statistic should only ever be a callable, no more "or"s in arguments
class Feature(BaseFeature):
    def __init__(self, data, statistic, name, is_nominal=False):
        """

        Args:
            data (``numpy.array``):
            statistic (``str`` or callable):
            name (``str``):
            is_nominal (``bool``):
        """

        self.data = data
        self.name = name
        self.is_nominal = is_nominal
        if isinstance(statistic, str):
            self.statistic = statistic_factory(statistic)
        elif isinstance(statistic, typing.Callable):
            self.statistic = statistic
        else:
            raise TypeError('statistic must be a string name or a callable')

    def __call__(self, item):
        """

        Args:
            item (``slice``):

        Returns:


        """
        return self.statistic(self.data[item])
