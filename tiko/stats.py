import numpy as np
from scipy import stats


__all__ = ['register_stat']


statistic_factory = {
    'min': np.min,
    'max': np.max,
    'mean': np.mean,
    'median': np.median,
    'std': np.std,
    'sum': np.sum,
    'cumsum': np.cumsum,
}


def register_stat(func=None, name=None):
    """Function to register stats callables.

    Decorator to register user defined stats function. Can be called
    as a function to register things like closures or partial functions.
    Stats are registered as their function name.

    Args:
        func (``Callable``): Function to register.
        name (``str``, optional): If not provided, function is
            registered as it's name.

    Returns:
        ``Callable``: If used as decorator.
        ``None``: If used as a function

    Raises:
        ``KeyError``: If function exists in ``tiko.stats.statistic_factory``

    Examples:
        >>>

    """
    def inner(func_):
        nonlocal name
        if name is None:
            name = func_.__name__

        if name in statistic_factory:
            raise KeyError('Function already exists. Chose a new name for %s' % name)

        statistic_factory[name] = func_
        return func_

    if func is None:
        return inner
    else:
        inner(func)


@register_stat
def mode_count(a):
    return stats.mode(a, axis=None)[1]


@register_stat
def mode(a):
    return stats.mode(a, axis=None)[0]


@register_stat
def first(a):
    return a[0]


@register_stat
def last(a):
    return a[-1]


@register_stat
def delta(a):
    return last(a) - first(a)
