import numpy as np
from scipy import stats


__all__ = ['register_stat']


STATISTIC_FACTORY = {
    'min': np.min,
    'max': np.max,
    'mean': np.mean,
    'median': np.median,
    'std': np.std,
    'sum': np.sum,
    'cumsum': np.cumsum,
    # 'mode': mode,
    # 'mode_count': mode_count,
    # 'first': first,
    # 'last': last,
    # 'delta': delta
}


def register_stat(func=None):
    """Function to register stats callables.

    Decorator to register user defined stats function. Can be called
    as a function to register things like closures or partial functions.
    Stats are registered as their function name.

    Args:
        func (``Callable``): Function to register.

    Returns:
        ``Callable``: if used as decorator, other returns ``None``

    Raises:
        ``KeyError``: If function exists in tiko.stats.STATISTIC_FACTORY

    Examples:
        >>>

    """
    def inner(func_):
        name = func_.__name__
        if name not in STATISTIC_FACTORY:
            STATISTIC_FACTORY[func_.__name__] = func_
        else:
            raise KeyError('Function already exists. Chose a new name for %s' % name)
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


def statistic_factory(statistic):
    """

    Args:
        statistic (``str``): Statistic string name

    Returns:
        ``Callable``:

    Raises:
        ``ValueError``: If stat doesn't exist in ``STATISTIC_FACTORY``

    """
    try:
        _statistic = STATISTIC_FACTORY[statistic]
    except KeyError:
        raise ValueError('%s is not a valid stat name' % statistic)

    return _statistic
