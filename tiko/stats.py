import numpy as np
from scipy import stats


def mode_count(a):
    return stats.mode(a, axis=None)[1]


def mode(a):
    return stats.mode(a, axis=None)[0]


def first(a):
    return a[0]


def last(a):
    return a[-1]


def delta(a):
    return last(a) - first(a)


STATISTIC_FACTORY = {
    'min': np.min,
    'max': np.max,
    'mean': np.mean,
    'median': np.median,
    'std': np.std,
    'sum': np.sum,
    'cumsum': np.cumsum,
    'mode': mode,
    'mode_count': mode_count,
    'first': first,
    'last': last,
    'delta': delta
}


def statistic_factory(statistic):
    """

    Args:
        statistic (``str``):

    Returns:

    """
    try:
        _statistic = STATISTIC_FACTORY[statistic]
    except KeyError:
        raise NotImplementedError('%s is not implemented' % statistic)

    return _statistic