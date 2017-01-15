import numpy as np


FACTORY = {
    'min': np.min,
    'max': np.max,
    'mean': np.mean,
}


class Feature:
    def __init__(self, data, function):
        self.data = data
        self.function = function

    def __call__(self, item=slice(None, None)):
        return self.function(self.data[item])


def get_feature(data, function):
    if isinstance(function, str):
        try:
            function = FACTORY[function]
        except KeyError:
            raise NotImplementedError('%s is not implemented' % function)

    def _call(item=slice(None, None)):
        return function(data[item])

    return _call
