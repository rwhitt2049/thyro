import numpy as np


__all__ = ['FeatureSpace']


class FeatureSpace:
    def __init__(self, *features, segments=None):
        if segments is None:
            self.segments = [slice(None, None)]
        if hasattr(segments, '__iter__') and not isinstance(segments, str):
            self.segments = segments
        else:
            raise TypeError('segments argument must be a sequence of slice objects')

        self.features = features

    @property
    def shape(self):
        return len(self.segments), len(self.features)

    def __len__(self):
        return len(self.segments)

    def __iter__(self):
        for segment in self.segments:
            yield np.array([feature(segment) for feature in self.features])
