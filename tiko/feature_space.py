__all__ = ['FeatureSpace']


class FeatureSpace:
    def __init__(self, features, segments=None):
        self.features = features
        if segments is None:
            self.segments = [slice(None, None)]
        else:
            self.segments = segments

    @property
    def shape(self):
        return len(self.segments), len(self.features)

    @property
    def feature_names(self):
        return tuple(feature.name for feature in self.features)

    def __len__(self):
        return len(self.segments)

    def __iter__(self):
        for segment in self.segments:
            yield tuple(feature(segment) for feature in self.features)
