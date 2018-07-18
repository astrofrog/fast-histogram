import numpy as np
from fast_histogram import histogram1d, histogram2d

DTYPES = ['>i4', '<i4', '>i8', '<i8', '>f4', '<f4', '>f8', '<f8']
x, y, w = {}, {}, {}
for dtype in DTYPES:
    x[dtype] = (np.random.random(1000000) * 10).astype(dtype)
    y[dtype] = (np.random.random(1000000) * 10).astype(dtype)
    w[dtype] = (np.random.random(1000000) * 10).astype(dtype)


class Histogram1D:

    params = ([False, True], DTYPES)
    param_names = ['fast', 'dtype']

    def setup(self, fast, dtype):
        self.x = x[dtype]
        self.w = w[dtype]

    def time_histogram1d(self, fast, dtype):
        if fast:
            try:
                histogram1d(self.x, range=[-1, 2], bins=30)
            except TypeError:  # old API
                histogram1d(self.x, 30, -1, 2)
        else:
            np.histogram(self.x, range=[-1, 2], bins=30)

    def time_histogram1d_weights(self, fast, dtype):
        if fast:
            histogram1d(self.x, range=[-1, 2], bins=30, weights=self.w)
        else:
            np.histogram(self.x, range=[-1, 2], bins=30, weights=self.w)


class Histogram2D:

    params = ([False, True], DTYPES)
    param_names = ['fast', 'dtype']

    def setup(self, fast, dtype):
        self.x = x[dtype]
        self.y = y[dtype]
        self.w = y[dtype]

    def time_histogram2d(self, fast, dtype):
        if fast:
            try:
                histogram2d(self.x, self.y, range=[[-1, 2], [-2, 4]], bins=30)
            except TypeError:  # old API
                histogram2d(self.x, self.y, 30, -1, 2, 30, -2, 4)
        else:
            np.histogram2d(self.x, self.y, range=[[-1, 2], [-2, 4]], bins=30)

    def time_histogram2d_weights(self, fast, dtype):
        if fast:
            histogram2d(self.x, self.y, range=[[-1, 2], [-2, 4]], bins=30, weights=self.w)
        else:
            np.histogram2d(self.x, self.y, range=[[-1, 2], [-2, 4]], bins=30, weights=self.w)
