import numpy as np
from fast_histogram import histogram1d, histogram2d

DTYPES = ['>i4', '<f4', '>f8', '<f8']
SIZES = [1e3, 1e4, 1e5, 1e6, 1e7]


class Histogram1D:

    params = ([False, True], DTYPES, SIZES)
    param_names = ['fast', 'dtype', 'size']

    def setup(self, fast, dtype, size):
        np.random.seed(12345)
        self.x = (np.random.random(int(size)) * 10).astype(dtype)
        self.w = (np.random.random(int(size)) * 10).astype(dtype)

    def time_histogram1d(self, fast, dtype, size):
        if fast:
            try:
                histogram1d(self.x, range=[-1, 2], bins=30)
            except TypeError:  # old API
                histogram1d(self.x, 30, -1, 2)
        else:
            np.histogram(self.x, range=[-1, 2], bins=30)

    def time_histogram1d_weights(self, fast, dtype, size):
        if fast:
            histogram1d(self.x, range=[-1, 2], bins=30, weights=self.w)
        else:
            np.histogram(self.x, range=[-1, 2], bins=30, weights=self.w)


class Histogram2D:

    params = ([False, True], DTYPES, SIZES)
    param_names = ['fast', 'dtype', 'size']

    def setup(self, fast, dtype, size):
        np.random.seed(12345)
        self.x = (np.random.random(int(size)) * 10).astype(dtype)
        self.y = (np.random.random(int(size)) * 10).astype(dtype)
        self.w = (np.random.random(int(size)) * 10).astype(dtype)

    def time_histogram2d(self, fast, dtype, size):
        if fast:
            try:
                histogram2d(self.x, self.y, range=[[-1, 2], [-2, 4]], bins=30)
            except TypeError:  # old API
                histogram2d(self.x, self.y, 30, -1, 2, 30, -2, 4)
        else:
            np.histogram2d(self.x, self.y, range=[[-1, 2], [-2, 4]], bins=30)

    def time_histogram2d_weights(self, fast, dtype, size):
        if fast:
            histogram2d(self.x, self.y, range=[[-1, 2], [-2, 4]], bins=30, weights=self.w)
        else:
            np.histogram2d(self.x, self.y, range=[[-1, 2], [-2, 4]], bins=30, weights=self.w)
