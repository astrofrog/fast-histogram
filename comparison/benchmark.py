# Script to compare the speedup provided by fast-histogram

import numpy as np
from timeit import timeit, repeat

SETUP_1D = """
import numpy as np
from numpy import histogram as np_histogram1d
from fast_histogram import histogram1d
x = np.random.random({size})
"""

NUMPY_1D_STMT = "np_histogram1d(x, range=[-1, 2], bins=30)"
FAST_1D_STMT = "histogram1d(x, range=[-1, 2], bins=30)"

SETUP_2D = """
import numpy as np
from numpy import histogram2d as np_histogram2d
from fast_histogram import histogram2d
x = np.random.random({size})
y = np.random.random({size})
"""

NUMPY_2D_STMT = "np_histogram2d(x, y, range=[[-1, 2], [-2, 4]], bins=30)"
FAST_2D_STMT = "histogram2d(x, y, range=[[-1, 2], [-2, 4]], bins=30)"

# How long each benchmark should aim to take
TARGET_TIME = 1.0


def time_stats(stmt=None, setup=None):

    # Call once to check how long it takes
    time_single = timeit(stmt=stmt, setup=setup, number=1)

    # Find out how many times we can call it. We always call it at least three
    # times for accuracy
    number = max(3, int(TARGET_TIME / time_single))

    print(' -> estimated time to complete test: {0:.1f}s'.format(time_single * 10 * number))

    times = repeat(stmt=stmt, setup=setup, repeat=10, number=number)

    return np.min(times) / number, np.mean(times) / number, np.median(times) / number


FMT_HEADER = '# {:7s}' + ' {:10s}' * 12 + '\n'
FMT = '{:9d}' + ' {:10.7e}' * 12 + '\n'

with open('benchmark_times.txt', 'w') as f:

    f.write(FMT_HEADER.format('size',
                              'np_1d_min', 'np_1d_mean', 'np_1d_median', 'fa_1d_min', 'fa_1d_mean', 'fa_1d_median',
                              'np_2d_min', 'np_2d_mean', 'np_2d_median', 'fa_2d_min', 'fa_2d_mean', 'fa_2d_median'))

    for log10_size in range(0, 9):

        size = int(10 ** log10_size)

        print('Running benchmarks for size={0}'.format(size))

        np_1d_min, np_1d_mean, np_1d_median = time_stats(stmt=NUMPY_1D_STMT, setup=SETUP_1D.format(size=size))
        fa_1d_min, fa_1d_mean, fa_1d_median = time_stats(stmt=FAST_1D_STMT, setup=SETUP_1D.format(size=size))
        np_2d_min, np_2d_mean, np_2d_median = time_stats(stmt=NUMPY_2D_STMT, setup=SETUP_2D.format(size=size))
        fa_2d_min, fa_2d_mean, fa_2d_median = time_stats(stmt=FAST_2D_STMT, setup=SETUP_2D.format(size=size))

        f.write(FMT.format(size,
                           np_1d_min, np_1d_mean, np_1d_median, fa_1d_min, fa_1d_mean, fa_1d_median,
                           np_2d_min, np_2d_mean, np_2d_median, fa_2d_min, fa_2d_mean, fa_2d_median))
        f.flush()
