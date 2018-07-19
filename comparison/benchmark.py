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
TARGET_TIME = 0.1


def average_time(stmt=None, setup=None):

    # Call once to check how long it takes
    time_single = timeit(stmt=stmt, setup=setup, number=1)

    # Find out how many times we can call it. We always call it at least three
    # times for accuracy
    n_repeats = max(3, int(TARGET_TIME / time_single))

    times = repeat(stmt=stmt, setup=setup, repeat=n_repeats, number=3)

    return np.mean(times), np.std(times)


FMT_HEADER = '# {:7s}' + ' {:10s}' * 8 + '\n'
FMT = '{:9d}' + ' {:10.7e}' * 8 + '\n'

with open('benchmark_times.txt', 'w') as f:

    f.write(FMT_HEADER.format('size',
                              'np_1d_mean', 'np_1d_std', 'fa_1d_mean', 'fa_1d_std',
                              'np_2d_mean', 'np_2d_std', 'fa_2d_mean', 'fa_2d_std'))

    for log10_size in range(0, 9):

        size = int(10 ** log10_size)

        print('Running benchmarks for size={0}'.format(size))

        np_1d_mean, np_1d_std = average_time(stmt=NUMPY_1D_STMT, setup=SETUP_1D.format(size=size))
        fa_1d_mean, fa_1d_std = average_time(stmt=FAST_1D_STMT, setup=SETUP_1D.format(size=size))
        np_2d_mean, np_2d_std = average_time(stmt=NUMPY_2D_STMT, setup=SETUP_2D.format(size=size))
        fa_2d_mean, fa_2d_std = average_time(stmt=FAST_2D_STMT, setup=SETUP_2D.format(size=size))

        f.write(FMT.format(size,
                           np_1d_mean, np_1d_std, fa_1d_mean, fa_1d_std,
                           np_2d_mean, np_2d_std, fa_2d_mean, fa_2d_std))
        f.flush()
