# Script to make the comparison plot for the benchmark

import numpy as np
import matplotlib.pyplot as plt

(size,
 np_1d_mean, np_1d_std, fa_1d_mean, fa_1d_std,
 np_2d_mean, np_2d_std, fa_2d_mean, fa_2d_std) = np.loadtxt('benchmark_times.txt', unpack=True)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(size, np_1d_mean / fa_1d_mean, color=(34 / 255, 122 / 255, 181 / 255), label='1D')
ax.plot(size, np_2d_mean / fa_2d_mean, color=(255 / 255, 133 / 255, 25 / 255), label='2D')
ax.set_xscale('log')
ax.set_xlim(0.3, 3e8)
ax.set_ylim(1, 35)
ax.grid()
ax.set_xlabel('Array size')
ax.set_ylabel('Speedup (fast-histogram / numpy)')
fig.savefig('speedup_compared.png', bbox_inches='tight')
