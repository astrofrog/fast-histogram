# Script to make the comparison plot for the benchmark

import matplotlib.pyplot as plt
import numpy as np

(
    size,
    np_1d_min,
    np_1d_mean,
    np_1d_median,
    fa_1d_min,
    fa_1d_mean,
    fa_1d_median,
    np_2d_min,
    np_2d_mean,
    np_2d_median,
    fa_2d_min,
    fa_2d_mean,
    fa_2d_median,
    np_3d_min,
    np_3d_mean,
    np_3d_median,
    fa_3d_min,
    fa_3d_mean,
    fa_3d_median,
) = np.loadtxt("benchmark_times.txt", unpack=True)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(size, np_1d_min / fa_1d_min, label="1D")
ax.plot(size, np_2d_min / fa_2d_min, label="2D")
ax.plot(size, np_3d_min / fa_3d_min, label="DD (3D)")
ax.set_xscale("log")
ax.set_xlim(0.3, 3e8)
ax.set_ylim(1, 20)
ax.grid()
ax.set_xlabel("Array size")
ax.set_ylabel(f"Speedup (fast-histogram / numpy (version {np.__version__})")
ax.legend()
fig.savefig("speedup_compared.png", bbox_inches="tight")
