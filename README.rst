|Travis Status| |AppVeyor Status|

About
-----

Sometimes you just want to compute simple 1D or 2D histograms with regular bins. Fast. No
nonsense. `Numpy's <http://www.numpy.org>`__ histogram functions are
versatile, and can handle for example non-regular binning, but this
versatility comes at the expense of performance.

The **fast-histogram** mini-package aims to provide simple and fast
histogram functions for regular bins that don't compromise on performance. It doesn't do
anything complicated - it just implements a simple histogram algorithm
in C and keeps it simple. The aim is to have functions that are fast but
also robust and reliable. The result is a 1D histogram function here that
is **7-15x faster** than ``numpy.histogram``, and a 2D histogram function
that is **20-25x faster** than ``numpy.histogram2d``.

To install:

::

    pip install fast-histogram

The ``fast_histogram`` module then provides two functions:
``histogram1d`` and ``histogram2d``:

.. code:: python

    from fast_histogram import histogram1d, histogram2d

Example
-------

Here's an example of binning 10 million points into a regular 2D
histogram:

.. code:: python

    In [1]: import numpy as np

    In [2]: x = np.random.random(10_000_000)

    In [3]: y = np.random.random(10_000_000)

    In [4]: %timeit _ = np.histogram2d(x, y, range=[[-1, 2], [-2, 4]], bins=30)
    935 ms ± 58.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    In [5]: from fast_histogram import histogram2d

    In [6]: %timeit _ = histogram2d(x, y, range=[[-1, 2], [-2, 4]], bins=30)
    40.2 ms ± 624 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

(note that ``10_000_000`` is possible in Python 3.6 syntax, use ``10000000`` instead in previous versions)

The version here is over 20 times faster! The following plot shows the
speedup as a function of array size for the bin parameters shown above:

.. figure:: https://github.com/astrofrog/fast-histogram/raw/master/speedup_compared.png
   :alt: Comparison of performance between Numpy and fast-histogram

as well as results for the 1D case, also with 30 bins. The speedup for
the 2D case is consistently between 20-25x, and for the 1D case goes
from 15x for small arrays to around 7x for large arrays.

Q&A
---

Why don't the histogram functions return the edges?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computing and returning the edges may seem trivial but it can slow things down by a factor of a few when computing histograms of 10^5 or fewer elements, so not returning the edges is a deliberate decision related to performance. You can easily compute the edges yourself if needed though, using ``numpy.linspace``.

Doesn't package X already do this, but better?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This may very well be the case! If this duplicates another package, or
if it is possible to use Numpy in a smarter way to get the same
performance gains, please open an issue and I'll consider deprecating
this package :)

One package that does include fast histogram functions (including in
n-dimensions) and can compute other statistics is
`vaex <https://github.com/maartenbreddels/vaex>`_, so take a look there
if you need more advanced functionality!

Are the 2D histograms not transposed compared to what they should be?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is technically no 'right' and 'wrong' orientation - here we adopt
the convention which gives results consistent with Numpy, so:

.. code:: python

    numpy.histogram2d(x, y, range=[[xmin, xmax], [ymin, ymax]], bins=[nx, ny])

should give the same result as:

.. code:: python

    fast_histogram.histogram2d(x, y, range=[[xmin, xmax], [ymin, ymax]], bins=[nx, ny])

Why not contribute this to Numpy directly?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned above, the Numpy functions are much more versatile, so they could not
be replaced by the ones here. One option would be to check in Numpy's functions for
cases that are simple and dispatch to functions such as the ones here, or add
dedicated functions for regular binning. I hope we can get this in Numpy in some form
or another eventually, but for now, the aim is to have this available to packages
that need to support a range of Numpy versions.

Why not use Cython?
~~~~~~~~~~~~~~~~~~~

I originally implemented this in Cython, but found that I could get a
50% performance improvement by going straight to a C extension.

What about using Numba?
~~~~~~~~~~~~~~~~~~~~~~~

I specifically want to keep this package as easy as possible to install,
and while `Numba <https://numba.pydata.org>`__ is a great package, it is
not trivial to install outside of Anaconda.

Could this be parallelized?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This may benefit from parallelization under certain circumstances. The
easiest solution might be to use OpenMP, but this won't work on all
platforms, so it would need to be made optional.

Couldn't you make it faster by using the GPU?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Almost certainly, though the aim here is to have an easily installable
and portable package, and introducing GPUs is going to affect both of
these.

Why make a package specifically for this? This is a tiny amount of functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Packages that need this could simply bundle their own C extension or
Cython code to do this, but the main motivation for releasing this as a
mini-package is to avoid making pure-Python packages into packages that
require compilation just because of the need to compute fast histograms.

Can I contribute?
~~~~~~~~~~~~~~~~~

Yes please! This is not meant to be a finished package, and I welcome
pull request to improve things.

.. |Travis Status| image:: https://travis-ci.org/astrofrog/fast-histogram.svg?branch=master
   :target: https://travis-ci.org/astrofrog/fast-histogram

.. |AppVeyor Status| image:: https://ci.appveyor.com/api/projects/status/ek63g9haku5on0q2/branch/master?svg=true
   :target: https://ci.appveyor.com/project/astrofrog/fast-histogram
