from __future__ import division

import numbers

import numpy as np

from ._histogram_core import (_histogram1d,
                              _histogram2d,
                              _histogramdd,
                              _histogram1d_weighted,
                              _histogram2d_weighted,
                              _histogramdd_weighted)

__all__ = ['histogram1d', 'histogram2d', 'histogramdd']


def histogram1d(x, bins, range, weights=None):
    """
    Compute a 1D histogram assuming equally spaced bins.

    Parameters
    ----------
    x : `~numpy.ndarray`
        The position of the points to bin in the 1D histogram
    bins : int
        The number of bins
    range : iterable
        The range as a tuple of (xmin, xmax)
    weights : `~numpy.ndarray`
        The weights of the points in the 1D histogram

    Returns
    -------
    array : `~numpy.ndarray`
        The 1D histogram array
    """

    nx = bins

    if not np.isscalar(bins):
        raise TypeError('bins should be an integer')

    xmin, xmax = range

    if not np.isfinite(xmin):
        raise ValueError("xmin should be finite")

    if not np.isfinite(xmax):
        raise ValueError("xmax should be finite")

    if xmax <= xmin:
        raise ValueError("xmax should be greater than xmin")

    if nx <= 0:
        raise ValueError("nx should be strictly positive")

    if weights is None:
        return _histogram1d(x, nx, xmin, xmax)
    else:
        return _histogram1d_weighted(x, weights, nx, xmin, xmax)


def histogram2d(x, y, bins, range, weights=None):
    """
    Compute a 2D histogram assuming equally spaced bins.

    Parameters
    ----------
    x, y : `~numpy.ndarray`
        The position of the points to bin in the 2D histogram
    bins : int or iterable
        The number of bins in each dimension. If given as an integer, the same
        number of bins is used for each dimension.
    range : iterable
        The range to use in each dimention, as an iterable of value pairs, i.e.
        [(xmin, xmax), (ymin, ymax)]
    weights : `~numpy.ndarray`
        The weights of the points in the 1D histogram

    Returns
    -------
    array : `~numpy.ndarray`
        The 2D histogram array
    """

    if isinstance(bins, numbers.Integral):
        nx = ny = bins
    else:
        nx, ny = bins

    if not np.isscalar(nx) or not np.isscalar(ny):
        raise TypeError('bins should be an iterable of two integers')

    (xmin, xmax), (ymin, ymax) = range

    if not np.isfinite(xmin):
        raise ValueError("xmin should be finite")

    if not np.isfinite(xmax):
        raise ValueError("xmax should be finite")

    if not np.isfinite(ymin):
        raise ValueError("ymin should be finite")

    if not np.isfinite(ymax):
        raise ValueError("ymax should be finite")

    if xmax <= xmin:
        raise ValueError("xmax should be greater than xmin")

    if ymax <= ymin:
        raise ValueError("xmax should be greater than xmin")

    if nx <= 0:
        raise ValueError("nx should be strictly positive")

    if ny <= 0:
        raise ValueError("ny should be strictly positive")

    if weights is None:
        return _histogram2d(x, y, nx, xmin, xmax, ny, ymin, ymax)
    else:
        return _histogram2d_weighted(x, y, weights, nx, xmin, xmax, ny, ymin, ymax)

def histogramdd(sample, bins, range, weights=None):
    """
    Compute a histogram of N samples in D dimensions.
    
    Parameters
    ----------
    sample : (N, D) `~numpy.ndarray`, or (D, N) array_like
        The data to be histogrammed.
        * When an array_like, each element is the list of values for single
          coordinate - such as ``histogramdd((X, Y, Z), bins, range)``.
        * When a `~numpy.ndarray`, each row is a coordinate in a D-dimensional space -
          such as ``histogramdd(np.array([p1, p2, p3]), bins, range)``.
        * In the special case of D = 1, it is allowed to pass an array or array_like
          with length N.
        The second form is converted internally into the first form, thus the first form
        is preferred.
    bins : int or iterable
        The number of bins in each dimension. If given as an integer, the same
        number of bins is used for every dimension.
    range : iterable
        The range to use in each dimention, as an iterable of D value pairs, i.e.
        [(xmin, xmax), (ymin, ymax)]
    weights : `~numpy.ndarray`
        The weights of the points in `sample`.

    Returns
    -------
    array : `~numpy.ndarray`
        The ND histogram array
    """
    
    if isinstance(sample, np.ndarray):
        _sample = tuple(np.atleast_2d(sample.T))
    else:
        # handle special case in 1D
        if isinstance(sample[0], numbers.Real):
            _sample = (sample,)
        else:
            _sample = tuple(sample)

    ndim = len(_sample)
    n = len(_sample[0])
    
    if isinstance(bins, numbers.Integral):
        _bins = bins * np.ones(ndim, dtype=np.intp)
    else:
        _bins = np.array(bins, dtype=np.intp)
    if len(_bins) != ndim:
        raise ValueError("number of bin counts does not match number of dimensions")
    if np.any(_bins <= 0):
        raise ValueError("all bin numbers should be strictly positive")
    
    _range = np.zeros((ndim, 2), dtype=np.double)
    
    if not len(range) == ndim:
        raise ValueError("number of ranges does not equal number of dimensions")
    for i, r in enumerate(range):
        if not len(r) == 2:
            raise ValueError("should pass a minimum and maximum value for each dimension")
        if r[0] >= r[1]:
            raise ValueError("each range should be strictly increasing")
        _range[i][0] = r[0]
        _range[i][1] = r[1]
    
    if weights is None:
        return _histogramdd(_sample, _bins, _range)
    else:
        return _histogramdd_weighted(_sample, _bins, _range, weights)
