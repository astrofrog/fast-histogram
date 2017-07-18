from __future__ import division

import numbers

import numpy as np

from ._histogram_core import _histogram1d, _histogram2d

__all__ = ['histogram1d', 'histogram2d']


def histogram1d(x, bins, range):
    """
    Compute a 1D histogram assuming equally spaced bins.

    Parameters
    ----------
    x : `~numpy.ndarray`
        The position of the points to bin in the 1D histogram
    bins : int
        The number of bins
    range : float, optional
        The range as a tuple of (xmin, xmax)

    Returns
    -------
    array : `~numpy.ndarray`
        The 1D histogram array
    """

    nx = bins
    xmin, xmax = range

    if not np.isfinite(xmin):
        raise ValueError("xmin should be finite")

    if not np.isfinite(xmax):
        raise ValueError("xmax should be finite")

    if xmax <= xmin:
        raise ValueError("xmax should be greater than xmin")

    if nx <= 0:
        raise ValueError("nx should be strictly positive")

    x = np.ascontiguousarray(x, np.float)

    return _histogram1d(x, nx, xmin, xmax)


def histogram2d(x, y, bins, range):
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

    Returns
    -------
    array : `~numpy.ndarray`
        The 2D histogram array
    """

    if isinstance(bins, numbers.Integral):
        nx = ny = bins
    else:
        nx, ny = bins

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

    x = np.ascontiguousarray(x, np.float)
    y = np.ascontiguousarray(y, np.float)

    return _histogram2d(x, y, nx, xmin, xmax, ny, ymin, ymax)
