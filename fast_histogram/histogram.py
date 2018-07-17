from __future__ import division

import numbers
import numpy as np

from ._histogram_core import (_histogram1d,
                              _histogram2d,
                              _histogram1d_weighted,
                              _histogram2d_weighted)

__all__ = ['histogram1d', 'histogram2d']

_cast_to = {
    np.dtype('float64'): None,
    np.dtype('float32'): None,
    np.dtype('bool')   : np.dtype('float32'),
    np.dtype('uint8')  : np.dtype('float32'),
    np.dtype('int8')   : np.dtype('float32'),
    np.dtype('uint16') : np.dtype('float32'),
    np.dtype('int16')  : np.dtype('float32'),
    np.dtype('uint32') : np.dtype('float64'),
    np.dtype('int32')  : np.dtype('float64'),
    np.dtype('uint64') : np.dtype('float64'),
    np.dtype('int64')  : np.dtype('float64')
}


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
    xmin, xmax = range

    if not np.isfinite(xmin):
        raise ValueError("xmin should be finite")

    if not np.isfinite(xmax):
        raise ValueError("xmax should be finite")

    if xmax <= xmin:
        raise ValueError("xmax should be greater than xmin")

    if nx <= 0:
        raise ValueError("nx should be strictly positive")

    castType = _cast_to[x.dtype]
    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x, castType)
    elif castType:
        x = x.astype(castType)

    if x.ndim > 1:
        x = x.ravel()

    if weights is None:
        return _histogram1d(x, nx, xmin, xmax)

     # Else weighted histogram
    if not weights.flags.c_contiguous:
        weights = np.ascontiguousarray(weights, castType)
    elif castType:
        weights = weights.astype(castType)
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

    castXType = _cast_to[x.dtype]
    castYType = _cast_to[y.dtype]
    castType = np.dtype('f'%np.maximum(castXType.itemsize, castYType.itemsize))

    if not x.flags.c_contiguous:
        x = np.ascontiguousarray(x, castXType)
    elif castXType != castType:
        x = x.astype(castXType)
    
    if not y.flags.c_contiguous:
        y = np.ascontiguousarray(y, castYType)
    elif castYType != castType:
        y = y.astype(castYType)

    if x.ndim > 1:
        x = x.ravel()

    if y.ndim > 1:
        y = y.ravel()

    if weights is None:
        return _histogram2d(x, y, nx, xmin, xmax, ny, ymin, ymax)

    # Else weighted histogram
    if not weights.flags.c_contiguous:
        weights = np.ascontiguousarray(weights, castXType)
    elif weights.dtype != castType:
        weights = weights.astype(castXType)

    return _histogram2d_weighted(x, y, weights, nx, xmin, xmax, ny, ymin, ymax)
