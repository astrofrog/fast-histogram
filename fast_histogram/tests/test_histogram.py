import numpy as np

import pytest

from hypothesis import given, settings, assume, example
from hypothesis import strategies as st
from hypothesis import HealthCheck
from hypothesis.extra.numpy import arrays

from ..histogram import histogram1d, histogram2d, histogramdd

# NOTE: for now we don't test the full range of floating-point values in the
# tests below, because Numpy's behavior isn't always deterministic in some
# of the extreme regimes. We should add manual (non-hypothesis and not
# comparing to Numpy) test cases.

@given(values=arrays(dtype='<f8', shape=st.integers(0, 200),
                     elements=st.floats(-1000, 1000), unique=True),
       nx=st.integers(1, 10),
       xmin=st.floats(-1e10, 1e10),
       xmax=st.floats(-1e10, 1e10),
       weights=st.booleans(),
       dtype=st.sampled_from(['>f4', '<f4', '>f8', '<f8']))
@settings(max_examples=500)
def test_1d_compare_with_numpy(values, nx, xmin, xmax, weights, dtype):

    if xmax <= xmin:
        return

    values = values.astype(dtype)

    size = len(values) // 2

    if weights:
        w = values[:size]
    else:
        w = None

    x = values[size:size * 2]

    try:
        reference = np.histogram(x, bins=nx, weights=w, range=(xmin, xmax))[0]
    except ValueError:
        if 'f4' in str(x.dtype):
            # Numpy has a bug in certain corner cases
            # https://github.com/numpy/numpy/issues/11586
            return
        else:
            raise

    # First, check the Numpy result because it sometimes doesn't make sense. See
    # bug report https://github.com/numpy/numpy/issues/9435
    # FIXME: for now use < since that's what our algorithm does
    inside = (x < xmax) & (x >= xmin)
    if weights:
        assume(np.allclose(np.sum(w[inside]), np.sum(reference)))
    else:
        n_inside = np.sum(inside)
        assume(n_inside == np.sum(reference))

    fast = histogram1d(x, bins=nx, weights=w, range=(xmin, xmax))

    # Numpy returns results for 32-bit results as a 32-bit histogram, but only
    # for 1D arrays. Since this is a summation variable it makes sense to
    # return 64-bit, so rather than changing the behavior of histogram1d, we
    # cast to 32-bit float here.
    if x.dtype.kind == 'f' and x.dtype.itemsize == 4:
        rtol = 1e-7
    else:
        rtol = 1e-14

    np.testing.assert_allclose(fast, reference, rtol=rtol)
    
    fastdd = histogramdd((x,), bins=nx, weights=w, range=[(xmin, xmax)])
    np.testing.assert_array_equal(fast, fastdd)

@given(values=arrays(dtype='<f8', shape=st.integers(0, 300),
                     elements=st.floats(-1000, 1000), unique=True),
       nx=st.integers(1, 10),
       xmin=st.floats(-1e10, 1e10), xmax=st.floats(-1e10, 1e10),
       ny=st.integers(1, 10),
       ymin=st.floats(-1e10, 1e10), ymax=st.floats(-1e10, 1e10),
       weights=st.booleans(),
       dtype=st.sampled_from(['>f4', '<f4', '>f8', '<f8']))
@settings(max_examples=500)
def test_2d_compare_with_numpy(values, nx, xmin, xmax, ny, ymin, ymax, weights, dtype):

    if xmax <= xmin or ymax <= ymin:
        return

    values = values.astype(dtype)

    size = len(values) // 3

    if weights:
        w = values[:size]
    else:
        w = None

    x = values[size:size * 2]
    y = values[size * 2:size * 3]

    try:
        reference = np.histogram2d(x, y, bins=(nx, ny), weights=w,
                                   range=((xmin, xmax), (ymin, ymax)))[0]
    except Exception:
        # If Numpy fails, we skip the comparison since this isn't our fault
        return

    # First, check the Numpy result because it sometimes doesn't make sense. See
    # bug report https://github.com/numpy/numpy/issues/9435.
    # FIXME: for now use < since that's what our algorithm does
    inside = (x < xmax) & (x >= xmin) & (y < ymax) & (y >= ymin)
    if weights:
        assume(np.allclose(np.sum(w[inside]), np.sum(reference)))
    else:
        n_inside = np.sum(inside)
        assume(n_inside == np.sum(reference))

    fast = histogram2d(x, y, bins=(nx, ny), weights=w,
                       range=((xmin, xmax), (ymin, ymax)))

    if x.dtype.kind == 'f' and x.dtype.itemsize == 4:
        rtol = 1e-7
    else:
        rtol = 1e-14

    np.testing.assert_allclose(fast, reference, rtol=rtol)
    
    fastdd = histogramdd((x, y), bins=(nx, ny), weights=w,
                         range=((xmin, xmax), (ymin, ymax)))
    np.testing.assert_array_equal(fast, fastdd)

@given(values=arrays(dtype='<f8', shape=st.integers(0, 1000),
                     elements=st.floats(-1000, 1000), unique=True),
       hist_size=st.integers(1, 1e5),
       bins=arrays(elements=st.integers(1, 10), shape=(10,), dtype=np.int32),
       ranges=arrays(elements=st.floats(1e-10, 1e5), dtype='<f8',
                     shape=(10,), unique=True),
       weights=st.booleans(),
       dtype=st.sampled_from(['>f4', '<f4', '>f8', '<f8']))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_dd_compare_with_numpy(values, hist_size, bins, ranges, weights, dtype):

    # To avoid generating huge histograms that take a long time, we only take
    # as many dimensions as we can such that the total hist_size is still within the
    # limit. If `hist_size = 1`, we will take all the leading ones in `bins`.
    _bins = []
    accum_size = 1
    for i in range(10):
        if bins[i] * accum_size > hist_size:
            break
        _bins.append(bins[i]) 
        accum_size *= bins[i]
    ndim = len(_bins)
    values = values.astype(dtype)
    ranges = ranges.astype(dtype)
    ranges = ranges[:ndim]
    # Ranges are symmetric because otherwise the probability of samples falling inside
    # is just too small and we would just be testing a bunch of empty histograms.
    ranges = np.vstack((-ranges, ranges)).T
    
    size = len(values) // (ndim + 1)

    if weights:
        w = values[:size]
    else:
        w = None
    
    sample = tuple(values[size*(i+1):size*(i+2)] for i in range(ndim))
    # for simplicity using the same range in all dimensions
    try:
        reference = np.histogramdd(sample, bins=_bins, weights=w, range=ranges)[0]
    except Exception:
        # If Numpy fails, we skip the comparison since this isn't our fault
        return

    # First, check the Numpy result because it sometimes doesn't make sense. See
    # bug report https://github.com/numpy/numpy/issues/9435.
    # FIXME: for now use < since that's what our algorithm does
    inside = (sample[0] < ranges[0][1]) & (sample[0] >= ranges[0][0])
    if ndim > 1:
        for i in range(ndim - 1):
            inside = inside & (sample[i+1] < ranges[i+1][1]) & (sample[i+1] >= ranges[i+1][0])
    if weights:
        assume(np.allclose(np.sum(w[inside]), np.sum(reference)))
    else:
        n_inside = np.sum(inside)
        assume(n_inside == np.sum(reference))

    fast = histogramdd(sample, bins=_bins, weights=w, range=ranges)

    if sample[0].dtype.kind == 'f' and sample[0].dtype.itemsize == 4:
        rtol = 1e-7
    else:
        rtol = 1e-14

    np.testing.assert_allclose(fast, reference, rtol=rtol)

def test_nd_arrays():

    x = np.random.random(1000)

    result_1d = histogram1d(x, bins=10, range=(0, 1))
    result_3d = histogram1d(x.reshape((10, 10, 10)), bins=10, range=(0, 1))
    result_3d_dd = histogramdd((x.reshape((10, 10, 10)),), bins=10, range=((0, 1), ))

    np.testing.assert_equal(result_1d, result_3d)
    np.testing.assert_equal(result_1d, result_3d_dd)

    y = np.random.random(1000)

    result_1d = histogram2d(x, y, bins=(10, 10), range=[(0, 1), (0, 1)])
    result_3d = histogram2d(x.reshape((10, 10, 10)), y.reshape((10, 10, 10)),
                            bins=(10, 10), range=[(0, 1), (0, 1)])
    result_3d_dd = histogramdd((x.reshape((10, 10, 10)), y.reshape((10, 10, 10))),
                               bins=(10, 10), range=[(0, 1), (0, 1)])
    
    np.testing.assert_equal(result_1d, result_3d)
    np.testing.assert_equal(result_1d, result_3d_dd)


def test_list():

    # Make sure that lists can be passed in

    x_list = [1.4, 2.1, 4.2]
    x_arr = np.array(x_list)

    result_list = histogram1d(x_list, bins=10, range=(0, 10))
    result_arr = histogram1d(x_arr, bins=10, range=(0, 10))

    np.testing.assert_equal(result_list, result_arr)
    
    result_list_dd = histogramdd(x_list, bins=10, range=((0, 10),))
    result_arr_dd = histogramdd(x_arr, bins=10, range=((0, 10),))

    np.testing.assert_equal(result_list_dd, result_arr_dd)

def test_histogramdd_interface():
    # make sure the interface of histogramdd works as numpy.histogramdd
    x_list = [1.4, 2.1, 4.2, 8.7, 5.1]
    x_arr = np.array(x_list)
    y_list = [6.6, 3.2, 2.9, 3.9, 0.1]
    y_arr = np.array(y_list)
    
    # test 1D (needs special handling in case the sample is a list)
    sample = x_arr
    result_np, _ = np.histogramdd(sample, bins=10, range=((0, 10),))
    result_fh = histogramdd(sample, bins=10, range=((0, 10),))
    np.testing.assert_equal(result_np, result_fh)
    
    sample = x_list
    result_np, _ = np.histogramdd(sample, bins=10, range=((0, 10),))
    result_fh = histogramdd(sample, bins=10, range=((0, 10),))
    np.testing.assert_equal(result_np, result_fh)
    
    # test (D, N) array_like
    sample = (x_arr, y_arr)
    result_np, _ = np.histogramdd(sample, bins=10, range=((0, 10), (0, 10)))
    result_fh = histogramdd(sample, bins=10, range=((0, 10), (0, 10)))
    np.testing.assert_equal(result_np, result_fh)
    
    sample = [x_arr, y_arr]
    result_np, _ = np.histogramdd(sample, bins=10, range=((0, 10), (0, 10)))
    result_fh = histogramdd(sample, bins=10, range=((0, 10), (0, 10)))
    np.testing.assert_equal(result_np, result_fh)
    
    sample = (x_list, y_list)
    result_np, _ = np.histogramdd(sample, bins=10, range=((0, 10), (0, 10)))
    result_fh = histogramdd(sample, bins=10, range=((0, 10), (0, 10)))
    np.testing.assert_equal(result_np, result_fh)
    
    sample = [x_list, y_list]
    result_np, _ = np.histogramdd(sample, bins=10, range=((0, 10), (0, 10)))
    result_fh = histogramdd(sample, bins=10, range=((0, 10), (0, 10)))
    np.testing.assert_equal(result_np, result_fh)
    
    # test (N, D) array
    sample = np.vstack([x_arr, y_arr]).T
    result_np, _ = np.histogramdd(sample, bins=10, range=((0, 10), (0, 10)))
    result_fh = histogramdd(sample, bins=10, range=((0, 10), (0, 10)))
    np.testing.assert_equal(result_np, result_fh)
    
    sample = np.vstack([x_list, y_list]).T
    result_np, _ = np.histogramdd(sample, bins=10, range=((0, 10), (0, 10)))
    result_fh = histogramdd(sample, bins=10, range=((0, 10), (0, 10)))
    np.testing.assert_equal(result_np, result_fh)
    
def test_non_contiguous():

    x = np.random.random((10, 10, 10))[::2, ::3, :]
    y = np.random.random((10, 10, 10))[::2, ::3, :]
    z = np.random.random((10, 10, 10))[::2, ::3, :]
    w = np.random.random((10, 10, 10))[::2, ::3, :]

    assert not x.flags.c_contiguous
    assert not x.flags.f_contiguous

    result_1 = histogram1d(x, bins=10, range=(0, 1))
    result_2 = histogram1d(x.copy(), bins=10, range=(0, 1))

    np.testing.assert_equal(result_1, result_2)

    result_1 = histogram1d(x, bins=10, range=(0, 1), weights=w)
    result_2 = histogram1d(x.copy(), bins=10, range=(0, 1), weights=w)

    np.testing.assert_equal(result_1, result_2)

    result_1 = histogram2d(x, y, bins=(10, 10), range=[(0, 1), (0, 1)])
    result_2 = histogram2d(x.copy(), y.copy(), bins=(10, 10),
                           range=[(0, 1), (0, 1)])

    np.testing.assert_equal(result_1, result_2)

    result_1 = histogram2d(x, y, bins=(10, 10), range=[(0, 1), (0, 1)], weights=w)
    result_2 = histogram2d(x.copy(), y.copy(), bins=(10, 10),
                           range=[(0, 1), (0, 1)], weights=w)

    np.testing.assert_equal(result_1, result_2)
    
    result_1 = histogramdd((x, y, z), bins=(10, 10, 10), range=[(0, 1), (0, 1), (0, 1)])
    result_2 = histogramdd((x.copy(), y.copy(), z.copy()), bins=(10, 10, 10),
                           range=[(0, 1), (0, 1), (0, 1)])

    np.testing.assert_equal(result_1, result_2)

    result_1 = histogramdd((x, y, z), bins=(10, 10, 10), range=[(0, 1), (0, 1), (0, 1)],
                           weights=w)
    result_2 = histogramdd((x.copy(), y.copy(), z.copy()), bins=(10, 10, 10),
                           range=[(0, 1), (0, 1), (0, 1)], weights=w)

    np.testing.assert_equal(result_1, result_2)


def test_array_bins():

    edges = np.array([0, 1, 2, 3, 4])

    with pytest.raises(TypeError) as exc:
        histogram1d([1, 2, 3], bins=edges, range=(0, 10))
    assert exc.value.args[0] == 'bins should be an integer'

    with pytest.raises(TypeError) as exc:
        histogram2d([1, 2, 3], [1, 2 ,3], bins=[edges, edges],
                    range=[(0, 10), (0, 10)])
    assert exc.value.args[0] == 'bins should be an iterable of two integers'


def test_mixed_strides():

    # Make sure all functions work properly when passed arrays with mismatched
    # strides.

    x = np.random.random((30, 20, 40, 50))[:, 10, :, 20]
    y = np.random.random((30, 40, 50))[:, :, 10]
    z = np.random.random((30, 10, 5, 80, 90))[:, 5, 2, ::2, 22]

    assert x.shape == y.shape and x.shape == z.shape
    assert x.strides != y.strides and y.strides != z.strides and z.strides != x.strides

    result_1 = histogram1d(x, bins=10, range=(0, 1))
    result_2, _ = np.histogram(x, bins=10, range=(0, 1))
    np.testing.assert_equal(result_1, result_2)

    result_3 = histogram1d(x, weights=y, bins=10, range=(0, 1))
    result_4, _ = np.histogram(x, weights=y, bins=10, range=(0, 1))
    np.testing.assert_equal(result_3, result_4)

    result_5 = histogram2d(x, y, bins=(10, 10), range=[(0, 1), (0, 1)])
    result_6, _, _ = np.histogram2d(x.ravel(), y.ravel(), bins=(10, 10), range=[(0, 1), (0, 1)])
    np.testing.assert_equal(result_5, result_6)

    result_7 = histogram2d(x, y, weights=z, bins=(10, 10), range=[(0, 1), (0, 1)])
    result_8, _, _ = np.histogram2d(x.ravel(), y.ravel(), weights=z.ravel(), bins=(10, 10), range=[(0, 1), (0, 1)])
    np.testing.assert_equal(result_7, result_8)
    
    result_9 = histogramdd((x, y), bins=(10, 10), range=[(0, 1), (0, 1)])
    result_10, _, _ = np.histogram2d(x.ravel(), y.ravel(), bins=(10, 10), range=[(0, 1), (0, 1)])
    np.testing.assert_equal(result_9, result_10)

    result_11 = histogramdd((x, y), weights=z, bins=(10, 10), range=[(0, 1), (0, 1)])
    result_12, _, _ = np.histogram2d(x.ravel(), y.ravel(), weights=z.ravel(), bins=(10, 10), range=[(0, 1), (0, 1)])
    np.testing.assert_equal(result_11, result_12)
