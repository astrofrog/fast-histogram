import numpy as np

import pytest

from hypothesis import given, settings, example, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ..histogram import histogram1d, histogram2d

# NOTE: for now we don't test the full range of floating-point values in the
# tests below, because Numpy's behavior isn't always deterministic in some
# of the extreme regimes. We should add manual (non-hypothesis and not
# comparing to Numpy) test cases.


@given(size=st.integers(0, 100),
       nx=st.integers(1, 10),
       xmin=st.floats(-1e10, 1e10), xmax=st.floats(-1e10, 1e10),
       weights=st.booleans(),
       dtype=st.sampled_from(['>f4', '<f4', '>f8', '<f8']))
@settings(max_examples=5000)
def test_1d_compare_with_numpy(size, nx, xmin, xmax, weights, dtype):

    if xmax <= xmin:
        return

    x = arrays(dtype, size, elements=st.floats(-1000, 1000)).example()
    if weights:
        w = arrays(dtype, size, elements=st.floats(-1000, 1000)).example()
    else:
        w = None

    try:
        reference = np.histogram(x, bins=nx, weights=w, range=(xmin, xmax))[0]
    except ValueError:
        if 'f4' in dtype:
            # Numpy has a bug in certain corner cases
            # https://github.com/numpy/numpy/issues/11586
            return
        else:
            raise

    # First, check the Numpy result because it sometimes doesn't make sense. See
    # bug report https://github.com/numpy/numpy/issues/9435
    inside = (x <= xmax) & (x >= xmin)
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
    if 'f4' in dtype:
        fast = fast.astype(np.float32)

    np.testing.assert_equal(fast, reference)


@given(size=st.integers(0, 100),
       nx=st.integers(1, 10),
       xmin=st.floats(-1e10, 1e10), xmax=st.floats(-1e10, 1e10),
       ny=st.integers(1, 10),
       ymin=st.floats(-1e10, 1e10), ymax=st.floats(-1e10, 1e10),
       weights=st.booleans(),
       dtype=st.sampled_from(['>f4', '<f4', '>f8', '<f8']))
@settings(max_examples=5000)
@example(size=5, nx=1, xmin=0.0, xmax=84.17833763374462, ny=1, ymin=-999999999.9999989, ymax=0.0, weights=False, dtype='<f8')
@example(size=1, nx=1, xmin=-2.2204460492503135e-06, xmax=0.0, ny=1, ymin=0.0, ymax=1.1102230246251567e-05, weights=False, dtype='<f8')
@example(size=3, nx=1, xmin=0.0, xmax=841.7833941010146, ny=1, ymin=-135.92383885097095, ymax=0.0, weights=True, dtype='>f8')
def test_2d_compare_with_numpy(size, nx, xmin, xmax, ny, ymin, ymax, weights, dtype):

    if xmax <= xmin or ymax <= ymin:
        return

    x = arrays(dtype, size, elements=st.floats(-1000, 1000)).example()
    y = arrays(dtype, size, elements=st.floats(-1000, 1000)).example()
    if weights:
        w = arrays(dtype, size, elements=st.floats(-1000, 1000)).example()
    else:
        w = None

    try:
        reference = np.histogram2d(x, y, bins=(nx, ny), weights=w,
                                   range=((xmin, xmax), (ymin, ymax)))[0]
    except Exception:
        # If Numpy fails, we skip the comparison since this isn't our fault
        return

    # First, check the Numpy result because it sometimes doesn't make sense. See
    # bug report https://github.com/numpy/numpy/issues/9435
    inside = (x <= xmax) & (x >= xmin) & (y <= ymax) & (y >= ymin)
    if weights:
        assume(np.allclose(np.sum(w[inside]), np.sum(reference)))
    else:
        n_inside = np.sum(inside)
        assume(n_inside == np.sum(reference))

    fast = histogram2d(x, y, bins=(nx, ny), weights=w,
                       range=((xmin, xmax), (ymin, ymax)))

    np.testing.assert_equal(fast, reference)


def test_nd_arrays():

    x = np.random.random(1000)

    result_1d = histogram1d(x, bins=10, range=(0, 1))
    result_3d = histogram1d(x.reshape((10, 10, 10)), bins=10, range=(0, 1))

    np.testing.assert_equal(result_1d, result_3d)

    y = np.random.random(1000)

    result_1d = histogram2d(x, y, bins=(10, 10), range=[(0, 1), (0, 1)])
    result_3d = histogram2d(x.reshape((10, 10, 10)), y.reshape((10, 10, 10)),
                            bins=(10, 10), range=[(0, 1), (0, 1)])

    np.testing.assert_equal(result_1d, result_3d)


def test_list():

    # Make sure that lists can be passed in

    x_list = [1.4, 2.1, 4.2]
    x_arr = np.array(x_list)

    result_list = histogram1d(x_list, bins=10, range=(0, 10))
    result_arr = histogram1d(x_arr, bins=10, range=(0, 10))

    np.testing.assert_equal(result_list, result_arr)


def test_non_contiguous():

    x = np.random.random((10, 10, 10))[::2, ::3, :]
    y = np.random.random((10, 10, 10))[::2, ::3, :]
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


def test_array_bins():

    edges = np.array([0, 1, 2, 3, 4])

    with pytest.raises(TypeError) as exc:
        histogram1d([1, 2, 3], bins=edges, range=(0, 10))
    assert exc.value.args[0] == 'bins should be an integer'

    with pytest.raises(TypeError) as exc:
        histogram2d([1, 2, 3], [1, 2 ,3], bins=[edges, edges],
                    range=[(0, 10), (0, 10)])
    assert exc.value.args[0] == 'bins should be an iterable of two integers'
