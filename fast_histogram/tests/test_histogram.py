import numpy as np

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
       weights=st.booleans())
@settings(max_examples=5000)
def test_1d_compare_with_numpy(size, nx, xmin, xmax, weights):

    if xmax <= xmin:
        return

    x = arrays(np.float, size, elements=st.floats(-1000, 1000)).example()
    if weights:
        w = arrays(np.float, size, elements=st.floats(-1000, 1000)).example()
    else:
        w = None

    reference = np.histogram(x, bins=nx, weights=w, range=(xmin, xmax))[0]

    # First, check the Numpy result because it sometimes doesn't make sense. See
    # bug report https://github.com/numpy/numpy/issues/9435
    inside = (x <= xmax) & (x >= xmin)
    if weights:
        assume(np.allclose(np.sum(w[inside]), np.sum(reference)))
    else:
        n_inside = np.sum(inside)
        assume(n_inside == np.sum(reference))

    fast = histogram1d(x, bins=nx, weights=w, range=(xmin, xmax))

    np.testing.assert_equal(fast, reference)


@given(size=st.integers(0, 100),
       nx=st.integers(1, 10),
       xmin=st.floats(-1e10, 1e10), xmax=st.floats(-1e10, 1e10),
       ny=st.integers(1, 10),
       ymin=st.floats(-1e10, 1e10), ymax=st.floats(-1e10, 1e10),
       weights=st.booleans())
@settings(max_examples=5000)
@example(size=5, nx=1, xmin=0.0, xmax=84.17833763374462, ny=1, ymin=-999999999.9999989, ymax=0.0, weights=False)
@example(size=1, nx=1, xmin=-2.2204460492503135e-06, xmax=0.0, ny=1, ymin=0.0, ymax=1.1102230246251567e-05, weights=False)
def test_2d_compare_with_numpy(size, nx, xmin, xmax, ny, ymin, ymax, weights):

    if xmax <= xmin or ymax <= ymin:
        return

    x = arrays(np.float, size, elements=st.floats(-1000, 1000)).example()
    y = arrays(np.float, size, elements=st.floats(-1000, 1000)).example()
    if weights:
        w = arrays(np.float, size, elements=st.floats(-1000, 1000)).example()
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
