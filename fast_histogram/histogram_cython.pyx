from __future__ import division

import numpy as np
cimport numpy as np
from libc.math cimport round

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cimport cython


@cython.boundscheck(False)
def histogram1d_cython(np.ndarray[DTYPE_t, ndim=1] x,
                       int nx, double xmin, double xmax):

    cdef int n = x.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=1] count = np.zeros([nx], dtype=DTYPE)

    cdef int ix
    cdef unsigned int i
    cdef double normx
    cdef double tx
    cdef double fnx = nx

    normx = 1. / (xmax - xmin)

    with nogil:
        for i in range(n):
            tx = x[i]
            if tx > xmin and tx < xmax:
                ix = int((tx - xmin) * normx * fnx)
                count[ix] += 1.

    return count


@cython.boundscheck(False)
@cython.cdivision(True)
def histogram2d_cython(np.ndarray[DTYPE_t, ndim=1] x,
                       np.ndarray[DTYPE_t, ndim=1] y,
                       int nx, double xmin, double xmax,
                       int ny, double ymin, double ymax):

    cdef int n = x.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] count = np.zeros([ny, nx], dtype=DTYPE)

    cdef int ix, iy
    cdef unsigned int i, j
    cdef double normx, normy
    cdef double tx, ty
    cdef double fnx = nx
    cdef double fny = ny

    normx = 1. / (xmax - xmin)
    normy = 1. / (ymax - ymin)

    with nogil:
        for i in range(n):
            tx = x[i]
            ty = y[i]
            if tx >= xmin and tx <= xmax and ty >= ymin and ty <= ymax:
                ix = int((tx - xmin) * normx * fnx)
                iy = int((ty - ymin) * normy * fny)
                count[iy, ix] += 1.

    return count
