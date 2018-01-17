#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

/* Define docstrings */
static char module_docstring[] = "Fast histogram functioins";
static char _histogram1d_docstring[] = "Compute a 1D histogram";
static char _histogram2d_docstring[] = "Compute a 2D histogram";
static char _histogram1d_weighted_docstring[] = "Compute a weighted 1D histogram";
static char _histogram2d_weighted_docstring[] = "Compute a weighted 2D histogram";

/* Declare the C functions here. */
static PyObject *_histogram1d(PyObject *self, PyObject *args);
static PyObject *_histogram2d(PyObject *self, PyObject *args);
static PyObject *_histogram1d_weighted(PyObject *self, PyObject *args);
static PyObject *_histogram2d_weighted(PyObject *self, PyObject *args);
static PyObject *_histogram1d_f32(PyObject *self, PyObject *args);
static PyObject *_histogram2d_f32(PyObject *self, PyObject *args);
static PyObject *_histogram1d_weighted_f32(PyObject *self, PyObject *args);
static PyObject *_histogram2d_weighted_f32(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"_histogram1d", _histogram1d, METH_VARARGS, _histogram1d_docstring},
    {"_histogram2d", _histogram2d, METH_VARARGS, _histogram2d_docstring},
    {"_histogram1d_weighted", _histogram1d_weighted, METH_VARARGS, _histogram1d_weighted_docstring},
    {"_histogram2d_weighted", _histogram2d_weighted, METH_VARARGS, _histogram2d_weighted_docstring},
    {"_histogram1d_f32", _histogram1d_f32, METH_VARARGS, _histogram1d_docstring},
    {"_histogram2d_f32", _histogram2d_f32, METH_VARARGS, _histogram2d_docstring},
    {"_histogram1d_weighted_f32", _histogram1d_weighted_f32, METH_VARARGS, _histogram1d_weighted_docstring},
    {"_histogram2d_weighted_f32", _histogram2d_weighted_f32, METH_VARARGS, _histogram2d_weighted_docstring},
    {NULL, NULL, 0, NULL}
};

/* This is the function that is called on import. */

#if PY_MAJOR_VERSION >= 3
  #define MOD_ERROR_VAL NULL
  #define MOD_SUCCESS_VAL(val) val
  #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
          ob = PyModule_Create(&moduledef);
#else
  #define MOD_ERROR_VAL
  #define MOD_SUCCESS_VAL(val)
  #define MOD_INIT(name) void init##name(void)
  #define MOD_DEF(ob, name, doc, methods) \
          ob = Py_InitModule3(name, methods, doc);
#endif

MOD_INIT(_histogram_core)
{
    PyObject *m;
    MOD_DEF(m, "_histogram_core", module_docstring, module_methods);
    if (m == NULL)
        return MOD_ERROR_VAL;
    import_array();
    return MOD_SUCCESS_VAL(m);
}


static PyObject *_histogram1d(PyObject *self, PyObject *args)
{
    long i, n;
    int ix, nx;
    npy_float64 xmin, xmax, fnx, normx;
    PyObject *x_obj, *x_array, *count_array;
    npy_intp dims[1];
    npy_float64 *x, *count;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "Oidd", &x_obj, &nx, &xmin, &xmax)) {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    x_array = PyArray_FROM_OTF(x_obj, NPY_FLOAT64, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        return NULL;
    }

    /* How many data points are there? */
    n = (long)PyArray_DIM(x_array, 0);

    /* Build the output array */
    dims[0] = nx;
    count_array = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (count_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_XDECREF(count_array);
        return NULL;
    }

    /* Releasing GIL, no allocation/referencing allowed
     (see `ndarrayTypes.h` for macros) */
    Py_BEGIN_ALLOW_THREADS
    PyArray_FILLWBYTE(count_array, 0);

    /* Get pointers to the data as C-Types. */
    x = (npy_float64*)PyArray_DATA(x_array);
    count = (npy_float64*)PyArray_DATA(count_array);

    fnx = (npy_float64)nx;
    normx = 1. / (xmax - xmin);

    for(i = 0; i < n; i++) {
        if (x[i] >= xmin && x[i] < xmax) {
            ix = (int)((x[i] - xmin) * normx * fnx);
            count[ix] += 1.;
        }
    }
    
    /* Clean up. */
    Py_END_ALLOW_THREADS
    Py_DECREF(x_array);

    return count_array;
}

static PyObject *_histogram1d_f32(PyObject *self, PyObject *args)
{
    long i, n;
    int ix, nx;
    npy_float32 xmin, xmax, fnx, normx;
    PyObject *x_obj, *x_array, *count_array;
    npy_intp dims[1];
    npy_float32 *x, *count;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "Oiff", &x_obj, &nx, &xmin, &xmax)) {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    x_array = PyArray_FROM_OTF(x_obj, NPY_FLOAT32, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        return NULL;
    }

    /* How many data points are there? */
    n = (long)PyArray_DIM(x_array, 0);

    /* Build the output array */
    dims[0] = nx;
    count_array = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (count_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_XDECREF(count_array);
        return NULL;
    }

    /* Releasing GIL, no allocation/referencing allowed
     (see `ndarrayTypes.h` for macros) */
    Py_BEGIN_ALLOW_THREADS
    PyArray_FILLWBYTE(count_array, 0);

    /* Get pointers to the data as C-Types. */
    x = (npy_float32*)PyArray_DATA(x_array);
    count = (npy_float32*)PyArray_DATA(count_array);

    fnx = (npy_float32)nx;
    normx = 1.f / (xmax - xmin);

    for(i = 0; i < n; i++) {
      if (x[i] >= xmin && x[i] < xmax) {
          ix = (int)((x[i] - xmin) * normx * fnx);
          count[ix] += 1.f;
      }

    }
    
    /* Clean up. */
    Py_END_ALLOW_THREADS
    Py_DECREF(x_array);

    return count_array;
}

static PyObject *_histogram2d(PyObject *self, PyObject *args)
{
    long i, n;
    int ix, iy, nx, ny;
    npy_float64 xmin, xmax, fnx, normx, ymin, ymax, fny, normy;
    PyObject *x_obj, *y_obj, *x_array, *y_array, *count_array;
    npy_intp dims[2];
    npy_float64 *x, *y, *count;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOiddidd", &x_obj, &y_obj, &nx, &xmin, &xmax, &ny, &ymin, &ymax)) {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    x_array = PyArray_FROM_OTF(x_obj, NPY_FLOAT64, NPY_IN_ARRAY);
    y_array = PyArray_FROM_OTF(y_obj, NPY_FLOAT64, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL || y_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    /* How many data points are there? */
    n = (long)PyArray_DIM(x_array, 0);

    /* Check the dimensions. */
    if (n != (long)PyArray_DIM(y_array, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Dimension mismatch between x and y");
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        return NULL;
    }

    /* Build the output array */
    dims[0] = nx;
    dims[1] = ny;

    count_array = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (count_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_XDECREF(count_array);
        return NULL;
    }

    /* Releasing GIL, no allocation/referencing allowed
     (see `ndarrayTypes.h` for macros) */
    Py_BEGIN_ALLOW_THREADS
    PyArray_FILLWBYTE(count_array, 0);

    /* Get pointers to the data as C-Types. */
    x = (npy_float64*)PyArray_DATA(x_array);
    y = (npy_float64*)PyArray_DATA(y_array);
    count = (npy_float64*)PyArray_DATA(count_array);

    fnx = (npy_float64)nx;
    fny = (npy_float64)ny;
    normx = 1. / (xmax - xmin);
    normy = 1. / (ymax - ymin);

    for(i = 0; i < n; i++) {
      if (x[i] >= xmin && x[i] < xmax && y[i] >= ymin && y[i] < ymax) {
          ix = (int)((x[i] - xmin) * normx * fnx);
          iy = (int)((y[i] - ymin) * normy * fny);
          count[iy + ny * ix] += 1.;
      }
    }
    
    /* Clean up. */
    Py_END_ALLOW_THREADS
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    return count_array;
}

static PyObject *_histogram2d_f32(PyObject *self, PyObject *args)
{
    long i, n;
    int ix, iy, nx, ny;
    npy_float32 xmin, xmax, fnx, normx, ymin, ymax, fny, normy;
    PyObject *x_obj, *y_obj, *x_array, *y_array, *count_array;
    npy_intp dims[2];
    npy_float32 *x, *y, *count;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOiffiff", &x_obj, &y_obj, &nx, &xmin, &xmax, &ny, &ymin, &ymax)) {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    x_array = PyArray_FROM_OTF(x_obj, NPY_FLOAT32, NPY_IN_ARRAY);
    y_array = PyArray_FROM_OTF(y_obj, NPY_FLOAT32, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL || y_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    /* How many data points are there? */
    n = (long)PyArray_DIM(x_array, 0);

    /* Check the dimensions. */
    if (n != (long)PyArray_DIM(y_array, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Dimension mismatch between x and y");
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        return NULL;
    }

    /* Build the output array */
    dims[0] = nx;
    dims[1] = ny;

    count_array = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (count_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_XDECREF(count_array);
        return NULL;
    }

    /* Releasing GIL, no allocation/referencing allowed
     (see `ndarrayTypes.h` for macros) */
    Py_BEGIN_ALLOW_THREADS
    PyArray_FILLWBYTE(count_array, 0);

    /* Get pointers to the data as C-Types. */
    x = (npy_float32*)PyArray_DATA(x_array);
    y = (npy_float32*)PyArray_DATA(y_array);
    count = (npy_float32*)PyArray_DATA(count_array);

    fnx = (npy_float32)nx;
    fny = (npy_float32)ny;
    normx = 1.f / (xmax - xmin);
    normy = 1.f / (ymax - ymin);

    for(i = 0; i < n; i++) {
      if (x[i] >= xmin && x[i] < xmax && y[i] >= ymin && y[i] < ymax) {
          ix = (int)((x[i] - xmin) * normx * fnx);
          iy = (int)((y[i] - ymin) * normy * fny);
          count[iy + ny * ix] += 1.f;
      }
    }
    
    /* Clean up. */
    Py_END_ALLOW_THREADS
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    return count_array;
}


static PyObject *_histogram1d_weighted(PyObject *self, PyObject *args)
{
    long i, n;
    int ix, nx;
    npy_float64 xmin, xmax, fnx, normx;
    PyObject *x_obj, *x_array, *w_obj, *w_array, *count_array;
    npy_intp dims[1];
    npy_float64 *x, *w, *count;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOidd", &x_obj, &w_obj, &nx, &xmin, &xmax)) {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    x_array = PyArray_FROM_OTF(x_obj, NPY_FLOAT64, NPY_IN_ARRAY);
    w_array = PyArray_FROM_OTF(w_obj, NPY_FLOAT64, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL || w_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        Py_XDECREF(w_array);
        return NULL;
    }

    /* How many data points are there? */
    n = (long)PyArray_DIM(x_array, 0);

    /* Check the dimensions. */
    if (n != (long)PyArray_DIM(w_array, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Dimension mismatch between x and w");
        Py_DECREF(x_array);
        Py_DECREF(w_array);
        return NULL;
    }

    /* Build the output array */
    dims[0] = nx;
    count_array = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (count_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_DECREF(w_array);
        Py_XDECREF(count_array);
        return NULL;
    }

    /* Releasing GIL, no allocation/referencing allowed
     (see `ndarrayTypes.h` for macros) */
    Py_BEGIN_ALLOW_THREADS
    PyArray_FILLWBYTE(count_array, 0);

    /* Get pointers to the data as C-Types. */

    x = (npy_float64*)PyArray_DATA(x_array);
    w = (npy_float64*)PyArray_DATA(w_array);
    count = (npy_float64*)PyArray_DATA(count_array);

    fnx = nx;
    normx = 1. / (xmax - xmin);

    for(i = 0; i < n; i++) {
      if (x[i] >= xmin && x[i] < xmax) {
          ix = (int)((x[i] - xmin) * normx * fnx);
          count[ix] += w[i];
      }
    }

    /* Clean up. */
    Py_END_ALLOW_THREADS
    Py_DECREF(x_array);
    Py_DECREF(w_array);

    return count_array;
}

static PyObject *_histogram1d_weighted_f32(PyObject *self, PyObject *args)
{
    long i, n;
    int ix, nx;
    npy_float32 xmin, xmax, fnx, normx;
    PyObject *x_obj, *x_array, *w_obj, *w_array, *count_array;
    npy_intp dims[1];
    npy_float32 *x, *w, *count;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOiff", &x_obj, &w_obj, &nx, &xmin, &xmax)) {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    x_array = PyArray_FROM_OTF(x_obj, NPY_FLOAT32, NPY_IN_ARRAY);
    w_array = PyArray_FROM_OTF(w_obj, NPY_FLOAT32, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL || w_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        Py_XDECREF(w_array);
        return NULL;
    }

    /* How many data points are there? */
    n = (long)PyArray_DIM(x_array, 0);

    /* Check the dimensions. */
    if (n != (long)PyArray_DIM(w_array, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Dimension mismatch between x and w");
        Py_DECREF(x_array);
        Py_DECREF(w_array);
        return NULL;
    }

    /* Build the output array */
    dims[0] = nx;
    count_array = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (count_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_DECREF(w_array);
        Py_XDECREF(count_array);
        return NULL;
    }

    /* Releasing GIL, no allocation/referencing allowed
     (see `ndarrayTypes.h` for macros) */
    Py_BEGIN_ALLOW_THREADS
    PyArray_FILLWBYTE(count_array, 0);

    /* Get pointers to the data as C-Types. */

    x = (npy_float32*)PyArray_DATA(x_array);
    w = (npy_float32*)PyArray_DATA(w_array);
    count = (npy_float32*)PyArray_DATA(count_array);

    fnx = nx;
    normx = 1. / (xmax - xmin);

    for(i = 0; i < n; i++) {
      if (x[i] >= xmin && x[i] < xmax) {
          ix = (int)((x[i] - xmin) * normx * fnx);
          count[ix] += w[i];
      }
    }

    /* Clean up. */
    Py_END_ALLOW_THREADS
    Py_DECREF(x_array);
    Py_DECREF(w_array);

    return count_array;
}

static PyObject *_histogram2d_weighted(PyObject *self, PyObject *args)
{
    long i, n;
    int ix, iy, nx, ny;
    npy_float64 xmin, xmax, fnx, normx, ymin, ymax, fny, normy;
    PyObject *x_obj, *y_obj, *w_obj, *x_array, *y_array, *w_array, *count_array;
    npy_intp dims[2];
    npy_float64 *x, *y, *w, *count;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOiddidd", &x_obj, &y_obj, &w_obj, &nx, &xmin, &xmax, &ny, &ymin, &ymax)) {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    x_array = PyArray_FROM_OTF(x_obj, NPY_FLOAT64, NPY_IN_ARRAY);
    y_array = PyArray_FROM_OTF(y_obj, NPY_FLOAT64, NPY_IN_ARRAY);
    w_array = PyArray_FROM_OTF(w_obj, NPY_FLOAT64, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL || y_array == NULL || w_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        Py_XDECREF(w_array);
        return NULL;
    }

    /* How many data points are there? */
    n = (long)PyArray_DIM(x_array, 0);

    /* Check the dimensions. */
    if (n != (long)PyArray_DIM(y_array, 0) || n != (long)PyArray_DIM(w_array, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Dimension mismatch between x, y, and w");
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        Py_DECREF(w_array);
        return NULL;
    }

    /* Build the output array */
    dims[0] = nx;
    dims[1] = ny;

    count_array = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (count_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_XDECREF(count_array);
        return NULL;
    }

    /* Releasing GIL, no allocation/referencing allowed
     (see `ndarrayTypes.h` for macros) */
    Py_BEGIN_ALLOW_THREADS
    PyArray_FILLWBYTE(count_array, 0);

    /* Get pointers to the data as C-Types. */
    x = (npy_float64*)PyArray_DATA(x_array);
    y = (npy_float64*)PyArray_DATA(y_array);
    w = (npy_float64*)PyArray_DATA(w_array);
    count = (npy_float64*)PyArray_DATA(count_array);

    fnx = (npy_float64)nx;
    fny = (npy_float64)ny;
    normx = 1. / (xmax - xmin);
    normy = 1. / (ymax - ymin);

    for(i = 0; i < n; i++) {
        if (x[i] >= xmin && x[i] < xmax && y[i] >= ymin && y[i] < ymax) {
            ix = (int)((x[i] - xmin) * normx * fnx);
            iy = (int)((y[i] - ymin) * normy * fny);
            count[iy + ny * ix] += w[i];
        }
    }

    /* Clean up. */
    Py_END_ALLOW_THREADS
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(w_array);

    return count_array;
}

static PyObject *_histogram2d_weighted_f32(PyObject *self, PyObject *args)
{
    long i, n;
    int ix, iy, nx, ny;
    npy_float32 xmin, xmax, fnx, normx, ymin, ymax, fny, normy;
    PyObject *x_obj, *y_obj, *w_obj, *x_array, *y_array, *w_array, *count_array;
    npy_intp dims[2];
    npy_float32 *x, *y, *w, *count;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOiffiff", &x_obj, &y_obj, &w_obj, &nx, &xmin, &xmax, &ny, &ymin, &ymax)) {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    x_array = PyArray_FROM_OTF(x_obj, NPY_FLOAT32, NPY_IN_ARRAY);
    y_array = PyArray_FROM_OTF(y_obj, NPY_FLOAT32, NPY_IN_ARRAY);
    w_array = PyArray_FROM_OTF(w_obj, NPY_FLOAT32, NPY_IN_ARRAY);

    /* If that didn't work, throw an `Exception`. */
    if (x_array == NULL || y_array == NULL || w_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't parse the input arrays.");
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        Py_XDECREF(w_array);
        return NULL;
    }

    /* How many data points are there? */
    n = (long)PyArray_DIM(x_array, 0);

    /* Check the dimensions. */
    if (n != (long)PyArray_DIM(y_array, 0) || n != (long)PyArray_DIM(w_array, 0)) {
        PyErr_SetString(PyExc_RuntimeError, "Dimension mismatch between x, y, and w");
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        Py_DECREF(w_array);
        return NULL;
    }

    /* Build the output array */
    dims[0] = nx;
    dims[1] = ny;

    count_array = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (count_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_XDECREF(count_array);
        return NULL;
    }

    /* Releasing GIL, no allocation/referencing allowed
     (see `ndarrayTypes.h` for macros) */
    Py_BEGIN_ALLOW_THREADS
    PyArray_FILLWBYTE(count_array, 0);

    /* Get pointers to the data as C-Types. */
    x = (npy_float32*)PyArray_DATA(x_array);
    y = (npy_float32*)PyArray_DATA(y_array);
    w = (npy_float32*)PyArray_DATA(w_array);
    count = (npy_float32*)PyArray_DATA(count_array);

    fnx = (npy_float32)nx;
    fny = (npy_float32)ny;
    normx = 1. / (xmax - xmin);
    normy = 1. / (ymax - ymin);

    for(i = 0; i < n; i++) {
        if (x[i] >= xmin && x[i] < xmax && y[i] >= ymin && y[i] < ymax) {
            ix = (int)((x[i] - xmin) * normx * fnx);
            iy = (int)((y[i] - ymin) * normy * fny);
            count[iy + ny * ix] += w[i];
        }
    }

    /* Clean up. */
    Py_END_ALLOW_THREADS
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(w_array);

    return count_array;
}