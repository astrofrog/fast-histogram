#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

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

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"_histogram1d", _histogram1d, METH_VARARGS, _histogram1d_docstring},
    {"_histogram2d", _histogram2d, METH_VARARGS, _histogram2d_docstring},
    {"_histogram1d_weighted", _histogram1d_weighted, METH_VARARGS, _histogram1d_weighted_docstring},
    {"_histogram2d_weighted", _histogram2d_weighted, METH_VARARGS, _histogram2d_weighted_docstring},
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

static PyObject *_histogram1d(PyObject *self, PyObject *args) {

  long n;
  int ix, nx;
  double xmin, xmax, tx, fnx, normx;
  PyObject *x_obj, *count_obj;
  PyArrayObject *x_array, *count_array;
  npy_intp dims[1];
  double *count;
  NpyIter *iter;
  NpyIter_IterNextFunc *iternext;
  char **dataptr;
  npy_intp *strideptr, *innersizeptr;
  PyArray_Descr *dtype;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "Oidd", &x_obj, &nx, &xmin, &xmax)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing input");
    return NULL;
  }

  /* Interpret the input objects as `numpy` arrays. */
  x_array = (PyArrayObject *)PyArray_FROM_O(x_obj);

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
  count_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  if (count_obj == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't build output array");
    Py_DECREF(x_array);
    Py_XDECREF(count_obj);
    return NULL;
  }

  count_array = (PyArrayObject *)count_obj;

  PyArray_FILLWBYTE(count_array, 0);

  if (n == 0) {
    Py_DECREF(x_array);
    return count_obj;
  }

  dtype = PyArray_DescrFromType(NPY_DOUBLE);
  iter = NpyIter_New(x_array,
                     NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED,
                     NPY_KEEPORDER, NPY_SAFE_CASTING, dtype);
  if (iter == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator");
    Py_DECREF(x_array);
    Py_DECREF(count_obj);
    Py_DECREF(count_array);
    return NULL;
  }

  /*
   * The iternext function gets stored in a local variable
   * so it can be called repeatedly in an efficient manner.
   */
  iternext = NpyIter_GetIterNext(iter, NULL);
  if (iternext == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator");
    NpyIter_Deallocate(iter);
    Py_DECREF(x_array);
    Py_DECREF(count_obj);
    Py_DECREF(count_array);
    return NULL;
  }

  /* The location of the data pointer which the iterator may update */
  dataptr = NpyIter_GetDataPtrArray(iter);

  /* The location of the stride which the iterator may update */
  strideptr = NpyIter_GetInnerStrideArray(iter);

  /* The location of the inner loop size which the iterator may update */
  innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

  /* Pre-compute variables for efficiency in the histogram calculation */
  fnx = nx;
  normx = 1. / (xmax - xmin);

  /* Get C array for output array */
  count = (double *)PyArray_DATA(count_array);

  Py_BEGIN_ALLOW_THREADS

  do {

    /* Get the inner loop data/stride/count values */
    npy_intp stride = *strideptr;
    npy_intp size = *innersizeptr;

    /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
    while (size--) {

      tx = *(double *)dataptr[0];

      if (tx >= xmin && tx < xmax) {
        ix = (tx - xmin) * normx * fnx;
        count[ix] += 1.;
      }

      dataptr[0] += stride;
    }

  } while (iternext(iter));

  Py_END_ALLOW_THREADS

  NpyIter_Deallocate(iter);

  /* Clean up. */
  Py_DECREF(x_array);

  return count_obj;
}

static PyObject *_histogram2d(PyObject *self, PyObject *args) {

  long n;
  int ix, iy, nx, ny;
  double xmin, xmax, tx, fnx, normx, ymin, ymax, ty, fny, normy;
  PyObject *x_obj, *y_obj, *count_obj;
  PyArrayObject *x_array, *y_array, *count_array, *arrays[2];
  npy_intp dims[2];
  double *count;
  NpyIter *iter;
  NpyIter_IterNextFunc *iternext;
  char **dataptr;
  npy_intp *strideptr, *innersizeptr;
  PyArray_Descr *dtypes[] = {PyArray_DescrFromType(NPY_DOUBLE), PyArray_DescrFromType(NPY_DOUBLE)};
  npy_uint32 op_flags[] = {NPY_ITER_READONLY, NPY_ITER_READONLY};

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOiddidd", &x_obj, &y_obj, &nx, &xmin, &xmax, &ny, &ymin, &ymax)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing input");
    return NULL;
  }

  /* Interpret the input objects as `numpy` arrays. */
  x_array = (PyArrayObject *)PyArray_FROM_O(x_obj);
  y_array = (PyArrayObject *)PyArray_FROM_O(y_obj);

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
  count_obj = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  if (count_obj == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't build output array");
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_XDECREF(count_obj);
    return NULL;
  }

  count_array = (PyArrayObject *)count_obj;

  PyArray_FILLWBYTE(count_array, 0);

  if (n == 0) {
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    return count_obj;
  }

  arrays[0] = x_array;
  arrays[1] = y_array;
  iter = NpyIter_AdvancedNew(2, arrays,
                             NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED,
                             NPY_KEEPORDER, NPY_SAFE_CASTING, op_flags, dtypes,
                             -1, NULL, NULL, 0);
  if (iter == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator");
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(count_obj);
    Py_DECREF(count_array);
    return NULL;
  }

  /*
   * The iternext function gets stored in a local variable
   * so it can be called repeatedly in an efficient manner.
   */
  iternext = NpyIter_GetIterNext(iter, NULL);
  if (iternext == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator");
    NpyIter_Deallocate(iter);
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(count_obj);
    Py_DECREF(count_array);
    return NULL;
  }

  /* The location of the data pointer which the iterator may update */
  dataptr = NpyIter_GetDataPtrArray(iter);

  /* The location of the stride which the iterator may update */
  strideptr = NpyIter_GetInnerStrideArray(iter);

  /* The location of the inner loop size which the iterator may update */
  innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

  /* Pre-compute variables for efficiency in the histogram calculation */
  fnx = nx;
  fny = ny;
  normx = 1. / (xmax - xmin);
  normy = 1. / (ymax - ymin);

  /* Get C array for output array */
  count = (double *)PyArray_DATA(count_array);

  Py_BEGIN_ALLOW_THREADS

  do {

    /* Get the inner loop data/stride/count values */
    npy_intp stride = *strideptr;
    npy_intp size = *innersizeptr;

    /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
    while (size--) {

      tx = *(double *)dataptr[0];
      ty = *(double *)dataptr[1];

      if (tx >= xmin && tx < xmax && ty >= ymin && ty < ymax) {
        ix = (tx - xmin) * normx * fnx;
        iy = (ty - ymin) * normy * fny;
        count[iy + ny * ix] += 1.;
      }

      dataptr[0] += stride;
      dataptr[1] += stride;
    }

  } while (iternext(iter));

  Py_END_ALLOW_THREADS

  NpyIter_Deallocate(iter);

  /* Clean up. */
  Py_DECREF(x_array);
  Py_DECREF(y_array);

  return count_obj;
}

static PyObject *_histogram1d_weighted(PyObject *self, PyObject *args) {

  long n;
  int ix, nx;
  double xmin, xmax, tx, tw, fnx, normx;
  PyObject *x_obj, *w_obj, *count_obj;
  PyArrayObject *x_array, *w_array, *count_array, *arrays[2];
  npy_intp dims[1];
  double *count;
  NpyIter *iter;
  NpyIter_IterNextFunc *iternext;
  char **dataptr;
  npy_intp *strideptr, *innersizeptr;
  PyArray_Descr *dtypes[] = {PyArray_DescrFromType(NPY_DOUBLE), PyArray_DescrFromType(NPY_DOUBLE)};
  npy_uint32 op_flags[] = {NPY_ITER_READONLY, NPY_ITER_READONLY};

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOidd", &x_obj, &w_obj, &nx, &xmin, &xmax)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing input");
    return NULL;
  }

  /* Interpret the input objects as `numpy` arrays. */
  x_array = (PyArrayObject *)PyArray_FROM_O(x_obj);
  w_array = (PyArrayObject *)PyArray_FROM_O(w_obj);

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
  count_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  if (count_obj == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't build output array");
    Py_DECREF(x_array);
    Py_DECREF(w_array);
    Py_XDECREF(count_obj);
    return NULL;
  }

  count_array = (PyArrayObject *)count_obj;

  PyArray_FILLWBYTE(count_array, 0);

  if (n == 0) {
    Py_DECREF(x_array);
    Py_DECREF(w_array);
    return count_obj;
  }

  arrays[0] = x_array;
  arrays[1] = w_array;
  iter = NpyIter_AdvancedNew(2, arrays,
                             NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED,
                             NPY_KEEPORDER, NPY_SAFE_CASTING, op_flags, dtypes,
                             -1, NULL, NULL, 0);
  if (iter == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator");
    Py_DECREF(x_array);
    Py_DECREF(w_array);
    Py_DECREF(count_obj);
    Py_DECREF(count_array);
    return NULL;
  }

  /*
   * The iternext function gets stored in a local variable
   * so it can be called repeatedly in an efficient manner.
   */
  iternext = NpyIter_GetIterNext(iter, NULL);
  if (iternext == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator");
    NpyIter_Deallocate(iter);
    Py_DECREF(x_array);
    Py_DECREF(w_array);
    Py_DECREF(count_obj);
    Py_DECREF(count_array);
    return NULL;
  }

  /* The location of the data pointer which the iterator may update */
  dataptr = NpyIter_GetDataPtrArray(iter);

  /* The location of the stride which the iterator may update */
  strideptr = NpyIter_GetInnerStrideArray(iter);

  /* The location of the inner loop size which the iterator may update */
  innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

  /* Pre-compute variables for efficiency in the histogram calculation */
  fnx = nx;
  normx = 1. / (xmax - xmin);

  /* Get C array for output array */
  count = (double *)PyArray_DATA(count_array);

  Py_BEGIN_ALLOW_THREADS

  do {

    /* Get the inner loop data/stride/count values */
    npy_intp stride = *strideptr;
    npy_intp size = *innersizeptr;

    /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
    while (size--) {

      tx = *(double *)dataptr[0];
      tw = *(double *)dataptr[1];

      if (tx >= xmin && tx < xmax) {
        ix = (tx - xmin) * normx * fnx;
        count[ix] += tw;
      }

      dataptr[0] += stride;
      dataptr[1] += stride;
    }

  } while (iternext(iter));

  Py_END_ALLOW_THREADS

  NpyIter_Deallocate(iter);

  /* Clean up. */
  Py_DECREF(x_array);
  Py_DECREF(w_array);

  return count_obj;
}

static PyObject *_histogram2d_weighted(PyObject *self, PyObject *args) {

  long n;
  int ix, iy, nx, ny;
  double xmin, xmax, tx, fnx, normx, ymin, ymax, ty, fny, normy, tw;
  PyObject *x_obj, *y_obj, *w_obj, *count_obj;
  PyArrayObject *x_array, *y_array, *w_array, *count_array, *arrays[3];
  npy_intp dims[2];
  double *count;
  NpyIter *iter;
  NpyIter_IterNextFunc *iternext;
  char **dataptr;
  npy_intp *strideptr, *innersizeptr;
  PyArray_Descr *dtypes[] = {PyArray_DescrFromType(NPY_DOUBLE), PyArray_DescrFromType(NPY_DOUBLE), PyArray_DescrFromType(NPY_DOUBLE)};
  npy_uint32 op_flags[] = {NPY_ITER_READONLY, NPY_ITER_READONLY, NPY_ITER_READONLY};

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOOiddidd", &x_obj, &y_obj, &w_obj, &nx, &xmin, &xmax, &ny, &ymin, &ymax)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing input");
    return NULL;
  }

  /* Interpret the input objects as `numpy` arrays. */
  x_array = (PyArrayObject *)PyArray_FROM_O(x_obj);
  y_array = (PyArrayObject *)PyArray_FROM_O(y_obj);
  w_array = (PyArrayObject *)PyArray_FROM_O(w_obj);

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
  count_obj = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  if (count_obj == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't build output array");
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(w_array);
    Py_XDECREF(count_obj);
    return NULL;
  }

  count_array = (PyArrayObject *)count_obj;

  PyArray_FILLWBYTE(count_array, 0);

  if (n == 0) {
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(w_array);
    return count_obj;
  }

  arrays[0] = x_array;
  arrays[1] = y_array;
  arrays[2] = w_array;
  iter = NpyIter_AdvancedNew(3, arrays,
                             NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED,
                             NPY_KEEPORDER, NPY_SAFE_CASTING, op_flags, dtypes,
                             -1, NULL, NULL, 0);
  if (iter == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator");
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(w_array);
    Py_DECREF(count_obj);
    Py_DECREF(count_array);
    return NULL;
  }

  /*
   * The iternext function gets stored in a local variable
   * so it can be called repeatedly in an efficient manner.
   */
  iternext = NpyIter_GetIterNext(iter, NULL);
  if (iternext == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator");
    NpyIter_Deallocate(iter);
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(w_array);
    Py_DECREF(count_obj);
    Py_DECREF(count_array);
    return NULL;
  }

  /* The location of the data pointer which the iterator may update */
  dataptr = NpyIter_GetDataPtrArray(iter);

  /* The location of the stride which the iterator may update */
  strideptr = NpyIter_GetInnerStrideArray(iter);

  /* The location of the inner loop size which the iterator may update */
  innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

  /* Pre-compute variables for efficiency in the histogram calculation */
  fnx = nx;
  fny = ny;
  normx = 1. / (xmax - xmin);
  normy = 1. / (ymax - ymin);

  /* Get C array for output array */
  count = (double *)PyArray_DATA(count_array);

  Py_BEGIN_ALLOW_THREADS

  do {

    /* Get the inner loop data/stride/count values */
    npy_intp stride = *strideptr;
    npy_intp size = *innersizeptr;

    /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
    while (size--) {

      tx = *(double *)dataptr[0];
      ty = *(double *)dataptr[1];
      tw = *(double *)dataptr[2];

      if (tx >= xmin && tx < xmax && ty >= ymin && ty < ymax) {
        ix = (tx - xmin) * normx * fnx;
        iy = (ty - ymin) * normy * fny;
        count[iy + ny * ix] += tw;
      }

      dataptr[0] += stride;
      dataptr[1] += stride;
      dataptr[2] += stride;
    }

  } while (iternext(iter));

  Py_END_ALLOW_THREADS

  NpyIter_Deallocate(iter);

  /* Clean up. */
  Py_DECREF(x_array);
  Py_DECREF(y_array);
  Py_DECREF(w_array);

  return count_obj;
}
