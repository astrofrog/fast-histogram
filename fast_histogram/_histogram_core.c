#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define Py_LIMITED_API 0x030900f0

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

/* Define docstrings */
static char module_docstring[] = "Fast histogram functioins";
static char _histogram1d_docstring[] = "Compute a 1D histogram";
static char _histogram2d_docstring[] = "Compute a 2D histogram";
static char _histogramdd_docstring[] = "Compute a histogram with arbitrary dimensionality";
static char _histogram1d_weighted_docstring[] = "Compute a weighted 1D histogram";
static char _histogram2d_weighted_docstring[] = "Compute a weighted 2D histogram";
static char _histogramdd_weighted_docstring[] = "Compute a weighted histogram with arbitrary dimensionality";

/* Declare the C functions here. */
static PyObject *_histogram1d(PyObject *self, PyObject *args);
static PyObject *_histogram2d(PyObject *self, PyObject *args);
static PyObject *_histogramdd(PyObject *self, PyObject *args);
static PyObject *_histogram1d_weighted(PyObject *self, PyObject *args);
static PyObject *_histogram2d_weighted(PyObject *self, PyObject *args);
static PyObject *_histogramdd_weighted(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"_histogram1d", _histogram1d, METH_VARARGS, _histogram1d_docstring},
    {"_histogram2d", _histogram2d, METH_VARARGS, _histogram2d_docstring},
    {"_histogramdd", _histogramdd, METH_VARARGS, _histogramdd_docstring},
    {"_histogram1d_weighted", _histogram1d_weighted, METH_VARARGS, _histogram1d_weighted_docstring},
    {"_histogram2d_weighted", _histogram2d_weighted, METH_VARARGS, _histogram2d_weighted_docstring},
    {"_histogramdd_weighted", _histogramdd_weighted, METH_VARARGS, _histogramdd_weighted_docstring},
    {NULL, NULL, 0, NULL}
};

/* This is the function that is called on import. */

#define MOD_ERROR_VAL NULL
#define MOD_SUCCESS_VAL(val) val
#define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#define MOD_DEF(ob, name, doc, methods) \
        static struct PyModuleDef moduledef = { \
        PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
        ob = PyModule_Create(&moduledef);

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
  normx = fnx / (xmax - xmin);

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
        ix = (tx - xmin) * normx;
        if(ix == nx) ix -= 1;
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
  normx = fnx / (xmax - xmin);
  normy = fny / (ymax - ymin);

  /* Get C array for output array */
  count = (double *)PyArray_DATA(count_array);

  Py_BEGIN_ALLOW_THREADS

  do {

    /* Get the inner loop data/stride/count values */
    npy_intp stride0 = strideptr[0];
    npy_intp stride1 = strideptr[1];
    npy_intp size = *innersizeptr;

    /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
    while (size--) {

      tx = *(double *)dataptr[0];
      ty = *(double *)dataptr[1];

      if (tx >= xmin && tx < xmax && ty >= ymin && ty < ymax) {
        ix = (tx - xmin) * normx;
        iy = (ty - ymin) * normy;
        if(ix == nx) ix -= 1;
        if(iy == ny) iy -= 1;
        count[iy + ny * ix] += 1.;
      }

      dataptr[0] += stride0;
      dataptr[1] += stride1;
    }

  } while (iternext(iter));

  Py_END_ALLOW_THREADS

  NpyIter_Deallocate(iter);

  /* Clean up. */
  Py_DECREF(x_array);
  Py_DECREF(y_array);

  return count_obj;
}

static PyObject *_histogramdd(PyObject *self, PyObject *args) {

  long n;
  int ndim, sample_parsing_success;
  PyObject *sample_obj, *range_obj, *bins_obj,  *count_obj;
  PyArrayObject **arrays, *range, *bins, *count_array;
  npy_intp *dims;
  double *count, *range_c, *fndim, *norms;
  double tx;
  int bin_idx, local_bin_idx, in_range, *stride;
  // using xmin and xmax for all dimensions
  double xmin, xmax;
  NpyIter *iter;
  NpyIter_IterNextFunc *iternext;
  char **dataptr;
  npy_intp *strideptr, *innersizeptr;
  PyArray_Descr *dtype;
  PyArray_Descr **dtypes;
  npy_uint32 *op_flags;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOO", &sample_obj, &bins_obj, &range_obj)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing input");
    return NULL;
  }

  ndim = (int)PyTuple_Size(sample_obj);

  /* Interpret the input objects as `numpy` arrays. */
  arrays = (PyArrayObject **)malloc(sizeof(PyArrayObject *) * ndim);
  sample_parsing_success = 1;
  for (int i = 0; i < ndim; i++){
    arrays[i] = (PyArrayObject *)PyArray_FROM_O(PyTuple_GetItem(sample_obj, i));
    if (arrays[i] == NULL){
      sample_parsing_success = 0;
    }
  }

  dtype = PyArray_DescrFromType(NPY_DOUBLE);
  range = (PyArrayObject *)PyArray_FromAny(range_obj, dtype, 2, 2, NPY_ARRAY_IN_ARRAY, NULL);
  dtype = PyArray_DescrFromType(NPY_INTP);
  bins = (PyArrayObject *)PyArray_FromAny(bins_obj, dtype, 1, 1, NPY_ARRAY_IN_ARRAY, NULL);

  /* If that didn't work, throw an `Exception`. */
  if (range == NULL || bins == NULL || !sample_parsing_success) {
    PyErr_SetString(PyExc_TypeError, "Couldn't parse at least one of the input arrays."
    " `range` must be passed as a 2D ndarray of type `np.double`,"
    " `bins` must be passed as a 1D ndarray of type `np.intp`.");
    for (int i = 0; i < ndim; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_XDECREF(range);
    Py_XDECREF(bins);
    free(arrays);
    return NULL;
  }

  /* How many data points are there? */
  n = (long)PyArray_DIM(arrays[0], 0);
  if (ndim > 1){
    for (int i = 0; i < ndim; i++){
      if (!((long)PyArray_DIM(arrays[i], 0) == n)){
        PyErr_SetString(PyExc_RuntimeError, "Lengths of sample arrays do not match.");
        for (int j = 0; j < ndim; j++){
          Py_XDECREF(arrays[i]);
        }
        Py_XDECREF(range);
        Py_XDECREF(bins);
        free(arrays);
        return NULL;
      }
    }
  }

  /* copy the content of `bins` into `dims` */
  dtype = PyArray_DescrFromType(NPY_INTP);
  iter = NpyIter_New(bins, NPY_ITER_READONLY, NPY_CORDER, NPY_SAFE_CASTING, dtype);
  if (iter == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator over binning.");
    for (int i = 0; i < ndim; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_XDECREF(range);
    Py_XDECREF(bins);
    free(arrays);
    return NULL;
  }
  iternext = NpyIter_GetIterNext(iter, NULL);
  if (iternext == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iteration function over binning.");
    for (int i = 0; i < ndim; i++){
      Py_XDECREF(arrays[i]);
    }
    NpyIter_Deallocate(iter);
    Py_XDECREF(range);
    Py_XDECREF(bins);
    free(arrays);
    return NULL;
  }
  dataptr = NpyIter_GetDataPtrArray(iter);
  dims = (npy_intp *)malloc(sizeof(npy_intp) * ndim);
  int i = 0;
  do{
    dims[i] = *(npy_intp *)dataptr[0];
    i++;
  } while (iternext(iter));
  NpyIter_Deallocate(iter);

  /* build the output array */
  count_obj = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
  if (count_obj == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't build output array");
    for (int i = 0; i < ndim; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_XDECREF(range);
    Py_XDECREF(bins);
    Py_XDECREF(count_obj);
    free(arrays);
    free(dims);
    return NULL;
  }

  count_array = (PyArrayObject *)count_obj;

  PyArray_FILLWBYTE(count_array, 0);

  if (n == 0) {
    for (int i = 0; i < ndim; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_XDECREF(range);
    Py_XDECREF(bins);
    free(arrays);
    free(dims);
    return count_obj;
  }

  /* copy the content of the numpy array `ranges` into a simple C array */
  // This just makes is easier to access the values later in the loop.
  range_c = (double *)malloc(sizeof(double) * ndim * 2);
  dtype = PyArray_DescrFromType(NPY_DOUBLE);
  iter = NpyIter_New(range, NPY_ITER_READONLY, NPY_CORDER, NPY_SAFE_CASTING, dtype);
  if (iter == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator over range. This needs to be passed as type `numpy.double`.");
    for (int i = 0; i < ndim; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_XDECREF(range);
    Py_XDECREF(bins);
    Py_DECREF(count_obj);
    free(arrays);
    free(dims);
    free(range_c);
    return NULL;
  }
  iternext = NpyIter_GetIterNext(iter, NULL);
  if (iternext == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iteration function over range. This needs to be passed as type `numpy.double`.");
    NpyIter_Deallocate(iter);
    for (int i = 0; i < ndim; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_XDECREF(range);
    Py_XDECREF(bins);
    Py_DECREF(count_obj);
    free(arrays);
    free(dims);
    free(range_c);
    return NULL;
  }

  dataptr = NpyIter_GetDataPtrArray(iter);
  i = 0;
  do{
    range_c[i] = *(double *)dataptr[0];
    i++;
  } while (iternext(iter));
  NpyIter_Deallocate(iter);

  /* now we pre-compute the bin normalizations for all dimensions */
  fndim = (double *)malloc(sizeof(double) * ndim);
  norms = (double *)malloc(sizeof(double) * ndim);
  for (int j = 0; j < ndim; j++){
    fndim[j] = (double)dims[j];
    norms[j] = fndim[j] / (range_c[j * 2 + 1] - range_c[j * 2]);
  }

  dtypes = (PyArray_Descr **)malloc(sizeof(PyArray_Descr *) * ndim);
  op_flags = (npy_uint32 *)malloc(sizeof(npy_uint32) * ndim);
  for (int i = 0; i < ndim; i++){
    dtypes[i] = PyArray_DescrFromType(NPY_DOUBLE);
    op_flags[i] = NPY_ITER_READONLY;
  }
  iter = NpyIter_AdvancedNew(ndim, arrays,
                             NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED,
                             NPY_KEEPORDER, NPY_SAFE_CASTING, op_flags, dtypes,
                             -1, NULL, NULL, 0);
  if (iter == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator");
    for (int i = 0; i < ndim; i++){
     Py_XDECREF(arrays[i]);
    }
    Py_DECREF(count_obj);
    free(arrays);
    free(dims);
    free(range_c);
    free(fndim);
    free(norms);
    free(dtypes);
    free(op_flags);
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
    for (int i = 0; i < ndim; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_DECREF(count_obj);
    free(arrays);
    free(dims);
    free(range_c);
    free(fndim);
    free(norms);
    free(dtypes);
    free(op_flags);
    return NULL;
  }

  /* The location of the data pointer which the iterator may update */
  dataptr = NpyIter_GetDataPtrArray(iter);

  /* The location of the stride which the iterator may update */
  strideptr = NpyIter_GetInnerStrideArray(iter);

  /* The location of the inner loop size which the iterator may update */
  innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

  /* Get C array for output array */
  count = (double *)PyArray_DATA(count_array);

  /* Pre-compute index stride */
  //  We comput the strides for the bin index for each dimension. The desired
  //  behavior is this:
  //  1D: bin_idx = ix
  //      --> stride = {1}
  //  2D: bin_idx = ny * ix + iy
  //      --> stride = {ny, 1}
  //  3D: bin_idx = nz * ny * ix + nz * iy + iz
  //      --> stride = {nz * ny, nz, 1}
  //  ... and so on for higher dimensions.
  //  Notice how the order of multiplication requires that we step through the
  //  dimensions backwards.
  stride = (int *)malloc(sizeof(int) * ndim);
  for (int i = 0; i < ndim; i++){
    stride[i] = 1;
  }
  if (ndim > 1){
    for (int i = ndim - 1; i > 0; i--){
      stride[i - 1] = stride[i] * (int)dims[i];
    }
  }

  Py_BEGIN_ALLOW_THREADS

  do {
    /* Get the inner loop data/stride/count values */
    npy_intp size = *innersizeptr;
    /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
    while (size--) {
      bin_idx = 0;
      in_range = 1;
      for (int i = 0; i < ndim; i++){
        xmin = range_c[i * 2];
        xmax = range_c[i * 2 + 1];
        tx = *(double *)dataptr[i];
        dataptr[i] += strideptr[i];
        if (tx < xmin || tx >= xmax){
          in_range = 0;
        } else {
          local_bin_idx = (tx - xmin) * norms[i];
          if(local_bin_idx == dims[i]) local_bin_idx -= 1;
          bin_idx += stride[i] * local_bin_idx;
        }
      }
      if (in_range){
count[bin_idx] += 1;
      }
    }

  } while (iternext(iter));

  Py_END_ALLOW_THREADS

  NpyIter_Deallocate(iter);
  for (int i = 0; i < ndim; i++){
    Py_XDECREF(arrays[i]);
  }
  Py_XDECREF(range);
  Py_XDECREF(bins);
  free(arrays);
  free(dims);
  free(range_c);
  free(fndim);
  free(norms);
  free(dtypes);
  free(op_flags);
  free(stride);
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
  normx = fnx / (xmax - xmin);

  /* Get C array for output array */
  count = (double *)PyArray_DATA(count_array);

  Py_BEGIN_ALLOW_THREADS

  do {

    /* Get the inner loop data/stride/count values */
    npy_intp stride0 = strideptr[0];
    npy_intp stride1 = strideptr[1];
    npy_intp size = *innersizeptr;

    /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
    while (size--) {

      tx = *(double *)dataptr[0];
      tw = *(double *)dataptr[1];

      if (tx >= xmin && tx < xmax) {
        ix = (tx - xmin) * normx;
        if(ix == nx) ix -= 1;
        count[ix] += tw;
      }

      dataptr[0] += stride0;
      dataptr[1] += stride1;
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
  normx = fnx / (xmax - xmin);
  normy = fny / (ymax - ymin);

  /* Get C array for output array */
  count = (double *)PyArray_DATA(count_array);

  Py_BEGIN_ALLOW_THREADS

  do {

    /* Get the inner loop data/stride/count values */
    npy_intp stride0 = strideptr[0];
    npy_intp stride1 = strideptr[1];
    npy_intp stride2 = strideptr[2];
    npy_intp size = *innersizeptr;

    /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
    while (size--) {

      tx = *(double *)dataptr[0];
      ty = *(double *)dataptr[1];
      tw = *(double *)dataptr[2];

      if (tx >= xmin && tx < xmax && ty >= ymin && ty < ymax) {
        ix = (tx - xmin) * normx;
        iy = (ty - ymin) * normy;
        if(ix == nx) ix -= 1;
        if(iy == ny) iy -= 1;
        count[iy + ny * ix] += tw;
      }

      dataptr[0] += stride0;
      dataptr[1] += stride1;
      dataptr[2] += stride2;
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

static PyObject *_histogramdd_weighted(PyObject *self, PyObject *args) {

  long n;
  int ndim, sample_parsing_success;
  PyObject *sample_obj, *range_obj, *bins_obj,  *count_obj, *weights_obj;
  PyArrayObject **arrays, *range, *bins, *count_array;
  npy_intp *dims;
  double *count, *range_c, *fndim, *norms;
  double tx, tw;
  int bin_idx, local_bin_idx, in_range, *stride;
  // using xmin and xmax for all dimensions
  double xmin, xmax;
  NpyIter *iter;
  NpyIter_IterNextFunc *iternext;
  char **dataptr;
  npy_intp *strideptr, *innersizeptr;
  PyArray_Descr *dtype;
  PyArray_Descr **dtypes;
  npy_uint32 *op_flags;

  /* Parse the input tuple */
  if (!PyArg_ParseTuple(args, "OOOO", &sample_obj, &bins_obj, &range_obj, &weights_obj)) {
    PyErr_SetString(PyExc_TypeError, "Error parsing input");
    return NULL;
  }

  ndim = (int)PyTuple_Size(sample_obj);

  /* Interpret the input objects as `numpy` arrays. */
  arrays = (PyArrayObject **)malloc(sizeof(PyArrayObject *) * (ndim + 1));
  sample_parsing_success = 1;
  for (int i = 0; i < ndim; i++){
    arrays[i] = (PyArrayObject *)PyArray_FROM_O(PyTuple_GetItem(sample_obj, i));
    if (arrays[i] == NULL){
      sample_parsing_success = 0;
    }
  }
  /* the last index is always the weights array */
  arrays[ndim] = (PyArrayObject *)PyArray_FROM_O(weights_obj);
  if (arrays[ndim] == NULL){
    sample_parsing_success = 0;
  }

  dtype = PyArray_DescrFromType(NPY_DOUBLE);
  range = (PyArrayObject *)PyArray_FromAny(range_obj, dtype, 2, 2, NPY_ARRAY_IN_ARRAY, NULL);
  dtype = PyArray_DescrFromType(NPY_INTP);
  bins = (PyArrayObject *)PyArray_FromAny(bins_obj, dtype, 1, 1, NPY_ARRAY_IN_ARRAY, NULL);

  /* If that didn't work, throw an `Exception`. */
  if (range == NULL || bins == NULL || !sample_parsing_success) {
    PyErr_SetString(PyExc_TypeError, "Couldn't parse at least one of the input arrays."
    " `range` must be passed as a 2D ndarray of type `np.double`,"
    " `bins` must be passed as a 1D ndarray of type `np.intp`.");
    for (int i = 0; i < ndim + 1; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_XDECREF(range);
    Py_XDECREF(bins);
    free(arrays);
    return NULL;
  }

  /* How many data points are there? */
  n = (long)PyArray_DIM(arrays[0], 0);
  for (int i = 0; i < ndim + 1; i++){
    if (!((long)PyArray_DIM(arrays[i], 0) == n)){
      PyErr_SetString(PyExc_RuntimeError, "Lengths of sample and/or weight arrays do not match.");
      for (int j = 0; j < ndim + 1; j++){
        Py_XDECREF(arrays[j]);
      }
      Py_XDECREF(range);
      Py_XDECREF(bins);
      free(arrays);
      return NULL;
    }
  }

  /* copy the content of `bins` into `dims` */
  dtype = PyArray_DescrFromType(NPY_INTP);
  iter = NpyIter_New(bins, NPY_ITER_READONLY, NPY_CORDER, NPY_SAFE_CASTING, dtype);
  if (iter == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator over binning.");
    for (int i = 0; i < ndim + 1; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_XDECREF(range);
    Py_XDECREF(bins);
    free(arrays);
    return NULL;
  }
  iternext = NpyIter_GetIterNext(iter, NULL);
  if (iternext == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iteration function over binning.");
    for (int i = 0; i < ndim + 1; i++){
      Py_XDECREF(arrays[i]);
    }
    NpyIter_Deallocate(iter);
    Py_XDECREF(range);
    Py_XDECREF(bins);
    free(arrays);
    return NULL;
  }
  dataptr = NpyIter_GetDataPtrArray(iter);
  dims = (npy_intp *)malloc(sizeof(npy_intp) * ndim);
  int i = 0;
  do{
    dims[i] = *(npy_intp *)dataptr[0];
    i++;
  } while (iternext(iter));
  NpyIter_Deallocate(iter);

  /* build the output array */
  count_obj = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
  if (count_obj == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't build output array");
    for (int i = 0; i < ndim + 1; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_XDECREF(range);
    Py_XDECREF(bins);
    Py_XDECREF(count_obj);
    free(arrays);
    free(dims);
    return NULL;
  }

  count_array = (PyArrayObject *)count_obj;

  PyArray_FILLWBYTE(count_array, 0);

  if (n == 0) {
    for (int i = 0; i < ndim + 1; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_XDECREF(range);
    Py_XDECREF(bins);
    free(arrays);
    free(dims);
    return count_obj;
  }

  /* copy the content of the numpy array `ranges` into a simple C array */
  // This just makes is easier to access the values later in the loop.
  range_c = (double *)malloc(sizeof(double) * ndim * 2);
  dtype = PyArray_DescrFromType(NPY_DOUBLE);
  iter = NpyIter_New(range, NPY_ITER_READONLY, NPY_CORDER, NPY_SAFE_CASTING, dtype);
  if (iter == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator over range. This needs to be passed as type `numpy.double`.");
    for (int i = 0; i < ndim + 1; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_XDECREF(range);
    Py_XDECREF(bins);
    Py_DECREF(count_obj);
    free(arrays);
    free(dims);
    free(range_c);
    return NULL;
  }
  iternext = NpyIter_GetIterNext(iter, NULL);
  if (iternext == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iteration function over range. This needs to be passed as type `numpy.double`.");
    NpyIter_Deallocate(iter);
    for (int i = 0; i < ndim + 1; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_XDECREF(range);
    Py_XDECREF(bins);
    Py_DECREF(count_obj);
    free(arrays);
    free(dims);
    free(range_c);
    return NULL;
  }

  dataptr = NpyIter_GetDataPtrArray(iter);
  i = 0;
  do{
    range_c[i] = *(double *)dataptr[0];
    i++;
  } while (iternext(iter));
  NpyIter_Deallocate(iter);

  /* now we pre-compute the bin normalizations for all dimensions */
  fndim = (double *)malloc(sizeof(double) * ndim);
  norms = (double *)malloc(sizeof(double) * ndim);
  for (int j = 0; j < ndim; j++){
    fndim[j] = (double)dims[j];
    norms[j] = fndim[j] / (range_c[j * 2 + 1] - range_c[j * 2]);
  }

  dtypes = (PyArray_Descr **)malloc(sizeof(PyArray_Descr *) * (ndim + 1));
  op_flags = (npy_uint32 *)malloc(sizeof(npy_uint32) * (ndim + 1));
  for (int i = 0; i < ndim + 1; i++){
    dtypes[i] = PyArray_DescrFromType(NPY_DOUBLE);
    op_flags[i] = NPY_ITER_READONLY;
  }
  iter = NpyIter_AdvancedNew(ndim + 1, arrays,
                             NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED,
                             NPY_KEEPORDER, NPY_SAFE_CASTING, op_flags, dtypes,
                             -1, NULL, NULL, 0);
  if (iter == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "Couldn't set up iterator");
    for (int i = 0; i < ndim + 1; i++){
     Py_XDECREF(arrays[i]);
    }
    Py_DECREF(count_obj);
    free(arrays);
    free(dims);
    free(range_c);
    free(fndim);
    free(norms);
    free(dtypes);
    free(op_flags);
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
    for (int i = 0; i < ndim + 1; i++){
      Py_XDECREF(arrays[i]);
    }
    Py_DECREF(count_obj);
    free(arrays);
    free(dims);
    free(range_c);
    free(fndim);
    free(norms);
    free(dtypes);
    free(op_flags);
    return NULL;
  }

  /* The location of the data pointer which the iterator may update */
  dataptr = NpyIter_GetDataPtrArray(iter);

  /* The location of the stride which the iterator may update */
  strideptr = NpyIter_GetInnerStrideArray(iter);

  /* The location of the inner loop size which the iterator may update */
  innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

  /* Get C array for output array */
  count = (double *)PyArray_DATA(count_array);

  /* Pre-compute index stride */
  //  We comput the strides for the bin index for each dimension. The desired
  //  behavior is this:
  //  1D: bin_idx = ix
  //      --> stride = {1}
  //  2D: bin_idx = ny * ix + iy
  //      --> stride = {ny, 1}
  //  3D: bin_idx = nz * ny * ix + nz * iy + iz
  //      --> stride = {nz * ny, nz, 1}
  //  ... and so on for higher dimensions.
  //  Notice how the order of multiplication requires that we step through the
  //  dimensions backwards.
  stride = (int *)malloc(sizeof(int) * ndim);
  for (int i = 0; i < ndim; i++){
    stride[i] = 1;
  }
  if (ndim > 1){
    for (int i = ndim - 1; i > 0; i--){
      stride[i - 1] = stride[i] * (int)dims[i];
    }
  }

  Py_BEGIN_ALLOW_THREADS

  do {
    /* Get the inner loop data/stride/count values */
    npy_intp size = *innersizeptr;
    /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
    while (size--) {
      bin_idx = 0;
      in_range = 1;
      for (int i = 0; i < ndim; i++){
        xmin = range_c[i * 2];
        xmax = range_c[i * 2 + 1];
        tx = *(double *)dataptr[i];
        dataptr[i] += strideptr[i];
        if (tx < xmin || tx >= xmax){
          in_range = 0;
        } else {
          local_bin_idx = (tx - xmin) * norms[i];
          if(local_bin_idx == dims[i]) local_bin_idx -= 1;
          bin_idx += stride[i] * local_bin_idx;
        }
      }
      tw = *(double *)dataptr[ndim];
      dataptr[ndim] += strideptr[ndim];
      if (in_range){
        count[bin_idx] += tw;
      }

    }

  } while (iternext(iter));

  Py_END_ALLOW_THREADS

  NpyIter_Deallocate(iter);
  for (int i = 0; i < ndim + 1; i++){
    Py_XDECREF(arrays[i]);
  }
  Py_XDECREF(range);
  Py_XDECREF(bins);
  free(arrays);
  free(dims);
  free(range_c);
  free(fndim);
  free(norms);
  free(dtypes);
  free(op_flags);
  free(stride);
  return count_obj;
}
