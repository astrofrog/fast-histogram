#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

/* Define docstrings */
static char module_docstring[] = "Fast histogram functioins";
static char _histogram1d_docstring[] = "Compute a 1D histogram";
static char _histogram2d_docstring[] = "Compute a 2D histogram";

/* Declare the C functions here. */
static PyObject *_histogram1d(PyObject *self, PyObject *args);
static PyObject *_histogram2d(PyObject *self, PyObject *args);

/* Define the methods that will be available on the module. */
static PyMethodDef module_methods[] = {
    {"_histogram1d", _histogram1d, METH_VARARGS, _histogram1d_docstring},
    {"_histogram2d", _histogram2d, METH_VARARGS, _histogram2d_docstring},
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
    double xmin, xmax, tx, fnx, normx;
    PyObject *x_obj, *x_array, *count_array;
    npy_intp dims[1];
    double *x, *count;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "Oidd", &x_obj, &nx, &xmin, &xmax)) {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);

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
    count_array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (count_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_XDECREF(count_array);
        return NULL;
    }

    PyArray_FILLWBYTE(count_array, 0);

    /* Get pointers to the data as C-types. */

    x = (double*)PyArray_DATA(x_array);
    count = (double*)PyArray_DATA(count_array);

    fnx = nx;
    normx = 1. / (xmax - xmin);

    for(i = 0; i < n; i++) {

      tx = x[i];

      if (tx >= xmin && tx < xmax) {
          ix = (tx - xmin) * normx * fnx;
          count[ix] += 1.;
      }

    }

    /* Clean up. */
    Py_DECREF(x_array);

    return count_array;

}


static PyObject *_histogram2d(PyObject *self, PyObject *args)
{

    long i, n;
    int ix, iy, nx, ny;
    double xmin, xmax, tx, fnx, normx, ymin, ymax, ty, fny, normy;
    PyObject *x_obj, *y_obj, *x_array, *y_array, *count_array;
    npy_intp dims[2];
    double *x, *y, *count;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOiddidd", &x_obj, &y_obj, &nx, &xmin, &xmax, &ny, &ymin, &ymax)) {
        PyErr_SetString(PyExc_TypeError, "Error parsing input");
        return NULL;
    }

    /* Interpret the input objects as `numpy` arrays. */
    x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);

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

    count_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (count_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Couldn't build output array");
        Py_DECREF(x_array);
        Py_XDECREF(count_array);
        return NULL;
    }

    PyArray_FILLWBYTE(count_array, 0);

    /* Get pointers to the data as C-types. */
    x = (double*)PyArray_DATA(x_array);
    y = (double*)PyArray_DATA(y_array);
    count = (double*)PyArray_DATA(count_array);

    fnx = nx;
    fny = ny;
    normx = 1. / (xmax - xmin);
    normy = 1. / (ymax - ymin);

    for(i = 0; i < n; i++) {

      tx = x[i];
      ty = y[i];

      if (tx >= xmin && tx < xmax && ty >= ymin && ty < ymax) {
          ix = (tx - xmin) * normx * fnx;
          iy = (ty - ymin) * normy * fny;
          count[iy + ny * ix] += 1.;
      }

    }

    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    return count_array;

}
