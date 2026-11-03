#!/usr/bin/env python

import os
import sysconfig

import numpy
from setuptools import setup
from setuptools.extension import Extension

py_limited_api = not sysconfig.get_config_var("Py_GIL_DISABLED")

setup(
    use_scm_version={"write_to": os.path.join("fast_histogram", "version.py")},
    ext_modules=[
        Extension(
            "fast_histogram._histogram_core",
            [os.path.join("fast_histogram", "_histogram_core.c")],
            py_limited_api=py_limited_api,
            define_macros=[("Py_LIMITED_API", "0x030a00f0")] if py_limited_api else [],
            include_dirs=[numpy.get_include()],
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp310"}} if py_limited_api else {},
)
