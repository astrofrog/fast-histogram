#!/usr/bin/env python

import os
import sys

import numpy

from setuptools import setup
from setuptools.extension import Extension

setup(use_scm_version={'write_to': os.path.join('fast_histogram', 'version.py')},
      ext_modules=[Extension("fast_histogram._histogram_core",
                             [os.path.join('fast_histogram', '_histogram_core.c')],
                             include_dirs=[numpy.get_include()])])
