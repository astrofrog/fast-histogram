import os

import numpy as np
from setuptools import setup
from setuptools.extension import Extension


# MSVC vectorization report: '/Qvec-report:2'
# GCC vectorization report: '-fopt-info-vec', '-fopt-info-vec-missed', '-fdiagnostics-color=always'
if os.name == 'nt':
      extra_compile_args = ['/Qvec-report:2']
else:
      extra_compile_args = ['-fopt-info-vec', '-fopt-info-vec-missed', '-fdiagnostics-color=always']

extensions = [Extension("fast_histogram._histogram_core",
                       [os.path.join('fast_histogram', '_histogram_core.c')],
                       include_dirs=[np.get_include()],
                       extra_compile_args=extra_compile_args
                       )]


setup(name='fast-histogram',
      version='0.4.dev0',
      description='Fast simple 1D and 2D histograms',
      long_description=open('README.rst').read(),
      install_requires=['numpy'],
      author='Thomas Robitaille',
      author_email='thomas.robitaille@gmail.com',
      license='BSD',
      url='https://github.com/astrofrog/fast-histogram',
      packages=['fast_histogram', 'fast_histogram.tests'],
      ext_modules=extensions)
