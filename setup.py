import os

import numpy as np
from setuptools import setup
from setuptools.extension import Extension

if os.path.exists(os.path.join('fast_histogram', 'histogram_cython.pyx')):

    extensions = [Extension("fast_histogram.histogram_cython",
                            [os.path.join('fast_histogram', 'histogram_cython.pyx')],
                            include_dirs=[np.get_include()])]

    from Cython.Build import cythonize
    extensions = cythonize(extensions)

    install_requires = ['numpy', 'Cython']

else:

    extensions = [Extension("fast_histogram.histogram_cython",
                            [os.path.join('fast_histogram', 'histogram_cython.c')],
                            include_dirs=[np.get_include()])]

    install_requires = ['numpy', 'Cython']

setup(name='fast-histogram',
      version='0.1.dev0',
      install_requires=install_requires,
      author='Thomas Robitaille',
      author_email='thomas.robitaille@gmail.com',
      license='BSD',
      url='https://github.com/astrofrog/fastogram',
      packages=['fast_histogram'],
      ext_modules=extensions)
